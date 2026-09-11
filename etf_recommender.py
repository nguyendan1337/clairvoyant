"""Discover, enrich, score, and publish high-momentum ETF recommendations.

Yahoo universe discovery and market data use yfinance. Gemini adds current
context to the quantitative results and the final output is rendered to HTML.
"""

import os
import re
import time
import random
import json
import hashlib
import html
import yaml
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types
from tqdm import tqdm
from yfinance import ETFQuery


YF_CACHE_FILE = 'caches/etf_yf_cache.json'
YF_CACHE_EXPIRY_DAYS = 1
TOP_QVM_CACHE_FILE = 'caches/top_qvm_etfs_cache.pkl'
TOP_QVM_CACHE_EXPIRY_HOURS = 6
TOP_QVM_CACHE_VERSION = 13
SCRIPT_DIR = Path(__file__).resolve().parent


def initialize_gemini_client():
    api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent / '.env'
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        raise ValueError('GEMINI_KEY not found in environment or .env')

    return genai.Client(api_key=api_key)


def build_gemini_config(
    thinking_budget,
    enable_search=True,
    response_mime_type='application/json',
    max_output_tokens=None,
):
    tools = None
    if enable_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
    kwargs = {
        'tools': tools,
        'temperature': 0,
        'thinking_config': types.ThinkingConfig(
            thinking_budget=int(thinking_budget),
        ),
    }
    if response_mime_type:
        kwargs['response_mime_type'] = response_mime_type
    if max_output_tokens:
        kwargs['max_output_tokens'] = int(max_output_tokens)
    return types.GenerateContentConfig(
        **kwargs,
    )


class GeminiRequestBudget:
    def __init__(self, total, research, reserved_summary=1, release_summary_for_research=False):
        self.total = int(total)
        self.research_limit = int(research)
        self.reserved_summary = int(reserved_summary)
        self.release_summary_for_research = bool(release_summary_for_research)
        self.total_used = 0
        self.research_used = 0
        self.summary_used = 0
        self.api_attempts = 0
        self.research_api_attempts = 0
        self.summary_api_attempts = 0
        self.context_api_attempts = 0

    def can_reserve(self, category):
        if self.total_used >= self.total:
            return False
        if category == 'research':
            required_reserve = 0 if self.release_summary_for_research else self.reserved_summary
            return (
                self.research_used < self.research_limit
                and self.total - self.total_used > required_reserve
            )
        return True

    def reserve(self, category):
        if not self.can_reserve(category):
            required_reserve = 0 if self.release_summary_for_research else self.reserved_summary
            if category == 'research' and self.total - self.total_used <= required_reserve:
                raise RuntimeError('Only the reserved summary call remains.')
            if category == 'research':
                raise RuntimeError('ETF research call budget exhausted.')
            raise RuntimeError('Gemini call budget exhausted.')
        if category == 'research':
            self.research_used += 1
        elif category == 'summary':
            self.summary_used += 1
        self.total_used += 1

    def record_api_attempt(self, category):
        self.api_attempts += 1
        if category == 'research':
            self.research_api_attempts += 1
        elif category == 'summary':
            self.summary_api_attempts += 1
        else:
            self.context_api_attempts += 1


def parse_json_response(text):
    if not text:
        raise ValueError('Empty response from Gemini.')
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        starts = [p for p in (cleaned.find('{'), cleaned.find('[')) if p >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(cleaned.rfind('}'), cleaned.rfind(']'))
        if end < start:
            raise
        return json.loads(cleaned[start:end + 1])


def extract_partial_etf_results(text):
    """Recover complete ETF result objects from a truncated JSON response."""
    decoder = json.JSONDecoder()
    results, seen = [], set()
    for match in re.finditer(r'\{', text or ''):
        try:
            value, _ = decoder.raw_decode((text or '')[match.start():])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        symbol = str(value.get('symbol') or '').strip().upper()
        if not symbol or symbol in seen or 'research_status' not in value:
            continue
        seen.add(symbol)
        results.append(value)
    return results


def extract_gemini_metadata(response):
    metadata = {
        'prompt_tokens': 0,
        'tool_tokens': 0,
        'cached_tokens': 0,
        'thinking_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_queries': [],
        'source_urls': [],
        'finish_reasons': [],
        'response_text_chars': 0,
    }
    usage = getattr(response, 'usage_metadata', None)
    if usage:
        mappings = {
            'prompt_tokens': 'prompt_token_count',
            'tool_tokens': 'tool_use_prompt_token_count',
            'cached_tokens': 'cached_content_token_count',
            'thinking_tokens': 'thoughts_token_count',
            'output_tokens': 'candidates_token_count',
            'total_tokens': 'total_token_count',
        }
        for target, source in mappings.items():
            metadata[target] = int(getattr(usage, source, 0) or 0)
    seen_queries, seen_urls = set(), set()
    seen_finish_reasons = set()
    for candidate in getattr(response, 'candidates', None) or []:
        finish_reason = str(getattr(candidate, 'finish_reason', '') or '').strip()
        if finish_reason and finish_reason not in seen_finish_reasons:
            seen_finish_reasons.add(finish_reason)
            metadata['finish_reasons'].append(finish_reason)
        grounding = getattr(candidate, 'grounding_metadata', None)
        if not grounding:
            continue
        for query in getattr(grounding, 'web_search_queries', None) or []:
            query = str(query).strip()
            if query and query not in seen_queries:
                seen_queries.add(query)
                metadata['search_queries'].append(query)
        for chunk in getattr(grounding, 'grounding_chunks', None) or []:
            web = getattr(chunk, 'web', None)
            url = str(getattr(web, 'uri', '') or '').strip()
            if url and url not in seen_urls:
                seen_urls.add(url)
                metadata['source_urls'].append(url)
    return metadata


def print_gemini_metadata(stage, metadata):
    token_fields = [
        ('prompt_tokens', 'prompt_tokens'),
        ('tool_tokens', 'tool_tokens'),
        ('cached_tokens', 'cached_tokens'),
        ('thinking_tokens', 'thinking_tokens'),
        ('output_tokens', 'output_tokens'),
        ('total_tokens', 'total_tokens'),
    ]
    values = ', '.join(f'{label}={metadata[key]}' for key, label in token_fields)
    print(f'Gemini usage [{stage}]: {values}')
    if metadata['search_queries']:
        print(f'Google searches performed [{stage}]: {len(metadata["search_queries"])}')
        for query in metadata['search_queries']:
            print(f'  Search: {query}')
    print(
        f'Gemini response [{stage}]: chars={metadata.get("response_text_chars", 0)}, '
        f'finish={metadata.get("finish_reasons") or ["UNKNOWN"]}, '
        f'grounding_urls={len(metadata.get("source_urls", []))}'
    )


def is_transient_gemini_error(exc):
    message = str(exc).lower()
    return any(token in message for token in (
        '429', '503', 'resource_exhausted', 'unavailable', 'high demand',
        'deadline', 'timeout', 'temporarily', 'empty response',
        'malformed', 'jsondecode', 'expecting value', 'expecting property name',
        'unterminated string',
    ))


def is_daily_quota_error(exc):
    message = str(exc).lower()
    return any(token in message for token in (
        'generate_content_free_tier_requests',
        'generaterequestsperdayperprojectpermodel',
        'requests per day',
        'daily quota',
    ))


def call_gemini_json(
    client,
    model,
    prompt,
    config,
    stage,
    budget,
    category,
    max_attempts,
    initial_delay,
    max_delay=60,
    allow_partial=False,
):
    last_error = None
    budget.reserve(category)
    logical_request = budget.total_used
    for attempt in range(1, int(max_attempts) + 1):
        budget.record_api_attempt(category)
        print(
            f'Gemini logical request {logical_request}/{budget.total}: {stage} '
            f'({model}, API attempt {attempt}/{max_attempts}; '
            f'total API attempts={budget.api_attempts})'
        )
        try:
            response = client.models.generate_content(
                model=model,
                config=config,
                contents=prompt,
            )
            response_text = response.text
            if not response_text:
                raise ValueError('Empty response from Gemini.')
            metadata = extract_gemini_metadata(response)
            metadata['response_text_chars'] = len(response_text)
            print_gemini_metadata(stage, metadata)
            try:
                data = parse_json_response(response_text)
            except json.JSONDecodeError:
                if not allow_partial:
                    raise
                partial = extract_partial_etf_results(response_text)
                if not partial:
                    raise
                print(
                    'Malformed ETF-batch JSON; recovered complete results for: '
                    + ', '.join(str(item['symbol']).upper() for item in partial)
                )
                data = {'results': partial}
            metadata['api_attempts_used'] = attempt
            metadata['logical_request_number'] = logical_request
            return data, model, metadata
        except Exception as exc:
            last_error = exc
            print(f'Gemini {stage} attempt {attempt}/{max_attempts} failed: {exc}')
            if (
                is_daily_quota_error(exc)
                or attempt >= int(max_attempts)
                or not is_transient_gemini_error(exc)
            ):
                break
            base_delay = min(float(max_delay), float(initial_delay) * (2 ** (attempt - 1)))
            delay = max(1.0, min(float(max_delay), base_delay * random.uniform(0.75, 1.5)))
            print(f'Retrying transient Gemini failure in {delay:.1f}s...')
            time.sleep(delay)
    raise RuntimeError(f'Gemini {stage} failed: {last_error}')


def load_json_object(path, default=None):
    default = {} if default is None else default
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except Exception as exc:
        print(f'Could not load {path}: {exc}')
        return default


def save_json_object_atomic(path, value):
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = f'{path}.tmp'
        with open(temporary, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    except Exception as exc:
        print(f'Could not save {path}: {exc}')


def parse_utc_timestamp(value):
    try:
        timestamp = datetime.fromisoformat(str(value))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return timestamp.astimezone(UTC)
    except Exception:
        return None


def cache_entry_is_fresh(entry, ttl_hours):
    timestamp = (
        parse_utc_timestamp(entry.get('timestamp') or entry.get('created_at'))
        if isinstance(entry, dict) else None
    )
    return bool(timestamp and datetime.now(UTC) - timestamp < timedelta(hours=ttl_hours))


def stable_json_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def update_html_page(
    final_recommendations,
    df_html_table,
    template_name,
    display_page,
    model_used,
):
    table_match = re.search(
        r'(<table.*?</table>)',
        final_recommendations,
        flags=re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r'(<div[^>]*class=["\']summary["\'][^>]*>.*?</div>)',
        final_recommendations,
        flags=re.DOTALL | re.IGNORECASE,
    )
    gemini_table_html = table_match.group(1).strip() if table_match else ''
    gemini_summary = summary_match.group(1).strip() if summary_match else ''

    if not gemini_table_html and '<table' in final_recommendations:
        start = final_recommendations.find('<table')
        end = final_recommendations.find('</table>', start)
        if end != -1:
            gemini_table_html = final_recommendations[start:end + 8]
    if not gemini_summary and '<div' in final_recommendations:
        start = final_recommendations.find('<div')
        end = final_recommendations.find('</div>', start)
        if end != -1:
            gemini_summary = final_recommendations[start:end + 6]

    with open(template_name, 'r', encoding='utf-8') as f:
        template = f.read()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_output = template.replace('<!--LAST_UPDATED_HERE-->', timestamp)
    html_output = html_output.replace(
        '<!--RECOMMENDATIONS_TABLE_HERE-->',
        gemini_table_html,
    )
    html_output = html_output.replace(
        '<!--RECOMMENDATIONS_SUMMARY_HERE-->',
        gemini_summary,
    )
    html_output = html_output.replace('<!--FULL_DF_TABLE_HERE-->', df_html_table)
    html_output = html_output.replace('<!--MODEL_USED_HERE-->', model_used)
    with open(display_page, 'w', encoding='utf-8') as f:
        f.write(html_output)



def existing_page_matches_etf_content(
    display_page,
    final_recommendations,
    df_html_table,
):
    """Return True when the existing page already contains current ETF content.

    Ignore volatile presentation metadata such as the rendered timestamp and model
    label. This allows a fully cached run to preserve etf_index.html byte-for-byte
    when its recommendation table, validated summary, and full data table are
    unchanged.
    """
    if not os.path.exists(display_page):
        return False
    try:
        with open(display_page, 'r', encoding='utf-8') as handle:
            existing_page = handle.read().replace('\r\n', '\n')
    except Exception as exc:
        print(f'Could not read existing {display_page} for no-op check: {exc}')
        return False

    table_match = re.search(
        r'(<table.*?</table>)',
        final_recommendations,
        flags=re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r'(<div[^>]*class=["\']summary["\'][^>]*>.*?</div>)',
        final_recommendations,
        flags=re.DOTALL | re.IGNORECASE,
    )
    recommendation_table = (
        table_match.group(1).strip() if table_match else ''
    )
    summary_html = summary_match.group(1).strip() if summary_match else ''

    required_fragments = [
        fragment.replace('\r\n', '\n').strip()
        for fragment in (recommendation_table, summary_html, df_html_table)
        if str(fragment or '').strip()
    ]
    if len(required_fragments) < 3:
        return False
    return all(fragment in existing_page for fragment in required_fragments)


def extract_existing_summary_html(display_page):
    """Return the existing rendered summary div from a prior published page."""
    if not os.path.exists(display_page):
        return None
    try:
        with open(display_page, 'r', encoding='utf-8') as handle:
            page_html = handle.read()
    except Exception as exc:
        print(f'Could not read existing {display_page} for summary reuse: {exc}')
        return None
    match = re.search(
        r'(<div[^>]*class=["\\\']summary["\\\'][^>]*>.*?</div>)',
        page_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else None


def extract_number_with_suffix(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {'N/A', 'NONE', '-', '', '—', 'NAN'}:
        return None
    s = s.replace(',', '').replace('$', '')
    match = re.search('-?[\\d.]+', s)
    if not match:
        return None
    try:
        number = float(match.group())
    except ValueError:
        return None
    if 'T' in s:
        number *= 1000000000000.0
    elif 'B' in s:
        number *= 1000000000.0
    elif 'M' in s:
        number *= 1000000.0
    elif 'K' in s:
        number *= 1000.0
    return number


def clean_percent(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().replace(',', '').replace('+', '').replace('%', '')
    try:
        return float(s)
    except ValueError:
        return None


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def load_json_cache(path, expiry_days):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except Exception as exc:
        print(f'Could not load {path}: {exc}')
        return {}
    fresh_cache = {}
    now = datetime.now(UTC)
    for key, entry in cache.items():
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if now - timestamp < timedelta(days=expiry_days):
                fresh_cache[key] = entry
        except Exception:
            continue
    return fresh_cache


def save_json_cache(path, cache):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except Exception as exc:
        print(f'Could not save {path}: {exc}')


def load_top_qvm_cache(requested_top_n, requested_research_pool_limit):
    print(f'Checking top ETF QVM cache: {TOP_QVM_CACHE_FILE}', flush=True)
    if not os.path.exists(TOP_QVM_CACHE_FILE):
        print('Top ETF QVM cache does not exist.', flush=True)
        return None
    try:
        cached = pd.read_pickle(TOP_QVM_CACHE_FILE)
        if not isinstance(cached, dict):
            print('Invalid top ETF QVM cache format. Rebuilding.')
            return None
        if cached.get('version') != TOP_QVM_CACHE_VERSION:
            print('Top ETF QVM cache version mismatch. Rebuilding.')
            return None
        if cached.get('requested_top_n') != int(requested_top_n):
            print(
                'Top ETF QVM cache depth mismatch '
                f'(cached={cached.get("requested_top_n")}, '
                f'requested={int(requested_top_n)}). Rebuilding.'
            )
            return None
        if (
            cached.get('requested_research_pool_limit')
            != int(requested_research_pool_limit)
        ):
            print(
                'Top ETF QVM cache research-pool depth mismatch '
                f'(cached={cached.get("requested_research_pool_limit")}, '
                f'requested={int(requested_research_pool_limit)}). Rebuilding.'
            )
            return None
        created_at = cached.get('created_at')
        if not created_at:
            print('Top ETF QVM cache has no creation timestamp. Rebuilding.')
            return None
        created_time = datetime.fromisoformat(created_at)
        if created_time.tzinfo is None:
            created_time = created_time.replace(tzinfo=UTC)
        age = datetime.now(UTC) - created_time
        if age >= timedelta(hours=TOP_QVM_CACHE_EXPIRY_HOURS):
            print(f'Top ETF QVM cache is stale ({age.total_seconds() / 3600:.1f}h old).')
            return None
        df = cached.get('data')
        if not isinstance(df, pd.DataFrame):
            print('Top ETF QVM cache data is invalid. Rebuilding.')
            return None
        if df.empty:
            print('Top ETF QVM cache is empty. Rebuilding.')
            return None
        print(
            f'Using cached top {len(df)} ETF QVM candidates '
            f'({age.total_seconds() / 3600:.1f}h old).'
        )
        return df
    except Exception as exc:
        print(f'Could not load top ETF QVM cache: {exc}')
        return None


def save_top_qvm_cache(df, requested_top_n, requested_research_pool_limit):
    try:
        cache = {
            'version': TOP_QVM_CACHE_VERSION,
            'requested_top_n': int(requested_top_n),
            'requested_research_pool_limit': int(requested_research_pool_limit),
            'created_at': datetime.now(UTC).isoformat(),
            'data': df.copy(),
        }
        pd.to_pickle(cache, TOP_QVM_CACHE_FILE)
        print(f'Saved top ETF QVM cache to {TOP_QVM_CACHE_FILE}.')
    except Exception as exc:
        print(f'Could not save top ETF QVM cache: {exc}')


def fetch_etf_universe(min_52_week_change, max_retries=4, page_size=250):
    """Fetch every US-region ETF meeting the 52-week return threshold."""
    query = ETFQuery(
        'and',
        [
            ETFQuery('eq', ['region', 'us']),
            ETFQuery(
                'gte',
                ['fiftytwowkpercentchange', min_52_week_change],
            ),
        ],
    )
    rows, offset = ([], 0)
    while True:
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = yf.screen(
                    query,
                    offset=offset,
                    size=page_size,
                    sortField='fiftytwowkpercentchange',
                    sortAsc=False,
                )
                break
            except Exception as exc:
                print(f'ETF screener attempt {attempt}/{max_retries} failed at offset {offset}: {exc}')
                if attempt < max_retries:
                    time.sleep(2 + random.uniform(0, 1))
        if result is None:
            raise RuntimeError(f'Yahoo ETF screener failed at offset {offset}.')
        page = result.get('quotes') or []
        if not page:
            break
        rows.extend(page)
        print(f'Fetched {len(page)} ETF screener rows at offset {offset}.')
        offset += len(page)
        total = result.get('total')
        if len(page) < page_size or (total is not None and offset >= total):
            break
    if not rows:
        raise RuntimeError('Yahoo ETF screener returned no matching ETFs.')
    df = pd.DataFrame(rows)
    if 'symbol' not in df.columns:
        raise RuntimeError('Yahoo ETF screener response has no symbol field.')
    df['Name'] = df.get('longName', pd.Series(index=df.index, dtype=object))
    if 'shortName' in df.columns:
        df['Name'] = df['Name'].fillna(df['shortName'])
    df = df.rename(
        columns={
            'symbol': 'Symbol',
            'fiftyTwoWeekChangePercent': '52 WkChange %',
            'regularMarketPrice': 'Price',
            'averageDailyVolume3Month': 'Avg Vol (3M)',
            'trailingThreeMonthReturns': '3 MonthReturn',
        }
    )
    if '52 WkChange %' not in df.columns and 'fiftyTwoWeekChange' in df.columns:
        df = df.rename(columns={'fiftyTwoWeekChange': '52 WkChange %'})
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    df = df[df['Symbol'].str.match('^[A-Z0-9.-]+$', na=False)]
    df = df.drop_duplicates('Symbol', keep='first')
    if '52 WkChange %' in df.columns:
        df['52 WkChange %'] = pd.to_numeric(df['52 WkChange %'], errors='coerce')
        df = df[df['52 WkChange %'] >= min_52_week_change]
        df = df.sort_values('52 WkChange %', ascending=False)
    return df.reset_index(drop=True)


def apply_basic_etf_filters(df, excluded_keywords, min_52_week_change, min_3_month_return):
    df = df.copy()
    if 'Name' not in df.columns:
        df['Name'] = df['Symbol']
    numeric_cols = ['Price', '50 DayAverage', '200 DayAverage', '52 WkChange %', '3 MonthReturn']
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col in {'52 WkChange %', '3 MonthReturn'}:
            df[col] = df[col].apply(clean_percent)
        else:
            df[col] = df[col].apply(extract_number_with_suffix)
    if excluded_keywords:
        pattern = '|'.join((re.escape(keyword) for keyword in excluded_keywords))
        df = df[~df['Name'].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
    if '52 WkChange %' in df.columns:
        df = df[df['52 WkChange %'].notna() & (df['52 WkChange %'] >= min_52_week_change)]
    if '3 MonthReturn' in df.columns:
        df = df[df['3 MonthReturn'].isna() | (df['3 MonthReturn'] >= min_3_month_return)]
    return df.drop_duplicates(subset='Symbol').reset_index(drop=True)


def apply_post_metadata_mandate_filter(
    df,
    benchmark_symbols,
    excluded_patterns,
):
    """Remove obvious non-U.S.-equity mandates before scoring or research."""
    df = df.copy()
    benchmark_symbols = {str(symbol).upper() for symbol in benchmark_symbols}
    text_columns = [
        column for column in ('Name', 'Category', 'LegalType', 'FundOverview')
        if column in df.columns
    ]
    if not text_columns or not excluded_patterns:
        return df
    combined = pd.Series('', index=df.index, dtype=object)
    for column in text_columns:
        combined = combined + ' ' + df[column].fillna('').astype(str)
    excluded = pd.Series(False, index=df.index)
    for pattern in excluded_patterns:
        try:
            excluded |= combined.str.contains(
                str(pattern), case=False, na=False, regex=True
            )
        except re.error as exc:
            raise ValueError(
                f'Invalid excluded_metadata_patterns regex {pattern!r}: {exc}'
            ) from exc
    is_benchmark = df['Symbol'].astype(str).str.upper().isin(benchmark_symbols)
    removed = df[excluded & ~is_benchmark]
    if not removed.empty:
        print(
            'Excluded obvious non-U.S.-equity ETF mandates after metadata: '
            + ', '.join(removed['Symbol'].astype(str))
        )
    return df[~excluded | is_benchmark].reset_index(drop=True)


def serializable_fund_value(value, maximum_rows=10):
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        records = value.head(maximum_rows).reset_index().to_dict('records')
        return [
            {str(key): json_safe_scalar(item) for key, item in row.items()}
            for row in records
        ]
    if isinstance(value, pd.Series):
        return {
            str(key): json_safe_scalar(item)
            for key, item in value.head(maximum_rows).items()
        }
    if isinstance(value, dict):
        return {
            str(key): json_safe_scalar(item)
            for key, item in value.items()
        }
    return json_safe_scalar(value)


def json_safe_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def append_structured_fund_data(df, cache_file=YF_CACHE_FILE, delay=0.2):
    """Add best-effort holdings/exposure data for the final research universe."""
    df = df.copy()
    cache = load_json_cache(cache_file, YF_CACHE_EXPIRY_DAYS)
    fund_data_by_symbol = {}
    print(f'Fetching structured fund data for {len(df)} ETF research candidates...')
    for symbol in tqdm(df['Symbol'].astype(str).str.upper().tolist()):
        entry = cache.get(symbol, {})
        fund_data = entry.get('fund_data')
        if fund_data is None:
            fund_data = {}
            try:
                funds = yf.Ticker(symbol).funds_data
                for output_key, attribute in (
                    ('FundOverview', 'fund_overview'),
                    ('AssetClasses', 'asset_classes'),
                    ('SectorWeightings', 'sector_weightings'),
                    ('TopHoldings', 'top_holdings'),
                ):
                    try:
                        fund_data[output_key] = serializable_fund_value(
                            getattr(funds, attribute)
                        )
                    except Exception:
                        fund_data[output_key] = None
            except Exception as exc:
                print(f'Could not retrieve structured fund data for {symbol}: {exc}')
            entry['fund_data'] = fund_data
            entry.setdefault('info', {})
            entry['timestamp'] = datetime.now(UTC).isoformat()
            cache[symbol] = entry
            time.sleep(delay + random.uniform(0, 0.15))
        fund_data_by_symbol[symbol] = fund_data or {}
    save_json_cache(cache_file, cache)
    for column in ('FundOverview', 'AssetClasses', 'SectorWeightings', 'TopHoldings'):
        df[column] = df['Symbol'].map(
            lambda symbol: fund_data_by_symbol.get(str(symbol).upper(), {}).get(column)
        )
    return df


def equity_asset_weight(asset_classes):
    if not isinstance(asset_classes, dict):
        return None
    for key, value in asset_classes.items():
        normalized = re.sub(r'[^a-z]', '', str(key).lower())
        if normalized in {'stockposition', 'equity', 'equityposition', 'stocks'}:
            numeric = safe_float(value)
            if numeric is None:
                return None
            return numeric / 100 if numeric > 1.5 else numeric
    return None


def apply_structured_equity_filter(df, benchmark_symbols, minimum_weight):
    if 'AssetClasses' not in df.columns or minimum_weight <= 0:
        return df
    weights = df['AssetClasses'].apply(equity_asset_weight)
    known_non_equity = weights.notna() & weights.lt(minimum_weight)
    is_benchmark = df['Symbol'].astype(str).str.upper().isin(
        {str(symbol).upper() for symbol in benchmark_symbols}
    )
    removed = df[known_non_equity & ~is_benchmark]
    if not removed.empty:
        print(
            f'Excluded ETFs below {minimum_weight:.0%} equity assets: '
            + ', '.join(removed['Symbol'].astype(str))
        )
    return df[~known_non_equity | is_benchmark].reset_index(drop=True)


CANONICAL_ETF_SECTORS = (
    'Technology', 'Healthcare', 'Financials', 'Consumer Cyclical',
    'Consumer Defensive', 'Energy', 'Industrials', 'Basic Materials',
    'Communication Services', 'Utilities', 'Real Estate',
)
SECTOR_TEXT_RULES = (
    ('Technology', ('TECHNOLOGY', 'INFORMATION TECH', 'SEMICONDUCTOR', 'SOFTWARE')),
    ('Healthcare', ('HEALTHCARE', 'HEALTH CARE', 'BIOTECH', 'PHARMACEUTICAL')),
    ('Financials', ('FINANCIAL', 'BANK', 'INSURANCE')),
    ('Consumer Cyclical', ('CONSUMER CYCLICAL', 'CONSUMER DISCRETIONARY')),
    ('Consumer Defensive', ('CONSUMER DEFENSIVE', 'CONSUMER STAPLES')),
    ('Energy', ('ENERGY', 'OIL', 'NATURAL GAS')),
    ('Industrials', ('INDUSTRIAL', 'AEROSPACE', 'DEFENSE')),
    ('Basic Materials', ('BASIC MATERIAL', 'MATERIALS', 'MINING')),
    ('Communication Services', ('COMMUNICATION SERVICE', 'TELECOMMUNICATION')),
    ('Utilities', ('UTILITIES', 'UTILITY')),
    ('Real Estate', ('REAL ESTATE', 'REIT')),
)


def canonical_sector_from_text(value):
    text = re.sub(r'[^A-Z0-9]+', ' ', str(value or '').upper()).strip()
    for sector, keywords in SECTOR_TEXT_RULES:
        if any(keyword in text for keyword in keywords):
            return sector
    return None


def canonical_sector_from_weightings(weightings, minimum_weight=0.50):
    if not isinstance(weightings, dict):
        return None
    best_sector, best_weight = None, -1.0
    for raw_name, raw_weight in weightings.items():
        sector = canonical_sector_from_text(raw_name)
        weight = safe_float(raw_weight)
        if sector is None or weight is None:
            continue
        if weight > 1.5:
            weight /= 100.0
        if weight > best_weight:
            best_sector, best_weight = sector, weight
    return best_sector if best_weight >= float(minimum_weight) else None


def inferred_style_category(candidate):
    supplied = str(candidate.get('Category') or '').strip()
    text = ' '.join(
        str(candidate.get(field) or '')
        for field in ('Name', 'Category', 'FundOverview')
    ).upper()

    # Prefer explicit strategy/factor identities over generic Morningstar-style
    # size buckets. These distinctions matter for the two-per-group capacity rule
    # and avoid misclassifying funds such as free-cash-flow ETFs as generic
    # mid-cap value merely because a vendor category says so.
    factor_rules = (
        ('Free Cash Flow', ('FREE CASH FLOW', 'CASH FLOW')),
        ('Quality GARP', ('GARP',)),
        ('Quality', ('QUALITY FACTOR', 'QUALITY ETF', 'QUALITY INDEX')),
        ('Momentum', ('MOMENTUM',)),
        ('Dividend', ('DIVIDEND', 'DIVIDEND GROWTH')),
        ('Low Volatility', ('LOW VOLATILITY', 'MINIMUM VOLATILITY', 'MIN VOL')),
        ('Equal Weight', ('EQUAL WEIGHT', 'EQUAL-WEIGHT')),
    )
    for label, tokens in factor_rules:
        if any(token in text for token in tokens):
            return label

    size = None
    if any(token in text for token in (
        'S&P 500', 'LARGE CAP', 'LARGE-CAP', 'RUSSELL 1000', 'MSCI USA',
    )):
        size = 'Large'
    elif any(token in text for token in (
        'RUSSELL 2000', 'S&P SMALLCAP 600', 'S&P 600', 'SMALL CAP', 'SMALL-CAP',
    )):
        size = 'Small'
    elif any(token in text for token in ('MID CAP', 'MID-CAP', 'S&P 400')):
        size = 'Mid-Cap'
    style = None
    if 'VALUE' in text:
        style = 'Value'
    elif 'GROWTH' in text:
        style = 'Growth'
    elif any(token in text for token in ('BLEND', 'BROAD MARKET', 'TOTAL MARKET')):
        style = 'Blend'
    if size and style:
        return f'{size} {style}'
    if supplied and supplied.casefold() not in {'other', 'miscellaneous'}:
        return supplied
    return style or 'Other'


def classify_etf_portfolio_group(candidate, minimum_sector_weight=0.50):
    sector = canonical_sector_from_weightings(
        candidate.get('SectorWeightings'), minimum_sector_weight
    )
    if sector is None:
        sector = canonical_sector_from_text(candidate.get('Category'))
    style = inferred_style_category(candidate)
    return sector, style, sector or style


def add_etf_portfolio_classifications(df, minimum_sector_weight=0.50):
    df = df.copy()
    values = df.apply(
        lambda row: classify_etf_portfolio_group(
            row.to_dict(), minimum_sector_weight
        ), axis=1,
    )
    df['CanonicalSector'] = values.map(lambda item: item[0])
    df['StyleCategory'] = values.map(lambda item: item[1])
    df['PortfolioGroup'] = values.map(lambda item: item[2])
    return df


def download_price_history(symbols, period='13mo'):
    if not symbols:
        return {}
    print(f'Downloading ETF price history for {len(symbols)} symbols...')
    try:
        price_data = yf.download(
            symbols,
            period=period,
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f'yfinance bulk download failed: {exc}')
        return {}
    histories = {}
    for symbol in symbols:
        try:
            if len(symbols) == 1:
                close = price_data['Close'].dropna()
            else:
                if (
                    not hasattr(price_data, 'columns')
                    or symbol not in price_data.columns.get_level_values(0)
                ):
                    continue
                close = price_data[symbol]['Close'].dropna()
            if not close.empty:
                histories[symbol] = close
        except Exception:
            continue
    return histories


def return_from_history(close, periods):
    if close is None:
        return None
    if len(close) <= periods:
        return None
    try:
        return float((close.iloc[-1] / close.iloc[-periods] - 1) * 100)
    except Exception:
        return None


def compute_momentum_metrics(close):
    if close is None:
        return {}
    if len(close) < 20:
        return {}
    return {
        'Price': safe_float(close.iloc[-1]),
        '1M Return': return_from_history(close, 21),
        '3M Return': return_from_history(close, 63),
        '6M Return': return_from_history(close, 126),
        '9M Return': return_from_history(close, 189),
        '1Y Return': return_from_history(close, 252),
        'Volatility 1Y': safe_float(
            close.pct_change().std() * np.sqrt(252) * 100
        ),
        '50D Average': safe_float(close.rolling(50).mean().iloc[-1]),
        '200D Average': safe_float(close.rolling(200).mean().iloc[-1]),
    }


def append_etf_yfinance_data(
    df,
    max_info_calls=500,
    delay=0.5,
    priority_symbols=None,
):
    df = df.copy()
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    symbols = df['Symbol'].dropna().astype(str).str.upper().unique().tolist()
    histories = download_price_history(symbols, period='13mo')
    data_map = {}
    momentum_strength = {}
    print('Computing ETF momentum metrics...')
    for symbol in symbols:
        metrics = compute_momentum_metrics(histories.get(symbol))
        data_map[symbol] = metrics
        momentum_components = [
            metrics.get('3M Return'),
            metrics.get('6M Return'),
            metrics.get('9M Return'),
        ]
        valid = [value for value in momentum_components if value is not None]
        if valid:
            momentum_strength[symbol] = float(np.mean(valid))
        else:
            momentum_strength[symbol] = -np.inf
    sorted_symbols = sorted(
        momentum_strength,
        key=lambda symbol: momentum_strength[symbol],
        reverse=True,
    )
    priority_symbols = {
        str(symbol).strip().upper()
        for symbol in (priority_symbols or [])
    }
    info_limit = int(max_info_calls or 0)
    ranked_for_info = (
        sorted_symbols[:info_limit] if info_limit > 0 else sorted_symbols
    )
    selected_for_info = list(
        dict.fromkeys(list(priority_symbols) + ranked_for_info)
    )
    print(
        'Fetching yfinance metadata for '
        + (
            f'top {len(selected_for_info)} momentum candidates...'
            if info_limit > 0
            else f'all {len(selected_for_info)} filtered momentum candidates...'
        )
    )
    cache = load_json_cache(YF_CACHE_FILE, YF_CACHE_EXPIRY_DAYS)
    info_fields = [
        'longName',
        'shortName',
        'quoteType',
        'category',
        'fundFamily',
        'legalType',
        'currency',
        'country',
        'exchange',
        'totalAssets',
        'netAssets',
        'annualReportExpenseRatio',
        'netExpenseRatio',
        'expenseRatio',
        'averageVolume',
        'averageVolume10days',
        'beta3Year',
        'threeYearAverageReturn',
        'fiveYearAverageReturn',
        'ytdReturn',
        'yield',
        'trailingPE',
        'priceToBook',
        'fundInceptionDate',
    ]
    # A fund-data refresh must not make missing ticker.info metadata look valid.
    # Require at least one identity field and one scoring/liquidity field before
    # reusing cached metadata; otherwise refetch ticker.info and preserve any
    # independently cached structured fund_data.
    identity_fields = ('longName', 'shortName', 'category', 'fundFamily')
    scoring_fields = (
        'totalAssets', 'netAssets', 'annualReportExpenseRatio',
        'netExpenseRatio', 'expenseRatio', 'averageVolume',
        'averageVolume10days', 'threeYearAverageReturn',
        'fiveYearAverageReturn', 'trailingPE', 'priceToBook',
    )
    for symbol in tqdm(selected_for_info):
        try:
            entry = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
            info = entry.get('info') if isinstance(entry.get('info'), dict) else {}
            info_is_usable = (
                any(info.get(field) is not None for field in identity_fields)
                and any(info.get(field) is not None for field in scoring_fields)
            )
            if not info_is_usable:
                ticker = yf.Ticker(symbol)
                raw_info = ticker.info or {}
                info = {field: raw_info.get(field) for field in info_fields}
                entry['info'] = info
                entry['info_timestamp'] = datetime.now(UTC).isoformat()
                entry['timestamp'] = entry['info_timestamp']
                cache[symbol] = entry
                time.sleep(delay + random.uniform(0, 0.3))
            data_map[symbol].update(
                {
                    'Name': info.get('longName') or info.get('shortName'),
                    'Category': info.get('category'),
                    'FundFamily': info.get('fundFamily'),
                    'LegalType': info.get('legalType'),
                    'Country': info.get('country'),
                    'Exchange': info.get('exchange'),
                    'AUM': safe_float(
                        info.get('totalAssets') or info.get('netAssets')
                    ),
                    'ExpenseRatio': safe_float(
                        info.get('annualReportExpenseRatio')
                        or info.get('netExpenseRatio')
                        or info.get('expenseRatio')
                    ),
                    'AverageVolume': safe_float(
                        info.get('averageVolume')
                        or info.get('averageVolume10days')
                    ),
                    'Beta3Y': safe_float(info.get('beta3Year')),
                    '3Y Average Return': safe_float(
                        info.get('threeYearAverageReturn')
                    ),
                    '5Y Average Return': safe_float(
                        info.get('fiveYearAverageReturn')
                    ),
                    'YTD Return': safe_float(info.get('ytdReturn')),
                    'Yield': safe_float(info.get('yield')),
                    'TrailingPE': safe_float(info.get('trailingPE')),
                    'PriceToBook': safe_float(info.get('priceToBook')),
                }
            )
        except Exception as exc:
            print(f'Error retrieving yfinance metadata for {symbol}: {exc}')
    save_json_cache(YF_CACHE_FILE, cache)
    columns = [
        'Name',
        'Category',
        'FundFamily',
        'LegalType',
        'Country',
        'Exchange',
        'Price',
        'AUM',
        'ExpenseRatio',
        'AverageVolume',
        'Beta3Y',
        '3Y Average Return',
        '5Y Average Return',
        'YTD Return',
        'Yield',
        'TrailingPE',
        'PriceToBook',
        'Volatility 1Y',
        '50D Average',
        '200D Average',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
    ]
    for col in columns:
        df[col] = df['Symbol'].map(lambda symbol: data_map.get(symbol, {}).get(col))
    discovery = df.set_index('Symbol').to_dict('index')
    if '52 WkChange %' in df.columns:
        df['52 WkChange %'] = df.apply(
            lambda row: (
                row['52 WkChange %']
                if pd.notna(row['52 WkChange %'])
                else discovery.get(row['Symbol'], {}).get(
                    '52 WkChange %'
                )
            ),
            axis=1,
        )
    return (df, histories)


def get_benchmark_data(histories, benchmark_symbols):
    missing = [symbol for symbol in benchmark_symbols if symbol not in histories]
    if missing:
        print(f'Downloading missing benchmark history: {missing}')
        missing_histories = download_price_history(missing, period='13mo')
        histories.update(missing_histories)
    benchmark_metrics = {}
    for symbol in benchmark_symbols:
        benchmark_metrics[symbol] = compute_momentum_metrics(histories.get(symbol))
    return benchmark_metrics


def get_discovery_thresholds(config, benchmark_metrics, benchmark_symbols):
    one_year_returns = [
        benchmark_metrics.get(symbol, {}).get('1Y Return')
        for symbol in benchmark_symbols
    ]
    three_month_returns = [
        benchmark_metrics.get(symbol, {}).get('3M Return')
        for symbol in benchmark_symbols
    ]
    one_year_returns = [value for value in one_year_returns if value is not None]
    three_month_returns = [
        value for value in three_month_returns if value is not None
    ]
    configured_1y = config['min_52_week_change']
    configured_3m = config['min_3_month_return']
    one_year_tolerance = config.get('benchmark_1y_discovery_tolerance', 10.0)
    three_month_tolerance = config.get('benchmark_3m_discovery_tolerance', 5.0)
    discovery_1y = configured_1y
    discovery_3m = configured_3m
    if one_year_returns:
        discovery_1y = min(
            configured_1y,
            min(one_year_returns) - one_year_tolerance,
        )
    if three_month_returns:
        discovery_3m = min(
            configured_3m,
            min(three_month_returns) - three_month_tolerance,
        )
    return discovery_1y, discovery_3m


def get_benchmark_superiority_mask(scored_etfs, benchmark_df, benchmark_symbols):
    candidates = ~scored_etfs['Symbol'].isin(benchmark_symbols)
    complete_history = (
        scored_etfs['3M Return'].notna()
        & scored_etfs['1Y Return'].notna()
    )
    if benchmark_df.empty or benchmark_df['QVMScore'].isna().all():
        raise ValueError('Benchmark QVM scores are required for candidate filtering.')
    qvm_superior = scored_etfs['QVMScore'] > benchmark_df['QVMScore'].max()
    benchmark_3m_floor = benchmark_df['3M Return'].mean()
    benchmark_1y_floor = benchmark_df['1Y Return'].mean()
    competitive_horizons = (
        scored_etfs['3M Return'].ge(benchmark_3m_floor)
        & scored_etfs['1Y Return'].ge(benchmark_1y_floor)
    )

    benchmark_mean_cols = [
        f'{period} Excess vs Benchmarks'
        for period in ['1M', '3M', '6M', '9M', '1Y']
        if f'{period} Excess vs Benchmarks' in scored_etfs.columns
    ]
    if len(benchmark_mean_cols) < 3:
        raise ValueError('At least three benchmark-relative periods are required.')
    majority_periods = (
        scored_etfs[benchmark_mean_cols].gt(0).sum(axis=1)
        >= int(np.ceil(len(benchmark_mean_cols) / 2))
    )

    relative_to_each = []
    for benchmark in benchmark_symbols:
        cols = [
            f'{period} Excess vs {benchmark}'
            for period in ['1M', '3M', '6M', '9M', '1Y']
            if f'{period} Excess vs {benchmark}' in scored_etfs.columns
        ]
        if cols:
            relative_to_each.append(scored_etfs[cols].mean(axis=1) > 0)
    if len(relative_to_each) != len(benchmark_symbols):
        raise ValueError('Relative-return data is required for every benchmark.')
    positive_average_vs_each = pd.concat(relative_to_each, axis=1).all(axis=1)
    return (
        candidates
        & complete_history
        & qvm_superior
        & competitive_horizons
        & majority_periods
        & positive_average_vs_each
    )


def get_benchmark_competitive_mask(
    scored_etfs,
    benchmark_df,
    benchmark_symbols,
    qvm_tolerance=10.0,
    three_month_tolerance=5.0,
    one_year_tolerance=10.0,
):
    """Broader research admission; final benchmark PASS still comes from research."""
    candidates = ~scored_etfs['Symbol'].isin(benchmark_symbols)
    complete_history = (
        scored_etfs['3M Return'].notna()
        & scored_etfs['1Y Return'].notna()
    )
    benchmark_qvm = benchmark_df['QVMScore'].max()
    benchmark_3m = benchmark_df['3M Return'].mean()
    benchmark_1y = benchmark_df['1Y Return'].mean()
    qvm_competitive = scored_etfs['QVMScore'].ge(
        benchmark_qvm - qvm_tolerance
    )
    horizon_competitive = (
        scored_etfs['3M Return'].ge(benchmark_3m - three_month_tolerance)
        & scored_etfs['1Y Return'].ge(benchmark_1y - one_year_tolerance)
    )
    excess_columns = [
        f'{period} Excess vs Benchmarks'
        for period in ['1M', '3M', '6M', '9M', '1Y']
        if f'{period} Excess vs Benchmarks' in scored_etfs.columns
    ]
    if excess_columns:
        positive_or_near = (
            scored_etfs[excess_columns].gt(0).sum(axis=1).ge(2)
            | scored_etfs[excess_columns].mean(axis=1).ge(-three_month_tolerance)
        )
    else:
        positive_or_near = pd.Series(True, index=scored_etfs.index)
    return (
        candidates
        & complete_history
        & qvm_competitive
        & horizon_competitive
        & positive_or_near
    )


def build_ranked_etf_research_pool(
    scored_etfs,
    benchmark_symbols,
    superiority_mask,
    competitive_mask,
    min_quality,
    top_n,
    research_pool_limit,
):
    """Retain the actual top-QVM stream before research eligibility decisions."""
    ranked = scored_etfs[
        ~scored_etfs['Symbol'].isin(benchmark_symbols)
        & (scored_etfs['QualityScore'] >= min_quality)
    ].sort_values(
        ['QVMScore', 'MomentumScore'],
        ascending=[False, False],
    ).head(top_n)
    candidates = ranked.head(research_pool_limit).copy()
    strict = superiority_mask.reindex(candidates.index, fill_value=False)
    competitive = competitive_mask.reindex(candidates.index, fill_value=False)
    candidates['ResearchAdmission'] = np.where(strict, 'STRICT', 'BACKFILL')
    counts = {
        'strict': int(strict.sum()),
        'competitive_backfill': int((competitive & ~strict).sum()),
        'top_qvm_reserve': int((~competitive & ~strict).sum()),
    }
    return candidates, counts


def add_benchmark_relative_metrics(df, benchmark_metrics, benchmark_symbols):
    df = df.copy()
    periods = ['1M', '3M', '6M', '9M', '1Y']
    for period in periods:
        col = f'{period} Return'
        if col not in df.columns:
            continue
        benchmark_returns = []
        for benchmark in benchmark_symbols:
            value = benchmark_metrics.get(benchmark, {}).get(col)
            if value is not None:
                benchmark_returns.append(value)
        if not benchmark_returns:
            continue
        benchmark_mean = float(np.mean(benchmark_returns))
        df[f'{period} Excess vs Benchmarks'] = df[col] - benchmark_mean
        for benchmark in benchmark_symbols:
            benchmark_return = benchmark_metrics.get(benchmark, {}).get(col)
            if benchmark_return is not None:
                df[f'{period} Excess vs {benchmark}'] = df[col] - benchmark_return
    excess_cols = [
        f'{period} Excess vs Benchmarks'
        for period in periods
        if f'{period} Excess vs Benchmarks' in df.columns
    ]
    if excess_cols:
        df['BenchmarkRelativeScoreRaw'] = df[excess_cols].mean(axis=1)
        df['BenchmarkRelativeScore'] = (
            df['BenchmarkRelativeScoreRaw'].rank(pct=True) * 100
        ).fillna(50)
    return df


def winsorized_rank(series, ascending=True):
    s = pd.to_numeric(series, errors='coerce').copy()
    valid = s.dropna()
    if len(valid) >= 20:
        lower = valid.quantile(0.02)
        upper = valid.quantile(0.98)
        s = s.clip(lower, upper)
    return s.rank(pct=True, ascending=ascending)


def score_etf_qvm(df, top_n=100, weights=None, min_quality=35):
    df = df.copy()
    if weights is None:
        weights = {'Quality': 0.25, 'Value': 0.15, 'Momentum': 0.6}
    numeric_columns = [
        'AUM',
        'ExpenseRatio',
        'AverageVolume',
        'Beta3Y',
        '3Y Average Return',
        '5Y Average Return',
        'YTD Return',
        'Yield',
        'TrailingPE',
        'PriceToBook',
        'Volatility 1Y',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
        'BenchmarkRelativeScore',
    ]
    for col in numeric_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if df.empty:
        return pd.DataFrame()
    quality_components = []
    if 'AUM' in df.columns:
        aum = np.log1p(df['AUM'].clip(lower=0))
        quality_components.append(winsorized_rank(aum, ascending=True))
    if 'AverageVolume' in df.columns:
        volume = np.log1p(df['AverageVolume'].clip(lower=0))
        quality_components.append(winsorized_rank(volume, ascending=True))
    if 'ExpenseRatio' in df.columns:
        expense = df['ExpenseRatio'].clip(lower=0)
        quality_components.append(winsorized_rank(expense, ascending=False))
    if '3Y Average Return' in df.columns:
        quality_components.append(winsorized_rank(df['3Y Average Return'], ascending=True))
    if '5Y Average Return' in df.columns:
        quality_components.append(winsorized_rank(df['5Y Average Return'], ascending=True))
    if quality_components:
        q = pd.concat(quality_components, axis=1)
        valid_count = q.notna().sum(axis=1)
        quality = q.mean(axis=1, skipna=True)
        confidence = np.where(
            valid_count >= 3,
            1.0,
            np.where(
                valid_count == 2,
                0.85,
                np.where(valid_count == 1, 0.65, 0.5),
            ),
        )
        quality = 0.5 + (quality - 0.5) * confidence
        quality = quality.where(valid_count > 0, 0.5)
        df['QualityScore'] = (quality * 100).clip(0, 100)
    else:
        df['QualityScore'] = 50.0
    if min_quality > 0:
        df = df[df['QualityScore'] >= min_quality].copy()
    if df.empty:
        return pd.DataFrame()
    value_components = []
    if 'ExpenseRatio' in df.columns:
        value_components.append(winsorized_rank(df['ExpenseRatio'].clip(lower=0), ascending=False))
    if 'TrailingPE' in df.columns:
        pe = df['TrailingPE'].where(df['TrailingPE'] > 0)
        value_components.append(winsorized_rank(pe, ascending=False))
    if 'PriceToBook' in df.columns:
        pb = df['PriceToBook'].where(df['PriceToBook'] > 0)
        value_components.append(winsorized_rank(pb, ascending=False))
    if value_components:
        v = pd.concat(value_components, axis=1)
        valid_count = v.notna().sum(axis=1)
        value = v.mean(axis=1, skipna=True)
        confidence = np.where(valid_count >= 2, 1.0, np.where(valid_count == 1, 0.65, 0.5))
        value = 0.5 + (value - 0.5) * confidence
        value = value.where(valid_count > 0, 0.5)
        df['ValueScore'] = (value * 100).clip(0, 100)
    else:
        df['ValueScore'] = 50.0
    momentum_columns = ['1M Return', '3M Return', '6M Return', '9M Return', '1Y Return']
    m_cols = [col for col in momentum_columns if col in df.columns]
    if m_cols:
        m = df[m_cols].copy().clip(lower=-100, upper=500)
        mean_return = m.mean(axis=1, skipna=True)
        raw_momentum = mean_return.rank(pct=True) * 100
        valid_count = m.notna().sum(axis=1)
        positive_windows = (m > 0).sum(axis=1)
        trend_alignment = positive_windows / valid_count.replace(0, np.nan) * 100
        trend_alignment = trend_alignment.fillna(50)
        consistency_columns = [
            col
            for col in ['3M Return', '6M Return', '1Y Return']
            if col in df.columns
        ]
        if len(consistency_columns) >= 2:
            consistency = df[consistency_columns].rank(pct=True, axis=0)
            consistency_score = consistency.mean(axis=1, skipna=True) * 100
            dispersion = consistency.std(axis=1, skipna=True)
            consistency_score = (consistency_score - dispersion * 25).clip(0, 100).fillna(50)
        else:
            consistency_score = raw_momentum
        if 'Volatility 1Y' in df.columns:
            volatility = df['Volatility 1Y'].clip(lower=1)
            risk_adjusted_return = mean_return / volatility
            risk_adjusted_score = (risk_adjusted_return.rank(pct=True) * 100).fillna(50)
        else:
            risk_adjusted_score = raw_momentum
        if '3M Return' in df.columns:
            recent_score = (df['3M Return'].rank(pct=True) * 100).fillna(50)
        else:
            recent_score = raw_momentum
        if 'BenchmarkRelativeScore' in df.columns:
            benchmark_score = df['BenchmarkRelativeScore'].fillna(50)
        else:
            benchmark_score = pd.Series(50, index=df.index)
        momentum = (
            0.25 * raw_momentum
            + 0.25 * trend_alignment
            + 0.20 * consistency_score
            + 0.10 * risk_adjusted_score
            + 0.10 * recent_score
            + 0.10 * benchmark_score
        )
        df['MomentumScore'] = momentum.fillna(50).clip(0, 100)
    else:
        df['MomentumScore'] = 50.0
    if '3M Return' in df.columns:
        recent_return = df['3M Return'].fillna(0)
        recent_penalty = np.where(recent_return < 0, np.minimum(20, -recent_return * 0.5), 0)
        df['MomentumScore'] = (df['MomentumScore'] - recent_penalty).clip(0, 100)
    df['QVMScore'] = (
        df['QualityScore'] * weights['Quality']
        + df['ValueScore'] * weights['Value']
        + df['MomentumScore'] * weights['Momentum']
    )
    if 'Volatility 1Y' in df.columns:
        extreme_volatility = df['Volatility 1Y'] > 70
        df.loc[extreme_volatility, 'QVMScore'] *= 0.92
    return (
        df.sort_values(
            ['QVMScore', 'MomentumScore'],
            ascending=[False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


SELECTABLE_RISKS = {'MINIMAL', 'LOW', 'MODERATE'}
MAX_OBSERVED_EVIDENCE_AGE_DAYS = 370


def json_safe_value(value):
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataframe_records(df):
    return [
        {key: json_safe_value(value) for key, value in row.items()}
        for row in df.to_dict('records')
    ]


def validate_sources(sources, label, minimum=2, maximum=5):
    original_type = type(sources).__name__
    if isinstance(sources, dict):
        sources = [sources]
    elif isinstance(sources, str):
        sources = [sources]
    elif not isinstance(sources, list):
        print(f'Invalid sources [{label}]: type={original_type}')
        raise ValueError(f'{label} sources must be a list or source object.')
    cleaned, seen = [], set()
    for source in sources:
        if isinstance(source, str):
            markdown_link = re.fullmatch(
                r'\s*\[([^\]]+)\]\((https?://[^)]+)\)\s*', source
            )
            if markdown_link:
                title, url = markdown_link.group(1), markdown_link.group(2)
            else:
                title = label
                url = source.strip()
        elif isinstance(source, dict):
            web_source = source.get('web') if isinstance(source.get('web'), dict) else {}
            title = str(
                source.get('title')
                or source.get('name')
                or web_source.get('title')
                or label
            ).strip()
            url = str(
                source.get('url')
                or source.get('uri')
                or source.get('link')
                or source.get('href')
                or source.get('source_url')
                or web_source.get('uri')
                or web_source.get('url')
                or ''
            ).strip()
        else:
            continue
        if not re.match(r'^https?://', url, flags=re.IGNORECASE):
            continue
        normalized = url.rstrip('/')
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append({'title': title, 'url': url})
    if len(cleaned) < minimum:
        preview = str(sources[:2])[:240]
        print(
            f'Invalid sources [{label}]: type={original_type}, '
            f'normalized={len(cleaned)}, preview={preview}'
        )
        raise ValueError(f'{label} has {len(cleaned)} valid sources; {minimum} required.')
    return cleaned[:maximum]


def validate_market_context(data):
    if not isinstance(data, dict):
        raise ValueError('Market context must be a JSON object.')
    data = dict(data)
    required = ['as_of_date', 'active_risk_events', 'sources']
    for field in required:
        if field not in data:
            raise ValueError(f'Market context missing {field}.')
    summary = str(data.get('market_summary') or data.get('market_intro') or '').strip()
    if len(summary) < 40:
        raise ValueError('Market summary is too short.')
    if not re.search(r'\b\d+(?:\.\d+)?\s*%', summary):
        raise ValueError('Market summary must include an explicit percentage.')
    data['market_summary'] = summary
    data['market_intro'] = summary
    if not isinstance(data['active_risk_events'], list):
        raise ValueError('active_risk_events must be a list.')
    for field in ('major_drivers', 'macro_conditions', 'strong_sectors',
                  'weak_sectors', 'strong_exposures', 'weak_exposures'):
        if field in data and not isinstance(data[field], list):
            raise ValueError(f'{field} must be a list.')
    data.setdefault('strong_sectors', list(data.get('strong_exposures') or []))
    data.setdefault('weak_sectors', list(data.get('weak_exposures') or []))
    data.setdefault('strong_exposures', list(data.get('strong_sectors') or []))
    data.setdefault('weak_exposures', list(data.get('weak_sectors') or []))
    if not isinstance(data.get('sector_context'), dict):
        data['sector_context'] = {}
    if not isinstance(data.get('factor_and_theme_context'), dict):
        data['factor_and_theme_context'] = {}
    data['sources'] = validate_sources(data['sources'], 'Market context', minimum=2)
    return data


def query_matches_etf(query, candidate):
    query = str(query).upper()
    symbol = str(candidate['Symbol']).upper()
    if re.search(rf'(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])', query):
        return True
    words = [
        word.upper() for word in re.findall(r'[A-Za-z0-9]+', str(candidate.get('Name') or ''))
        if len(word) >= 4 and word.lower() not in {'fund', 'etf', 'trust', 'index', 'shares'}
    ]
    return bool(words and sum(word in query for word in words[:4]) >= min(2, len(words)))


def normalize_enum_token(value):
    """Normalize harmless Gemini enum formatting without changing semantics."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    text = re.sub(r'\\s+|/|-', '_', text)
    text = re.sub(r'[^A-Z0-9_]+', '', text)
    text = re.sub(r'_+', '_', text).strip('_')
    aliases = {
        'US_EQUITIES': 'US_EQUITY',
        'U_S_EQUITY': 'US_EQUITY',
        'NOT_US_EQUITIES': 'NOT_US_EQUITY',
        'NOT_U_S_EQUITY': 'NOT_US_EQUITY',
        'N_A': 'NONE',
        'NA': 'NONE',
        'NULL': 'NONE',
        'NO_MECHANISM': 'NONE',
    }
    return aliases.get(text, text)


def normalize_etf_result(result):
    result = dict(result)
    result['symbol'] = str(result.get('symbol') or '').strip().upper()
    enum_fields = (
        'research_status', 'benchmark_assessment',
        'mechanism_status', 'normalization_probability', 'probability_basis',
        'risk_materiality', 'risk_time_horizon', 'risk_basis', 'reversal_risk',
        'driver_dependence', 'mandate_assessment', 'mandate_basis',
        'evidence_scope',
    )
    for field in enum_fields:
        if result.get(field) is not None:
            result[field] = normalize_enum_token(result[field])
    # continuation_outlook is narrative text, not an enum.
    if result.get('continuation_outlook') is not None:
        result['continuation_outlook'] = str(
            result['continuation_outlook']
        ).strip()
    observed = result.get('adverse_change_observed')
    if isinstance(observed, str):
        lowered = observed.strip().lower()
        observed = True if lowered in {'true', 'yes', '1'} else (
            False if lowered in {'false', 'no', '0'} else None
        )
    elif observed is not None:
        observed = bool(observed)
    result['adverse_change_observed'] = observed
    result['us_equity_weight_estimate'] = safe_float(
        result.get('us_equity_weight_estimate')
    )
    reported_eligible = result.get('eligible')
    if isinstance(reported_eligible, str):
        reported_eligible = reported_eligible.strip().lower() in {
            'true', 'yes', '1'
        }
    elif reported_eligible is not None:
        reported_eligible = bool(reported_eligible)
    result['model_eligible'] = reported_eligible
    result['model_eligibility_reason'] = result.get('eligibility_reason')
    return result


def hypothetical_moderate_is_supported(result):
    """Require a real current weakening signal before HYPOTHETICAL reaches MODERATE."""
    if result.get('mechanism_status') != 'HYPOTHETICAL':
        return False
    if result.get('normalization_probability') != 'POSSIBLE':
        return False
    if result.get('risk_materiality') != 'HIGH' or result.get('driver_dependence') != 'HIGH':
        return False
    if result.get('risk_basis') not in {
        'TEMPORARY_DRIVER_NORMALIZATION', 'NORMALIZED_EXPOSURE_DETERIORATION'
    }:
        return False
    if result.get('probability_basis') in {None, 'NONE'}:
        return False
    required_evidence = (
        'current_driver_evidence', 'probability_evidence',
        'material_effect', 'reversal_mechanism',
    )
    if not all(
        len(str(result.get(field) or '').strip()) >= 15
        for field in required_evidence
    ):
        return False

    # HYPOTHETICAL MODERATE is intentionally rare. The text must describe an
    # actual current deterioration signal, not just a conditional future risk.
    evidence_text = ' '.join(
        str(result.get(field) or '')
        for field in ('current_driver_evidence', 'probability_evidence', 'reversal_mechanism')
    ).casefold()
    purely_conditional = any(re.search(pattern, evidence_text) for pattern in (
        r'\bif\s+',
        r'\bcould\s+',
        r'\bmay\s+',
        r'\bmight\s+',
        r'\bpossible\b',
        r'\bpotential\b',
    ))
    weakening_terms = (
        'weaken', 'deteriorat', 'declin', 'falling', 'slowing', 'soften',
        'compression', 'outflow', 'negative revision', 'downgrade',
        'inventory build', 'demand slowdown', 'supply increase', 'spread narrowing',
        'margin pressure', 'earnings pressure', 'policy change', 'guidance cut',
    )
    has_weakening_signal = any(term in evidence_text for term in weakening_terms)
    return has_weakening_signal and not purely_conditional


def derive_reversal_risk(result):
    status = result['mechanism_status']
    probability = result['normalization_probability']
    materiality = result['risk_materiality']
    dependence = result['driver_dependence']
    if status == 'NONE':
        minimal = (
            probability == 'UNLIKELY' and materiality == 'LOW'
            and dependence == 'LOW' and result.get('probability_basis') == 'NONE'
            and result.get('risk_basis') == 'NONE'
        )
        return 'MINIMAL' if minimal else 'LOW'
    if status == 'HYPOTHETICAL':
        return 'MODERATE' if hypothetical_moderate_is_supported(result) else 'LOW'
    if probability == 'LIKELY' and materiality == 'HIGH' and dependence == 'HIGH':
        return 'SEVERE'
    if (
        materiality == 'HIGH' and probability in {'POSSIBLE', 'LIKELY'}
    ) or (
        materiality == 'MODERATE'
        and probability == 'LIKELY'
        and dependence == 'HIGH'
    ):
        return 'ELEVATED'
    if probability in {'POSSIBLE', 'LIKELY'} and materiality in {'MODERATE', 'HIGH'}:
        return 'MODERATE'
    return 'LOW'


def risk_derivation_summary(result):
    parts = [
        f'mechanism={result.get("mechanism_status") or "N/A"}',
        f'probability={result.get("normalization_probability") or "N/A"}',
        f'materiality={result.get("risk_materiality") or "N/A"}',
        f'dependence={result.get("driver_dependence") or "N/A"}',
        f'basis={result.get("risk_basis") or "N/A"}',
    ]
    event = str(result.get('primary_risk_event_id') or '').strip()
    if event:
        parts.append(f'event={event}')
    return ', '.join(parts)


def normalize_hypothetical_consistency(result):
    """Deterministically enforce schema semantics without changing investment evidence."""
    result = dict(result)
    status = result.get('mechanism_status')
    if status == 'HYPOTHETICAL':
        # HYPOTHETICAL means no present adverse transition is evidenced. Clear
        # OBSERVED-only fields and cap probability at POSSIBLE. This is schema
        # normalization, not a discretionary risk downgrade.
        result['adverse_change_observed'] = False
        result['adverse_change_date'] = None
        result['adverse_change_indicator'] = None
        result['mechanism_evidence_source'] = None
        result['evidence_scope'] = 'NONE'
        if result.get('normalization_probability') == 'LIKELY':
            result['normalization_probability'] = 'POSSIBLE'
        if result.get('risk_basis') == 'NONE':
            result['primary_risk_event_id'] = None
            result['risk_exposure_group'] = None
    elif status == 'NONE':
        # These values are logically implied by NONE and are safe to normalize
        # when Gemini omits or harmlessly misformats them.
        result['adverse_change_observed'] = False
        result['adverse_change_date'] = None
        result['adverse_change_indicator'] = None
        result['mechanism_evidence_source'] = None
        result['evidence_scope'] = 'NONE'
        result['normalization_probability'] = 'UNLIKELY'
        result['probability_basis'] = 'NONE'
        result['risk_basis'] = 'NONE'
        result['primary_risk_event_id'] = None
        result['risk_exposure_group'] = None
        result['reversal_mechanism'] = None
        if result.get('risk_materiality') is None:
            result['risk_materiality'] = 'LOW'
        if result.get('driver_dependence') is None:
            result['driver_dependence'] = 'LOW'
    return result


def observed_mechanism_is_conditionally_worded(result):
    """Flag conditional scenarios that do not themselves prove present deterioration."""
    text = ' '.join(
        str(result.get(field) or '')
        for field in (
            'reversal_mechanism', 'adverse_change_indicator',
            'probability_evidence', 'material_effect'
        )
    ).casefold()
    conditional_patterns = (
        r'\bif\s+[^.]{0,80}\b',
        r'\bcould\s+(?:lead|cause|decline|fall|weaken|normalize)\b',
        r'\bmay\s+(?:lead|cause|decline|fall|weaken|normalize)\b',
        r'\bpotential\s+(?:de[- ]?escalation|normalization|decline|weakening)\b',
        r'\bde[- ]?escalation\s+could\b',
    )
    return any(re.search(pattern, text) for pattern in conditional_patterns)


def material_effect_is_generic(result, candidate):
    """Reject purely sector-level reversal effects that never connect to the ETF."""
    if result.get('mechanism_status') not in {'HYPOTHETICAL', 'OBSERVED'}:
        return False
    text = re.sub(
        r'\s+', ' ', str(result.get('material_effect') or '').strip().casefold()
    )
    if len(text) < 15:
        return True
    symbol = str(candidate.get('Symbol') or '').strip().casefold()
    name_words = {
        word.casefold()
        for word in re.findall(r'[A-Za-z0-9]+', str(candidate.get('Name') or ''))
        if len(word) >= 4 and word.casefold() not in {
            'fund', 'etf', 'trust', 'index', 'shares'
        }
    }
    construction_terms = {
        'holding', 'holdings', 'portfolio', 'weight', 'weighted', 'equal-weight',
        'equal weight', 'index', 'factor', 'screen', 'concentration',
        'exposure', 'constituent', 'constituents', 'allocation',
    }
    has_etf_reference = bool(symbol and symbol in text) or any(
        word in text for word in name_words
    )
    has_construction_reference = any(term in text for term in construction_terms)
    generic_patterns = (
        r'^(?:this|the)\s+(?:sector|industry|market|theme)\b',
        r'\b(?:could|may|would)\s+(?:pressure|hurt|weaken)\s+'
        r'(?:the\s+)?(?:sector|industry|technology|growth stocks?|energy|healthcare)\b',
    )
    looks_generic = any(re.search(pattern, text) for pattern in generic_patterns)
    return looks_generic and not (has_etf_reference or has_construction_reference)


def validate_etf_result(result, candidate, minimum_sources=2, maximum_sources=5):
    result = normalize_etf_result(result)
    result = normalize_hypothetical_consistency(result)
    symbol = str(candidate['Symbol']).upper()
    if result['symbol'] != symbol:
        raise ValueError(f'Expected {symbol}, received {result["symbol"] or "missing symbol"}.')
    if result.get('research_status') != 'COMPLETE':
        reason = str(result.get('research_incomplete_reason') or 'unspecified reason').strip()
        raise ValueError(f'{symbol} research incomplete: {reason}')
    required_text = [
        'fund_name', 'exposure_group', 'holdings_evidence',
        'current_driver_evidence', 'continuation_outlook', 'risk_basis',
        'risk_materiality', 'risk_time_horizon', 'explanation',
    ]
    for field in required_text:
        if not str(result.get(field) or '').strip():
            raise ValueError(f'{symbol} missing {field}.')
    result['sources'] = validate_sources(
        result.get('sources'), symbol, minimum=minimum_sources,
        maximum=maximum_sources,
    )
    if result.get('benchmark_assessment') not in {'PASS', 'FAIL'}:
        raise ValueError(f'{symbol} has invalid benchmark_assessment.')
    if result.get('mandate_assessment') not in {'US_EQUITY', 'NOT_US_EQUITY'}:
        raise ValueError(f'{symbol} has invalid mandate_assessment.')
    if len(str(result.get('mandate_evidence') or '').strip()) < 10:
        raise ValueError(f'{symbol} missing mandate_evidence.')
    if result.get('mandate_basis') not in {
        'INDEX_MANDATE', 'HOLDINGS_WEIGHT', 'ASSET_CLASS', 'OTHER'
    }:
        raise ValueError(f'{symbol} has invalid mandate_basis.')
    us_equity_weight = result.get('us_equity_weight_estimate')
    if us_equity_weight is None or not 0 <= us_equity_weight <= 100:
        raise ValueError(
            f'{symbol} has invalid us_equity_weight_estimate; 0-100 required.'
        )
    if result['mandate_assessment'] == 'US_EQUITY' and us_equity_weight < 80:
        raise ValueError(
            f'{symbol} US_EQUITY mandate requires '
            'us_equity_weight_estimate of at least 80.'
        )
    if result['mandate_assessment'] == 'NOT_US_EQUITY' and us_equity_weight >= 80:
        raise ValueError(
            f'{symbol} NOT_US_EQUITY conflicts with '
            'us_equity_weight_estimate of at least 80.'
        )
    if result.get('mechanism_status') not in {'NONE', 'HYPOTHETICAL', 'OBSERVED'}:
        raise ValueError(f'{symbol} has invalid mechanism_status.')
    if result.get('normalization_probability') not in {'UNLIKELY', 'POSSIBLE', 'LIKELY'}:
        raise ValueError(f'{symbol} has invalid normalization_probability.')
    if result.get('risk_materiality') not in {'LOW', 'MODERATE', 'HIGH'}:
        raise ValueError(f'{symbol} has invalid risk_materiality.')
    if result.get('driver_dependence') not in {'LOW', 'MODERATE', 'HIGH'}:
        raise ValueError(f'{symbol} has invalid driver_dependence.')
    if result.get('risk_basis') not in {
        'NONE', 'NORMALIZED_EXPOSURE_DETERIORATION',
        'TEMPORARY_DRIVER_NORMALIZATION', 'NONRECURRING_COMPARISON_ONLY',
    }:
        raise ValueError(f'{symbol} has invalid risk_basis.')
    if result.get('probability_basis') not in {
        'OFFICIAL_GUIDANCE', 'REALIZED_OPERATING_DATA',
        'CONTRACT_OR_POLICY_TIMELINE', 'SUPPLY_DEMAND_DATA',
        'HOLDINGS_OR_EARNINGS_DATA', 'VALUATION_OR_FLOW_DATA',
        'MULTI_SOURCE_DIRECTIONAL_EVIDENCE', 'NONE',
    }:
        raise ValueError(f'{symbol} has invalid probability_basis.')
    if result.get('evidence_scope') not in {
        'FUND_SPECIFIC', 'EXPOSURE_SPECIFIC', 'MARKET_ONLY', 'NONE'
    }:
        raise ValueError(f'{symbol} has invalid evidence_scope.')
    if not isinstance(result.get('adverse_change_observed'), bool):
        raise ValueError(f'{symbol} has invalid adverse_change_observed.')
    if result['mechanism_status'] == 'OBSERVED':
        if not result['adverse_change_observed']:
            raise ValueError(
                f'{symbol} observed mechanism lacks adverse_change_observed.'
            )
        if result['evidence_scope'] not in {
            'FUND_SPECIFIC', 'EXPOSURE_SPECIFIC'
        }:
            raise ValueError(
                f'{symbol} observed mechanism has invalid evidence_scope; '
                'market context alone is insufficient.'
            )
        indicator = str(result.get('adverse_change_indicator') or '').strip()
        if len(indicator) < 15:
            raise ValueError(
                f'{symbol} observed mechanism lacks adverse_change_indicator.'
            )
        date_text = str(result.get('adverse_change_date') or '').strip()
        try:
            evidence_date = datetime.fromisoformat(
                date_text.replace('Z', '+00:00')
            ).date()
        except (TypeError, ValueError):
            raise ValueError(
                f'{symbol} observed mechanism has invalid adverse_change_date.'
            )
        age_days = (datetime.now(UTC).date() - evidence_date).days
        if age_days > MAX_OBSERVED_EVIDENCE_AGE_DAYS or age_days < -7:
            raise ValueError(
                f'{symbol} adverse_change_date is not current.'
            )
        mechanism_source = str(
            result.get('mechanism_evidence_source') or ''
        ).strip().rstrip('/')
        source_urls = {
            str(item.get('url') or '').strip().rstrip('/')
            for item in result['sources']
        }
        if not mechanism_source or mechanism_source not in source_urls:
            raise ValueError(
                f'{symbol} mechanism_evidence_source is not in sources.'
            )
        if observed_mechanism_is_conditionally_worded(result):
            raise ValueError(
                f'{symbol} OBSERVED mechanism is conditional rather than a '
                'current adverse directional change.'
            )
    elif result['adverse_change_observed']:
        raise ValueError(
            f'{symbol} adverse_change_observed conflicts with '
            f'{result["mechanism_status"]} mechanism_status.'
        )
    if result['mechanism_status'] == 'NONE':
        if result['normalization_probability'] != 'UNLIKELY':
            raise ValueError(f'{symbol} mechanism NONE requires UNLIKELY normalization.')
        if result['probability_basis'] != 'NONE' or result['risk_basis'] != 'NONE':
            raise ValueError(
                f'{symbol} mechanism NONE requires probability_basis and risk_basis NONE.'
            )
    if result['mechanism_status'] == 'OBSERVED' and result['probability_basis'] == 'NONE':
        raise ValueError(f'{symbol} OBSERVED mechanism lacks a probability basis.')
    generic_transmission = material_effect_is_generic(result, candidate)
    reported_risk = result.get('reversal_risk')
    risk = derive_reversal_risk(result)
    if generic_transmission and result.get('mechanism_status') == 'OBSERVED' \
            and risk in {'MODERATE', 'ELEVATED', 'SEVERE'}:
        raise ValueError(
            f'{symbol} observed reversal material_effect is generic and lacks '
            'an ETF-specific transmission path.'
        )
    result['reported_reversal_risk'] = reported_risk
    result['reversal_risk'] = risk
    if reported_risk is not None and reported_risk != risk:
        print(
            f'Normalized ETF reversal risk [{symbol}]: '
            f'{reported_risk or "missing"} -> {risk}'
        )
    admission = str(candidate.get('ResearchAdmission') or 'STRICT').upper()
    preliminarily_ineligible = (
        result['mandate_assessment'] != 'US_EQUITY'
        or risk in {'ELEVATED', 'SEVERE'}
        or (
            admission == 'BACKFILL'
            and result['benchmark_assessment'] != 'PASS'
        )
    )
    evidence_warnings = []
    if generic_transmission and risk == 'LOW':
        evidence_warnings.append(
            f'{symbol} reversal material_effect may lack an ETF-specific '
            'transmission path.'
        )
    if risk in {'MODERATE', 'ELEVATED', 'SEVERE'}:
        hypothetical_moderate = (
            risk == 'MODERATE'
            and result['mechanism_status'] == 'HYPOTHETICAL'
            and hypothetical_moderate_is_supported(result)
        )
        if risk in {'ELEVATED', 'SEVERE'} and result['mechanism_status'] != 'OBSERVED':
            raise ValueError(f'{symbol} {risk} risk lacks an observed mechanism.')
        if risk == 'MODERATE' and result['mechanism_status'] not in {'OBSERVED', 'HYPOTHETICAL'}:
            raise ValueError(f'{symbol} MODERATE risk lacks a supported mechanism.')
        if risk == 'MODERATE' and result['mechanism_status'] == 'HYPOTHETICAL' and not hypothetical_moderate:
            raise ValueError(
                f'{symbol} hypothetical MODERATE risk lacks the required '
                'POSSIBLE/HIGH/HIGH normalization evidence.'
            )
        for field in ('reversal_mechanism', 'probability_evidence', 'material_effect'):
            if len(str(result.get(field) or '').strip()) < 15:
                message = f'{symbol} {risk} risk lacks {field}.'
                if preliminarily_ineligible:
                    evidence_warnings.append(message)
                else:
                    raise ValueError(message)
    result['evidence_warnings'] = evidence_warnings
    if evidence_warnings:
        print(
            f'Validated exclusion with evidence warning [{symbol}]: '
            + ' '.join(evidence_warnings)
        )
    result['research_admission'] = admission
    result['eligible'] = True
    if result['mandate_assessment'] != 'US_EQUITY':
        result['eligible'] = False
        result['eligibility_reason'] = (
            'Excluded because current holdings do not satisfy the U.S.-equity mandate. '
            + str(result.get('mandate_evidence') or '')
        ).strip()
    elif risk in {'ELEVATED', 'SEVERE'}:
        result['eligible'] = False
        result['eligibility_reason'] = (
            f'Excluded by Python because the evidence-derived reversal risk is '
            f'{risk} ({risk_derivation_summary(result)}).'
        )
    elif admission == 'BACKFILL' and result['benchmark_assessment'] != 'PASS':
        result['eligible'] = False
        result['eligibility_reason'] = (
            'Excluded because this benchmark-competitive backfill lacks a positive '
            'forward benchmark assessment.'
        )
    else:
        result['eligibility_reason'] = (
            'Python eligibility passed: U.S.-equity mandate, selectable reversal '
            f'risk and {admission.lower()} quantitative admission.'
        )
    return result


def validate_etf_batch(data, candidates, minimum_sources=2, maximum_sources=5):
    if isinstance(data, dict):
        results = data.get('results')
    else:
        results = data
    if not isinstance(results, list):
        raise ValueError('ETF batch response must contain a results list.')
    expected = {str(item['Symbol']).upper(): item for item in candidates}
    raw_by_symbol = {}
    for item in results:
        if isinstance(item, dict):
            symbol = str(item.get('symbol') or '').strip().upper()
            if symbol in expected and symbol not in raw_by_symbol:
                raw_by_symbol[symbol] = item
    valid, errors = {}, {}
    for symbol, candidate in expected.items():
        if symbol not in raw_by_symbol:
            errors[symbol] = 'Gemini omitted the ETF.'
            continue
        try:
            valid[symbol] = validate_etf_result(
                raw_by_symbol[symbol], candidate, minimum_sources, maximum_sources
            )
        except Exception as exc:
            errors[symbol] = str(exc)
    return valid, errors, raw_by_symbol


def merge_structural_repair_patches(data, repair_payload):
    """Expand compact field-only repairs into the prior full ETF drafts."""
    if not repair_payload:
        return data, {}
    response = dict(data) if isinstance(data, dict) else {'results': data}
    results = response.get('results')
    if not isinstance(results, list):
        results = []
    full_symbols = {
        str(item.get('symbol') or '').strip().upper()
        for item in results if isinstance(item, dict)
    }
    repairs = response.get('repairs')
    if not isinstance(repairs, list):
        repairs = []
    patches = {
        str(item.get('symbol') or '').strip().upper(): item
        for item in repairs
        if isinstance(item, dict) and item.get('symbol')
    }
    applied = {}
    for request in repair_payload:
        symbol = str(request.get('symbol') or '').strip().upper()
        if not symbol or symbol in full_symbols:
            continue
        prior = request.get('prior_draft')
        patch = patches.get(symbol)
        if not isinstance(prior, dict) or not isinstance(patch, dict):
            continue
        validation_error = str(request.get('validation_error') or '').lower()
        repairable_fields = {
            'research_status', 'research_incomplete_reason', 'fund_name',
            'sources', 'exposure_group', 'holdings_evidence',
            'current_driver_evidence', 'mandate_assessment',
            'mandate_evidence', 'us_equity_weight_estimate',
            'mandate_basis', 'benchmark_assessment',
            'continuation_outlook', 'reversal_mechanism',
            'mechanism_status', 'normalization_probability',
            'probability_basis', 'probability_evidence', 'material_effect',
            'risk_time_horizon', 'risk_materiality', 'driver_dependence',
            'risk_basis', 'primary_risk_event_id', 'risk_exposure_group',
            'adverse_change_observed', 'adverse_change_date',
            'adverse_change_indicator', 'mechanism_evidence_source',
            'evidence_scope',
            'explanation',
        }
        allowed_fields = {
            field for field in repairable_fields
            if field.lower() in validation_error
        }
        if 'sources' in validation_error or 'ungrounded' in validation_error:
            allowed_fields.add('sources')
        if any(fragment in validation_error for fragment in (
            'observed mechanism', 'adverse_change', 'evidence_scope',
            'mechanism_evidence_source', 'market context alone',
            'etf-specific transmission path', 'material_effect is generic',
        )):
            allowed_fields.update({
                'mechanism_status', 'reversal_mechanism',
                'normalization_probability', 'probability_evidence',
                'risk_materiality', 'driver_dependence', 'material_effect',
                'adverse_change_observed', 'adverse_change_date',
                'adverse_change_indicator', 'mechanism_evidence_source',
                'evidence_scope', 'probability_basis', 'risk_basis',
            })
        if any(fragment in validation_error for fragment in (
            'mandate', 'us_equity_weight_estimate', 'mandate_basis',
        )):
            allowed_fields.update({
                'mandate_assessment', 'mandate_evidence',
                'us_equity_weight_estimate', 'mandate_basis',
            })
        if 'basic_metadata_repair' in validation_error:
            # This path is only used for a high-ranked ETF whose first search
            # returned grounding but Gemini left obvious fund-identity fields
            # incomplete. Reconstruct the COMPLETE record from the supplied
            # candidate, prior draft and prior grounding without another search.
            allowed_fields.update(repairable_fields)
        if 'consistency conflict' in validation_error:
            allowed_fields.update({
                'exposure_group', 'benchmark_assessment',
                'continuation_outlook', 'reversal_mechanism',
                'mechanism_status', 'normalization_probability',
                'probability_basis', 'probability_evidence', 'material_effect',
                'risk_time_horizon', 'risk_materiality', 'driver_dependence',
                'risk_basis', 'primary_risk_event_id', 'risk_exposure_group',
                'adverse_change_observed', 'adverse_change_date',
                'adverse_change_indicator', 'mechanism_evidence_source',
                'evidence_scope', 'mandate_assessment', 'mandate_evidence',
                'us_equity_weight_estimate', 'mandate_basis', 'explanation',
            })
        merged = dict(prior)
        changed_fields = []
        for field, value in patch.items():
            if field == 'symbol' or field not in allowed_fields:
                continue
            merged[field] = value
            changed_fields.append(field)
        merged['symbol'] = symbol
        results.append(merged)
        applied[symbol] = sorted(changed_fields)
    response['results'] = results
    return response, applied


def is_basic_metadata_repair_candidate(message, draft=None):
    """Identify incomplete high-rank drafts repairable from already-grounded evidence."""
    text = str(message or '').lower()
    if 'research incomplete' not in text or not isinstance(draft, dict):
        return False
    # Besides obvious identity/mandate omissions, treat missing recent-driver or
    # outlook prose as an evidence-completion problem rather than proof that a
    # second Google search is needed. The repair call receives the candidate's
    # structured fund data, MARKET_CONTEXT, prior draft and grounding URLs.
    repairable_markers = (
        'fund name', 'full fund name', 'holdings identity', 'index exposure',
        'mandate', 'issuer', 'primary exposure identity', 'holdings',
        'recent performance driver', 'performance drivers', 'recent driver',
        'current driver', 'continuation outlook', '6-12 month thesis',
        '6-12 month outlook', 'recent outlook', 'and outlook',
    )
    return any(marker in text for marker in repairable_markers)


def validation_error_requires_fresh_research(
    message,
    draft=None,
    grounding_urls=None,
):
    """Separate evidence failures from schema/classification repair failures."""
    text = str(message or '').lower()
    return any(fragment in text for fragment in (
        'research incomplete',
        'omitted the etf',
        'did not expose an etf-specific search',
        'ungrounded',
    ))


def is_source_validation_error(message):
    text = str(message or '').lower()
    return any(fragment in text for fragment in (
        'valid sources',
        'sources must be a list or source object',
    ))


def source_shape(value):
    if value is None:
        return 'missing/null'
    if isinstance(value, list):
        return f'list[{len(value)}]'
    if isinstance(value, dict):
        return f'object[{len(value)}]'
    if isinstance(value, str):
        return f'string[{len(value)}]'
    return type(value).__name__


def provisional_exposure_group(candidate):
    text = ' '.join(
        str(candidate.get(field) or '')
        for field in ('PortfolioGroup', 'CanonicalSector', 'Name', 'Category')
    ).upper()
    rules = (
        ('OIL_REFINERS', ('REFINER', 'REFINING')),
        ('OIL_GAS_PRODUCERS', ('EXPLORATION', 'PRODUCTION ETF', 'OIL & GAS E&P')),
        ('BROAD_ENERGY', ('ENERGY', 'OIL', 'GAS')),
        ('BIOTECH', ('BIOTECH', 'BIOTECHNOLOGY')),
        ('TECHNOLOGY', ('TECHNOLOGY', 'SEMICONDUCTOR', 'SOFTWARE')),
        ('FINANCIALS', ('FINANCIAL', 'BANK', 'INSURANCE')),
        ('INDUSTRIALS', ('INDUSTRIAL', 'AEROSPACE', 'DEFENSE')),
        ('HEALTHCARE', ('HEALTH', 'PHARMA', 'MEDICAL')),
        ('VALUE', ('VALUE',)),
        ('MOMENTUM', ('MOMENTUM',)),
        ('DIVIDEND', ('DIVIDEND',)),
        ('SMALL_CAP', ('SMALL CAP', 'SMALL-CAP')),
    )
    for group, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return group
    category = re.sub(r'[^A-Z0-9]+', '_', str(candidate.get('Category') or '').upper()).strip('_')
    return category or 'OTHER'


def normalized_risk_event_key(research):
    event = str(research.get('primary_risk_event_id') or '').strip().upper()
    exposure = str(research.get('risk_exposure_group') or '').strip().upper()
    return (event, exposure) if event and exposure else None


def candidate_sector_group(candidate):
    return str(
        candidate.get('PortfolioGroup') or candidate.get('CanonicalSector')
        or candidate.get('Category') or 'Other'
    ).strip() or 'Other'


def audit_etf_consistency(candidates, research_by_symbol):
    warnings = []
    candidate_by_symbol = {
        str(candidate['Symbol']).upper(): candidate for candidate in candidates
    }
    by_exposure, by_group, explanations = {}, {}, {}
    risk_order = {'MINIMAL': 0, 'LOW': 1, 'MODERATE': 2, 'ELEVATED': 3, 'SEVERE': 4}
    for symbol, research in research_by_symbol.items():
        candidate = candidate_by_symbol.get(symbol)
        if not candidate:
            continue
        group = candidate_sector_group(candidate)
        supplied = str(candidate.get('Category') or '').strip()
        if supplied and supplied.casefold() not in {'other', 'miscellaneous'} \
                and supplied.casefold() != group.casefold():
            warnings.append(f'{symbol}: normalized {supplied} to {group}.')
        exposure = str(research.get('exposure_group') or '').strip().upper()
        if exposure:
            by_exposure.setdefault(exposure, []).append((symbol, research))
        by_group.setdefault(group, []).append((symbol, research))
        explanation = re.sub(r'\s+', ' ', str(research.get('explanation') or '').strip().casefold())
        if explanation:
            explanations.setdefault(explanation, []).append(symbol)
    for exposure, rows in by_exposure.items():
        mandates = {row.get('mandate_assessment') for _, row in rows}
        if len(mandates) > 1:
            warnings.append(f'{exposure}: conflicting mandates for ' + ', '.join(s for s, _ in rows))
        levels = [risk_order.get(str(row.get('reversal_risk') or '').upper()) for _, row in rows]
        levels = [level for level in levels if level is not None]
        if levels and max(levels) - min(levels) >= 2:
            warnings.append(f'{exposure}: risk spread of at least two levels for ' + ', '.join(s for s, _ in rows))
    for group, rows in by_group.items():
        if len(rows) >= 3 and all(str(row.get('reversal_risk') or '').upper() in {'ELEVATED', 'SEVERE'} for _, row in rows):
            warnings.append(f'{group}: all {len(rows)} researched ETFs are ELEVATED/SEVERE; verify ETF-specific evidence.')
    for symbols in explanations.values():
        if len(symbols) > 1:
            warnings.append('Identical explanations for ' + ', '.join(symbols))
    if warnings:
        print('ETF cross-candidate consistency warnings:')
        for warning in warnings:
            print(f'  - {warning}')
    else:
        print('ETF cross-candidate consistency: no material conflicts detected.')
    return warnings


def find_actionable_consistency_conflicts(
    candidates,
    research_by_symbol,
    selected,
):
    """Return peer conflicts that could change the current ranked portfolio."""
    rank_by_symbol = {
        str(candidate['Symbol']).upper(): rank
        for rank, candidate in enumerate(candidates, start=1)
    }
    selected_symbols = {
        str(item['candidate']['Symbol']).upper() for item in selected
    }
    selected_ranks = [
        rank_by_symbol[symbol]
        for symbol in selected_symbols if symbol in rank_by_symbol
    ]
    cutoff_rank = max(selected_ranks, default=0)
    risk_order = {
        'MINIMAL': 0, 'LOW': 1, 'MODERATE': 2, 'ELEVATED': 3, 'SEVERE': 4
    }
    by_exposure = {}
    for symbol, research in research_by_symbol.items():
        exposure = str(research.get('exposure_group') or '').strip().upper()
        if exposure:
            by_exposure.setdefault(exposure, []).append((symbol, research))

    conflicts = {}
    peer_fields = (
        'symbol', 'exposure_group', 'benchmark_assessment',
        'mandate_assessment', 'us_equity_weight_estimate', 'reversal_risk',
        'mechanism_status', 'normalization_probability', 'probability_basis',
        'risk_materiality', 'driver_dependence', 'risk_basis',
        'adverse_change_observed', 'adverse_change_date',
        'adverse_change_indicator', 'evidence_scope',
    )
    for exposure, rows in by_exposure.items():
        if len(rows) < 2:
            continue
        levels = [
            risk_order.get(str(row.get('reversal_risk') or '').upper())
            for _, row in rows
        ]
        levels = [level for level in levels if level is not None]
        issues = []
        if levels and max(levels) - min(levels) >= 2:
            issues.append('reversal risk differs by at least two levels')
        benchmarks = {
            str(row.get('benchmark_assessment') or '').upper()
            for _, row in rows
        }
        if len(benchmarks - {''}) > 1:
            issues.append('benchmark assessment conflicts')
        mandates = {
            str(row.get('mandate_assessment') or '').upper()
            for _, row in rows
        }
        if len(mandates - {''}) > 1:
            issues.append('mandate assessment conflicts')
        if not issues:
            continue
        symbols = [symbol for symbol, _ in rows]
        affects_portfolio = bool(selected_symbols.intersection(symbols)) or any(
            0 < rank_by_symbol.get(symbol, float('inf')) <= cutoff_rank
            for symbol in symbols
        )
        if not affects_portfolio:
            continue
        peers = [
            {field: row.get(field) for field in peer_fields}
            for _, row in rows
        ]
        message = (
            f'Consistency conflict for {exposure}: '
            + '; '.join(issues)
            + '. Review peer classifications and preserve differences only '
              'with concrete fund-specific evidence.'
        )
        for symbol, _ in rows:
            conflicts[symbol] = {
                'message': message,
                'peer_classifications': peers,
            }
    return conflicts


def enqueue_consistency_repairs(
    conflicts,
    candidate_by_symbol,
    research_by_symbol,
    research_cache,
    cache_keys,
    pending_retries,
    reviewed_symbols,
    structural_repairs_by_symbol,
    max_structural_repairs,
):
    """Queue evidence-only peer reconciliation without new searches."""
    queued = []
    pending_symbols = {
        str(item['candidate']['Symbol']).upper() for item in pending_retries
    }
    for symbol, conflict in conflicts.items():
        if (
            symbol in reviewed_symbols
            or symbol in pending_symbols
            or structural_repairs_by_symbol.get(symbol, 0)
            >= max_structural_repairs
        ):
            continue
        candidate = candidate_by_symbol.get(symbol)
        prior = research_by_symbol.get(symbol)
        cache_key = cache_keys.get(symbol)
        if not candidate or not isinstance(prior, dict) or not cache_key:
            continue
        source_urls = [
            str(item.get('url') or '').strip()
            for item in prior.get('sources', [])
            if isinstance(item, dict) and item.get('url')
        ]
        research_cache['deferred_entries'][cache_key] = {
            'timestamp': datetime.now(UTC).isoformat(),
            'error': conflict['message'],
            'draft': prior,
            'peer_classifications': conflict.get('peer_classifications', []),
            'grounding_source_urls': source_urls,
        }
        research_cache['entries'].pop(cache_key, None)
        research_by_symbol.pop(symbol, None)
        pending_retries.append({
            'candidate': candidate,
            'needs_research': False,
        })
        reviewed_symbols.add(symbol)
        pending_symbols.add(symbol)
        queued.append(symbol)
    if queued:
        print(
            'Queued evidence-only ETF consistency repair: '
            + ', '.join(queued)
        )
    return queued


def lower_ranked_candidate_blocked_by_sector_capacity(
    candidate,
    selected,
    rank_by_symbol,
    max_per_sector,
):
    sector_key = candidate_sector_group(candidate).casefold()
    selected_in_sector = [
        item for item in selected
        if candidate_sector_group(item['candidate']).casefold() == sector_key
    ]
    if len(selected_in_sector) < max_per_sector:
        return False
    candidate_rank = rank_by_symbol.get(str(candidate['Symbol']).upper(), float('inf'))
    worst_selected_rank = max(
        rank_by_symbol.get(
            str(item['candidate']['Symbol']).upper(), float('inf')
        )
        for item in selected_in_sector
    )
    return candidate_rank > worst_selected_rank


def pending_candidate_can_improve_full_portfolio(
    candidate, selected, rank_by_symbol, max_per_sector
):
    """After 10/10, only keep unresolved candidates that can displace a selection."""
    if not selected:
        return True
    symbol = str(candidate['Symbol']).upper()
    candidate_rank = rank_by_symbol.get(symbol, float('inf'))
    selected_ranks = [
        rank_by_symbol.get(str(item['candidate']['Symbol']).upper(), float('inf'))
        for item in selected
    ]
    worst_selected_rank = max(selected_ranks, default=float('inf'))
    if candidate_rank >= worst_selected_rank:
        return False

    sector_key = candidate_sector_group(candidate).casefold()
    same_sector = [
        item for item in selected
        if candidate_sector_group(item['candidate']).casefold() == sector_key
    ]
    if len(same_sector) < max_per_sector:
        return True
    worst_same_sector_rank = max(
        rank_by_symbol.get(
            str(item['candidate']['Symbol']).upper(), float('inf')
        )
        for item in same_sector
    )
    return candidate_rank < worst_same_sector_rank


def preview_portfolio(candidates, research_by_symbol, target, max_per_sector, max_moderate_event):
    selected, sector_counts, event_counts = [], {}, {}
    for candidate in candidates:
        if len(selected) >= target:
            break
        symbol = str(candidate['Symbol']).upper()
        research = research_by_symbol.get(symbol)
        if not research or not research.get('eligible', True):
            continue
        risk = str(research.get('reversal_risk') or '').upper()
        if risk not in SELECTABLE_RISKS:
            continue
        sector = candidate_sector_group(candidate)
        sector_key = sector.casefold()
        if sector_counts.get(sector_key, 0) >= max_per_sector:
            continue
        event_key = normalized_risk_event_key(research)
        if risk == 'MODERATE' and event_key and event_counts.get(event_key, 0) >= max_moderate_event:
            continue
        selected.append({'candidate': candidate, 'research': research})
        sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
        if risk == 'MODERATE' and event_key:
            event_counts[event_key] = event_counts.get(event_key, 0) + 1
    return selected


def print_validated_research_inspection(candidates, research_by_symbol):
    rows = []
    for candidate in candidates:
        symbol = str(candidate['Symbol']).upper()
        research = research_by_symbol.get(symbol)
        if not research:
            continue
        rows.append((candidate, research))
    if not rows:
        return
    print('Validated ETF classification inspection:')
    for candidate, research in rows:
        symbol = str(candidate['Symbol']).upper()
        print(
            f'  {symbol}: admission={research.get("research_admission") or candidate.get("ResearchAdmission")}, '
            f'benchmark={research.get("benchmark_assessment")}, '
            f'mandate={research.get("mandate_assessment")}, '
            f'risk={research.get("reversal_risk")}, '
            f'model_eligible={research.get("model_eligible")}, '
            f'python_eligible={research.get("eligible")}, '
            f'portfolio_group={candidate_sector_group(candidate)}, '
            f'canonical_sector={candidate.get("CanonicalSector") or "N/A"}, '
            f'style={candidate.get("StyleCategory") or "N/A"}, '
            f'exposure={research.get("exposure_group")}; '
            f'{risk_derivation_summary(research)}; '
            f'{research.get("eligibility_reason")}'
        )
        if research.get('reversal_risk') in {'MODERATE', 'ELEVATED', 'SEVERE'}:
            print(
                f'    reversal evidence: {research.get("reversal_mechanism")}; '
                f'probability: {research.get("probability_evidence")}; '
                f'effect: {research.get("material_effect")}'
            )


def print_decision_ledger(ledger, heading='ETF portfolio-rule inspection:'):
    print(heading)
    for row in ledger:
        print(
            f'  rank={row["qvm_rank"]} {row["symbol"]}: {row["status"]}; '
            f'portfolio_group={row.get("sector_group") or "N/A"}; '
            f'{row.get("risk_derivation") or "risk=N/A"}; '
            f'{row.get("explanation") or "No explanation."}'
        )


def build_context_review(
    decision_ledger,
    research_by_symbol,
    selected,
    researched_this_run,
    charged_search_attempts,
    structural_repairs_by_symbol,
    batch_research_diagnostics,
):
    """Mirror the stock recommender's terminal context review for ETFs."""
    selected_symbols = [
        str(item['candidate']['Symbol']).upper() for item in selected
    ]
    validated_symbols = set(research_by_symbol)
    researched_symbols = set(researched_this_run)
    current_validated = validated_symbols.intersection(researched_symbols)
    cached_validated = validated_symbols - researched_symbols
    warning_count = sum(
        len(research.get('evidence_warnings') or [])
        for research in research_by_symbol.values()
    )
    lines = [
        'CONTEXT REVIEW',
        '',
        'Research summary: '
        f'{len(validated_symbols)} ETFs have validated research '
        f'({len(current_validated)} researched and validated this run, '
        f'{len(cached_validated)} supplied by validated cache); '
        f'{charged_search_attempts} charged ETF search attempts, '
        f'{sum(structural_repairs_by_symbol.values())} structural repair '
        f'attempts, {len(batch_research_diagnostics)} research batches, '
        f'{warning_count} accepted exclusion-only evidence warnings.',
        'Selected ETFs: ' + (', '.join(selected_symbols) or 'None'),
        '',
    ]
    headers = [
        'QVM Rank', 'Symbol', 'Sector Group', 'Reversal Risk / Status',
        'Sector Selected After', 'Total Selected After', 'Risk Derivation',
        'Explanation',
    ]
    rows = [
        [
            decision['qvm_rank'],
            decision['symbol'],
            decision['sector_group'],
            decision['status'],
            decision['sector_selected_after'],
            decision['total_selected_after'],
            decision.get('risk_derivation'),
            decision['explanation'],
        ]
        for decision in decision_ledger
    ]
    table = pd.DataFrame(rows, columns=headers)
    try:
        rendered_table = table.to_markdown(index=False)
    except ImportError:
        rendered_table = table.to_string(index=False)
    return '\n'.join(lines) + rendered_table


def build_decision_ledger(candidates, research_by_symbol, target, max_per_sector,
                          max_moderate_event, research_disposition=None):
    selected, ledger, sector_counts, event_counts = [], [], {}, {}
    research_disposition = research_disposition or {}
    for rank, candidate in enumerate(candidates, start=1):
        if len(selected) >= target:
            break
        symbol = str(candidate['Symbol']).upper()
        research = research_by_symbol.get(symbol)
        status, reason = None, None
        sector = candidate_sector_group(candidate)
        sector_key = sector.casefold()
        if not research:
            disposition = research_disposition.get(symbol, 'NOT_NEEDED')
            if disposition == 'VALIDATION_FAILED':
                status = 'NOT SELECTED — RESEARCH VALIDATION FAILED'
                reason = 'Research was attempted but did not validate.'
            elif disposition == 'SECTOR_CAPACITY':
                status = 'NOT RESEARCHED — SECTOR CAPACITY'
                reason = f'{max_per_sector} higher-ranked ETFs already use {sector}.'
            elif disposition == 'PROVISIONAL_GROUP_CAP':
                status = 'NOT RESEARCHED — PROVISIONAL GROUP CAP'
                reason = 'Deferred to avoid over-researching one provisional group.'
            elif disposition == 'API_UNAVAILABLE':
                status = 'NOT SELECTED — GEMINI API UNAVAILABLE'
                reason = 'Gemini never returned usable ETF-level research for this candidate.'
            elif disposition == 'RESEARCH_BUDGET_EXHAUSTED':
                status = 'NOT RESEARCHED — RESEARCH BUDGET EXHAUSTED'
                reason = 'The run exhausted its logical Gemini research-call budget before reaching this candidate.'
            else:
                status = 'NOT RESEARCHED — NOT NEEDED'
                reason = 'The portfolio filled before research was needed.'
        else:
            risk = str(research.get('reversal_risk') or '').upper()
            event_key = normalized_risk_event_key(research)
            if not research.get('eligible', True):
                status = 'NOT SELECTED — INELIGIBLE'
                reason = research.get('eligibility_reason') or research.get('explanation')
            elif risk not in SELECTABLE_RISKS:
                status = f'NOT SELECTED — {risk}'
                reason = research.get('explanation')
            elif sector_counts.get(sector_key, 0) >= max_per_sector:
                status = 'SKIPPED — SECTOR CAPACITY'
                reason = f'{max_per_sector} selected ETFs already use the {sector} sector/category.'
            elif (
                risk == 'MODERATE' and event_key
                and event_counts.get(event_key, 0) >= max_moderate_event
            ):
                status = 'SKIPPED — RISK-EVENT CAPACITY'
                reason = f'The {event_key[0]} / {event_key[1]} group is already full.'
            else:
                selected.append({'candidate': candidate, 'research': research})
                sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
                if risk == 'MODERATE' and event_key:
                    event_counts[event_key] = event_counts.get(event_key, 0) + 1
                status = f'SELECTED — {risk}'
                reason = research.get('explanation')
        ledger.append({
            'qvm_rank': rank,
            'symbol': symbol,
            'sector_group': sector,
            'status': status,
            'sector_selected_after': sector_counts.get(sector_key, 0),
            'total_selected_after': len(selected),
            'risk_derivation': risk_derivation_summary(research) if research else None,
            'explanation': reason,
        })
    return selected, ledger


def build_recommendations_table(selected):
    def format_metric(value, percent=False):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 'N/A'
        if not np.isfinite(number):
            return 'N/A'
        return f'{number:.2f}%' if percent else f'{number:.2f}'

    rows = []
    for item in selected:
        candidate, research = item['candidate'], item['research']
        symbol = html.escape(str(candidate['Symbol']))
        name = html.escape(str(candidate.get('Name') or research.get('fund_name') or symbol))
        link = f'https://finance.yahoo.com/quote/{symbol}/'
        rows.append(
            '<tr>'
            f'<td><a href="{link}" target="_blank">{symbol}</a></td>'
            f'<td><a href="{link}" target="_blank">{name}</a></td>'
            f'<td>{html.escape(candidate_sector_group(candidate))}</td>'
            f'<td>{format_metric(candidate.get("3M Return"), percent=True)}</td>'
            f'<td>{format_metric(candidate.get("1Y Return"), percent=True)}</td>'
            f'<td>{format_metric(candidate.get("QVMScore"))}</td>'
            f'<td><strong>{html.escape(str(research["reversal_risk"]))}</strong></td>'
            '</tr>'
        )
    return (
        '<table class="recommendations-table"><thead><tr>'
        '<th>Symbol</th><th>Name</th><th>Sector</th>'
        '<th>3 Month Return</th><th>1 Yr Return</th>'
        '<th>QVM Score</th><th>Reversal Risk</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table>'
    )


def yahoo_link(symbol, label):
    safe_symbol = html.escape(str(symbol), quote=True)
    safe_label = html.escape(str(label))
    return (
        f'<a href="https://finance.yahoo.com/quote/{safe_symbol}/" '
        f'target="_blank"><strong>{safe_label}</strong></a>'
    )


def build_fallback_summary(market_context, selected):
    by_sector = {}
    for item in selected:
        sector = candidate_sector_group(item['candidate'])
        by_sector.setdefault(sector, []).append(item)
    paragraphs = [
        '<h2>Market Chat:</h2>',
        f'<p>{html.escape(str(market_context.get("market_summary") or "Current market research was completed."))}</p>'
    ]
    for sector, items in by_sector.items():
        symbol_links = ', '.join(
            yahoo_link(item['candidate']['Symbol'], item['candidate']['Symbol'])
            for item in items
        )
        sector_drivers = list(dict.fromkeys(
            str(item['research'].get('current_driver_evidence') or '').strip()
            for item in items
            if str(item['research'].get('current_driver_evidence') or '').strip()
        ))
        sector_sentence = (
            sector_drivers[0]
            if sector_drivers
            else f'{sector} contains the strongest qualifying ETFs in this category.'
        )
        discussions = []
        for item in items:
            candidate, research = item['candidate'], item['research']
            symbol_link = yahoo_link(candidate['Symbol'], candidate['Symbol'])
            name_link = yahoo_link(
                candidate['Symbol'],
                candidate.get('Name') or research.get('fund_name') or candidate['Symbol'],
            )
            summary = research.get('holdings_evidence') or research.get('explanation')
            attractive = ' '.join(filter(None, [
                str(research.get('current_driver_evidence') or '').strip(),
                str(research.get('continuation_outlook') or '').strip(),
            ]))
            risk_detail = (
                research.get('reversal_mechanism')
                or research.get('material_effect')
                or research.get('risk_basis')
            )
            discussions.append(
                f'{symbol_link} ({name_link}) provides '
                f'{html.escape(str(summary))} It is attractive because '
                f'{html.escape(str(attractive or research.get("explanation") or "its current evidence remains supportive"))} '
                f'Its reversal risk is <strong>{html.escape(str(research["reversal_risk"]))}</strong>: '
                f'{html.escape(str(risk_detail or "no material current reversal mechanism was identified"))}.'
            )
        paragraphs.append(
            f'<p><strong>{html.escape(sector)}</strong> ({symbol_links}): '
            f'{html.escape(sector_sentence)} '
            + ' '.join(discussions)
            + '</p>'
        )
    return '<div class="summary">' + ''.join(paragraphs) + '</div>'


def build_summary_input(selected):
    """Keep the prose-only Gemini call small and frozen."""
    compact = []
    for item in selected:
        candidate = item['candidate']
        research = item['research']
        compact.append({
            'symbol': candidate['Symbol'],
            'name': candidate.get('Name') or research.get('fund_name'),
            'sector': candidate_sector_group(candidate),
            'reversal_risk': research.get('reversal_risk'),
            'explanation': research.get('explanation'),
            'holdings_evidence': research.get('holdings_evidence'),
            'current_driver_evidence': research.get('current_driver_evidence'),
            'continuation_outlook': research.get('continuation_outlook'),
            'reversal_mechanism': research.get('reversal_mechanism'),
            'material_effect': research.get('material_effect'),
        })
    return compact


def validate_summary_response(data, selected):
    summary_html = data.get('summary_html') if isinstance(data, dict) else None
    if not isinstance(summary_html, str) or not summary_html.strip():
        raise ValueError('ETF summary response has no summary_html.')
    div_match = re.fullmatch(
        r'\s*<div\s+class=["\']summary["\']\s*>(.*)</div>\s*',
        summary_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not div_match:
        raise ValueError('ETF summary must contain exactly one summary div.')
    body = div_match.group(1)

    def plain_text(fragment):
        return html.unescape(
            re.sub(r'<[^>]+>', ' ', str(fragment))
        ).replace('\xa0', ' ').strip()

    heading_match = re.match(
        r'\s*<h2\s*>\s*Market Chat:\s*</h2>',
        body,
        flags=re.IGNORECASE,
    )
    if not heading_match:
        raise ValueError('ETF summary must begin with the Market Chat heading.')
    paragraphs = re.findall(
        r'<p\b[^>]*>(.*?)</p>',
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    sectors = list(dict.fromkeys(
        candidate_sector_group(item['candidate']) for item in selected
    ))
    if len(paragraphs) != len(sectors) + 1:
        raise ValueError(
            'ETF summary must contain one market paragraph and exactly one '
            'paragraph per represented sector/category.'
        )
    expected_symbols = {
        str(item['candidate']['Symbol']).upper() for item in selected
    }
    linked_symbols = {
        match.upper()
        for match in re.findall(
            r'https://finance\.yahoo\.com/quote/([^/]+)/',
            summary_html,
            flags=re.IGNORECASE,
        )
    }
    if linked_symbols != expected_symbols:
        raise ValueError(
            f'ETF summary symbol mismatch: expected {sorted(expected_symbols)}, '
            f'received {sorted(linked_symbols)}.'
        )
    for sector, paragraph in zip(sectors, paragraphs[1:]):
        sector_match = re.match(
            r'\s*<strong>(.*?)</strong>',
            paragraph,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not sector_match or plain_text(sector_match.group(1)) != sector:
            raise ValueError(f'ETF summary sector order mismatch at {sector}.')
        items = [
            item for item in selected
            if candidate_sector_group(item['candidate']) == sector
        ]
        paragraph_text = plain_text(paragraph)
        for item in items:
            candidate, research = item['candidate'], item['research']
            symbol = str(candidate['Symbol'])
            name = str(candidate.get('Name') or research.get('fund_name') or symbol)
            matching_links = [
                plain_text(link_text)
                for linked_symbol, link_text in re.findall(
                    r'<a\b[^>]*href=["\']https://finance\.yahoo\.com/quote/'
                    r'([^/]+)/["\'][^>]*>(.*?)</a>',
                    paragraph,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if linked_symbol.upper() == symbol.upper()
            ]
            if symbol not in matching_links or name not in matching_links:
                raise ValueError(
                    f'ETF summary must link both the symbol and name for {symbol}.'
                )
            if str(research['reversal_risk']) not in paragraph_text:
                raise ValueError(f'ETF summary omitted {symbol} reversal risk.')
    return summary_html.strip()


start_time = time.perf_counter()
config_path = SCRIPT_DIR / 'etf_config.yml'
with config_path.open('r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
required_prompt_keys = {
    'prompt_market_context', 'prompt_etf_batch', 'prompt_html_summary'
}
missing_prompt_keys = sorted(required_prompt_keys - set(config or {}))
if missing_prompt_keys:
    raise ValueError(
        f'Incompatible ETF config loaded from {config_path}. Missing: '
        + ', '.join(missing_prompt_keys)
        + '. Use the etf_config.yml distributed with this recommender.'
    )
TOP_QVM_CACHE_EXPIRY_HOURS = config.get(
    'top_qvm_cache_expiry_hours', TOP_QVM_CACHE_EXPIRY_HOURS
)
excluded_keywords = config['excluded_keywords']
min_52_week_change = config['min_52_week_change']
min_3_month_return = config['min_3_month_return']
max_retries = config['max_retries']
initial_delay = config['initial_delay']
model_primary = config['model_primary']
model_fallback = config['model_fallback']
thinking_budget = config.get('thinking_budget', 12288)
summary_thinking_budget = config.get('summary_thinking_budget', 4096)
gemini_max_output_tokens = config.get('gemini_max_output_tokens', 49152)
gemini_batch_size = config.get('gemini_batch_size', 15)
target_selected_etfs = config.get('target_selected_etfs', 10)
max_etfs_per_sector_group = config.get('max_etfs_per_sector_group', 2)
max_candidates_per_provisional_group = config.get(
    'max_candidates_per_provisional_group', 4
)
max_moderate_per_risk_event = config.get('max_moderate_per_risk_event', 2)
max_gemini_calls_per_run = config.get('max_gemini_calls_per_run', 7)
max_etf_research_calls_per_run = config.get('max_etf_research_calls_per_run', 5)
reserved_summary_calls = config.get('reserved_summary_calls', 1)
max_transient_api_attempts = config.get('max_transient_api_attempts', 3)
max_transient_delay = config.get('max_transient_delay', 60)
summary_reservation_releasable_on_shortfall = config.get(
    'summary_reservation_releasable_on_shortfall', True
)
transport_retry_batch_fraction = float(config.get('transport_retry_batch_fraction', 0.67))
max_fresh_research_attempts_per_etf = config.get(
    'max_fresh_research_attempts_per_etf', 2
)
max_structural_repairs_per_etf = config.get(
    'max_structural_repairs_per_etf', 2
)
top_rank_basic_metadata_repair_limit = int(
    config.get('top_rank_basic_metadata_repair_limit', 20)
)
min_structural_repair_batch_size = config.get(
    'min_structural_repair_batch_size', 3
)
research_cache_file = config.get(
    'etf_research_cache_file', 'caches/gemini_etf_research_cache.json'
)
market_context_cache_file = config.get(
    'market_context_cache_file', 'caches/gemini_market_context_cache.json'
)
research_cache_hours = config.get('gemini_research_cache_hours', 12)
market_context_cache_hours = config.get('market_context_cache_hours', 12)
market_context_schema_version = int(config.get('market_context_schema_version', 1))
market_context_prompt_version = int(config.get('market_context_prompt_version', 1))
cache_version = config.get('cache_version', 1)
research_cache_version = config.get(
    'etf_research_cache_version', cache_version
)
benchmark_etfs = [
    str(symbol).strip().upper()
    for symbol in config['benchmark_etfs']
]
if not benchmark_etfs:
    raise ValueError('benchmark_etfs must contain at least one symbol.')
max_info_calls = config.get('max_info_calls', 500)
top_n = config.get('qvm_top_n', 100)
research_pool_limit = min(
    top_n,
    int(config.get('research_candidate_pool_limit', top_n)),
)
qvm_weights = config.get('qvm_weights', {'Quality': 0.25, 'Value': 0.15, 'Momentum': 0.6})
min_quality = config.get('min_quality', 35)
top_etfs = load_top_qvm_cache(top_n, research_pool_limit)
if top_etfs is None:
    seed_histories = download_price_history(benchmark_etfs, period='13mo')
    seed_benchmark_metrics = get_benchmark_data(seed_histories, benchmark_etfs)
    discovery_1y, discovery_3m = get_discovery_thresholds(
        config,
        seed_benchmark_metrics,
        benchmark_etfs,
    )
    print(
        'Benchmark-relative discovery floors: '
        f'1Y {discovery_1y:.2f}%, 3M {discovery_3m:.2f}%.'
    )
    df = fetch_etf_universe(discovery_1y, max_retries)
    print(f'\nRaw ETF universe: {len(df)}')
    df = apply_basic_etf_filters(
        df,
        excluded_keywords,
        discovery_1y,
        discovery_3m,
    )
    print(f'After basic ETF filtering: {len(df)}')
    benchmark_rows = pd.DataFrame({'Symbol': benchmark_etfs, 'Name': benchmark_etfs})
    df = pd.concat([df, benchmark_rows], ignore_index=True)
    df = df.drop_duplicates(subset='Symbol', keep='first').reset_index(drop=True)
    df_yf, histories = append_etf_yfinance_data(
        df,
        max_info_calls=max_info_calls,
        priority_symbols=benchmark_etfs,
    )
    df_yf = apply_post_metadata_mandate_filter(
        df_yf,
        benchmark_etfs,
        config.get('excluded_metadata_patterns', []),
    )
    benchmark_metrics = get_benchmark_data(histories, benchmark_etfs)
    df_yf = add_benchmark_relative_metrics(df_yf, benchmark_metrics, benchmark_etfs)
    is_benchmark = df_yf['Symbol'].isin(benchmark_etfs)
    if 'Price' in df_yf.columns:
        price_eligible = (
            df_yf['Price'].notna()
            & (df_yf['Price'] >= config.get('min_price', 5.0))
        )
        df_yf = df_yf[is_benchmark | price_eligible].copy()
        is_benchmark = df_yf['Symbol'].isin(benchmark_etfs)
    if 'AverageVolume' in df_yf.columns:
        volume_eligible = (
            df_yf['AverageVolume'].isna()
            | (
                df_yf['AverageVolume']
                >= config.get('min_average_volume', 100000)
            )
        )
        df_yf = df_yf[is_benchmark | volume_eligible].copy()
        is_benchmark = df_yf['Symbol'].isin(benchmark_etfs)
    if 'AUM' in df_yf.columns:
        aum_eligible = (
            df_yf['AUM'].isna()
            | (df_yf['AUM'] >= config.get('min_aum', 100000000))
        )
        df_yf = df_yf[is_benchmark | aum_eligible].copy()

    scored_etfs = score_etf_qvm(
        df_yf,
        top_n=len(df_yf),
        weights=qvm_weights,
        min_quality=0,
    )
    benchmark_df = scored_etfs[
        scored_etfs['Symbol'].isin(benchmark_etfs)
    ].copy()
    print(
        'Final benchmark-relative floors: '
        f'3M {benchmark_df["3M Return"].mean():.2f}%, '
        f'1Y {benchmark_df["1Y Return"].mean():.2f}%.'
    )
    superiority_mask = get_benchmark_superiority_mask(
        scored_etfs,
        benchmark_df,
        benchmark_etfs,
    )
    competitive_mask = get_benchmark_competitive_mask(
        scored_etfs,
        benchmark_df,
        benchmark_etfs,
        qvm_tolerance=config.get('benchmark_backfill_qvm_tolerance', 10.0),
        three_month_tolerance=config.get(
            'benchmark_backfill_3m_tolerance', 5.0
        ),
        one_year_tolerance=config.get(
            'benchmark_backfill_1y_tolerance', 10.0
        ),
    ) & (scored_etfs['QualityScore'] >= min_quality)

    # Match the stock recommender: preserve the ranked top-QVM stream first,
    # then let research, risk and portfolio rules decide final eligibility.
    candidate_etfs, admission_counts = build_ranked_etf_research_pool(
        scored_etfs,
        benchmark_etfs,
        superiority_mask,
        competitive_mask,
        min_quality,
        top_n,
        research_pool_limit,
    )
    print(
        f'ETF research candidates: {len(candidate_etfs)} '
        f'({admission_counts["strict"]} strict benchmark-superior, '
        f'{admission_counts["competitive_backfill"]} benchmark-competitive '
        'backfill, '
        f'{admission_counts["top_qvm_reserve"]} additional top-QVM reserve; '
        f'top-QVM target: {top_n}).'
    )
    if not benchmark_df.empty:
        top_etfs = pd.concat(
            [candidate_etfs, benchmark_df],
            ignore_index=True,
        )
        top_etfs = top_etfs.drop_duplicates(subset='Symbol', keep='first')
    else:
        top_etfs = candidate_etfs
    top_etfs = top_etfs.sort_values(
        'QVMScore',
        ascending=False,
        na_position='last',
    ).reset_index(drop=True)
    top_etfs['Role'] = np.where(
        top_etfs['Symbol'].isin(benchmark_etfs),
        'BENCHMARK',
        'CANDIDATE',
    )
    save_top_qvm_cache(top_etfs, top_n, research_pool_limit)

# Reapply the mandate filter to cached QVM data as a defensive check.
top_etfs = apply_post_metadata_mandate_filter(
    top_etfs,
    benchmark_etfs,
    config.get('excluded_metadata_patterns', []),
)

# Structured holdings and asset-class data are fetched only for the compact QVM
# research universe, not for every ETF seen during discovery.
top_etfs = append_structured_fund_data(
    top_etfs,
    delay=config.get('structured_fund_data_delay', 0.2),
)
top_etfs = apply_post_metadata_mandate_filter(
    top_etfs, benchmark_etfs, config.get('excluded_metadata_patterns', [])
)
top_etfs = apply_structured_equity_filter(
    top_etfs,
    benchmark_etfs,
    config.get('min_equity_asset_weight', 0.80),
)
top_etfs = add_etf_portfolio_classifications(
    top_etfs, config.get('canonical_sector_min_weight', 0.50)
)
print('\nTop ETF QVM candidates:')
display_columns = [
    col
    for col in [
        'Role',
        'Symbol',
        'QVMScore',
        'QualityScore',
        'ValueScore',
        'MomentumScore',
        'BenchmarkRelativeScore',
        '3M Return',
        '6M Return',
        '1Y Return',
    ]
    if col in top_etfs.columns
]
print(top_etfs[display_columns].head(top_n).to_string(index=False))
inspection_columns = [
    col
    for col in [
        'Role',
        'Symbol',
        'Name',
        'Category',
        'QVMScore',
        'QualityScore',
        'ValueScore',
        'MomentumScore',
        'BenchmarkRelativeScore',
        'AUM',
        'ExpenseRatio',
        'AverageVolume',
        'Volatility 1Y',
        'TrailingPE',
        'PriceToBook',
        'FundOverview',
        'AssetClasses',
        'SectorWeightings',
        'TopHoldings',
        'CanonicalSector',
        'StyleCategory',
        'PortfolioGroup',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
    ]
    if col in top_etfs.columns
]
inspection_columns += [
    f'{period} Excess vs {benchmark}'
    for benchmark in benchmark_etfs
    for period in ['1M', '3M', '6M', '9M', '1Y']
    if f'{period} Excess vs {benchmark}' in top_etfs.columns
]
with open('caches/top_qvm_etfs.md', 'w', encoding='utf-8') as f:
    f.write(top_etfs[inspection_columns].to_markdown(index=False))

gemini_columns = [
    col
    for col in [
        'Role', 'Symbol', 'Name', 'Category', 'FundFamily', 'QVMScore',
        'ResearchAdmission',
        'QualityScore', 'ValueScore', 'MomentumScore',
        'BenchmarkRelativeScore', 'AUM', 'ExpenseRatio', 'AverageVolume',
        'Volatility 1Y', 'TrailingPE', 'PriceToBook', '1M Return',
        '3M Return', '6M Return', '9M Return', '1Y Return',
        'FundOverview', 'AssetClasses', 'SectorWeightings', 'TopHoldings',
        'CanonicalSector', 'StyleCategory', 'PortfolioGroup',
    ]
    if col in top_etfs.columns
]
gemini_columns += [
    f'{period} Excess vs {benchmark}'
    for benchmark in benchmark_etfs
    for period in ['1M', '3M', '6M', '9M', '1Y']
    if f'{period} Excess vs {benchmark}' in top_etfs.columns
]
df_gemini = top_etfs[gemini_columns].copy()
candidate_df = df_gemini[df_gemini['Role'] == 'CANDIDATE'].copy()
candidate_df = candidate_df.sort_values('QVMScore', ascending=False).head(top_n)
candidate_records = dataframe_records(candidate_df)
if len(candidate_records) < target_selected_etfs:
    print(
        'ETF candidate warning: '
        f'Only {len(candidate_records)} top-QVM ETF candidates remain after '
        'structured mandate and asset-class filtering; '
        f'the run will publish fewer than {target_selected_etfs} selections if '
        'the validated candidate pool cannot fill the target.'
    )

client = initialize_gemini_client()
request_budget = GeminiRequestBudget(
    max_gemini_calls_per_run,
    max_etf_research_calls_per_run,
    reserved_summary_calls,
    release_summary_for_research=False,
)
call_diagnostics = []
batch_research_diagnostics = []
models_used = []

# Research the shared market backdrop once and cache it separately from ETFs.
market_prompt = config['prompt_market_context'].rstrip() + (
    f'\n\nCURRENT_DATE_UTC: {datetime.now(UTC).date().isoformat()}\n'
)
market_prompt_hash = stable_json_hash({
    'schema_version': market_context_schema_version,
    'prompt_version': market_context_prompt_version,
    'model': model_primary,
    'prompt': market_prompt,
})
market_cache = load_json_object(market_context_cache_file)
market_context = None
cached_market_data = (
    market_cache.get('data') or market_cache.get('market_context')
    if isinstance(market_cache, dict) else None
)
cache_model = market_cache.get('model') if isinstance(market_cache, dict) else None
current_shared_contract = (
    market_cache.get('schema_version') == market_context_schema_version
    and market_cache.get('prompt_hash') == market_prompt_hash
)
compatible_stock_contract = (
    market_cache.get('version') == cache_version
    and isinstance(market_cache.get('market_context'), dict)
)
if (
    (current_shared_contract or compatible_stock_contract)
    and cache_model in {model_primary, model_fallback}
    and cache_entry_is_fresh(market_cache, market_context_cache_hours)
):
    try:
        market_context = validate_market_context(cached_market_data)
        timestamp = parse_utc_timestamp(market_cache.get('timestamp') or market_cache.get('created_at'))
        age = datetime.now(UTC) - timestamp
        label = 'shared' if current_shared_contract else 'compatible stock'
        print(f'Using validated {label} market context cache ({age.total_seconds() / 3600:.1f}h old).')
    except Exception as exc:
        print(f'Ignoring invalid shared market context cache: {exc}')
if market_context is None:
    data, used_model, metadata = call_gemini_json(
        client,
        model_primary,
        market_prompt,
        build_gemini_config(
            thinking_budget,
            enable_search=True,
            response_mime_type=None,
            max_output_tokens=gemini_max_output_tokens,
        ),
        'ETF market context',
        request_budget,
        'context',
        max_transient_api_attempts,
        initial_delay,
        max_transient_delay,
    )
    if not metadata.get('search_queries') and metadata.get('tool_tokens', 0) <= 0:
        raise RuntimeError('Gemini returned ungrounded ETF market context.')
    market_context = validate_market_context(data)
    models_used.append(used_model)
    call_diagnostics.append({'stage': 'market context', **metadata})
    if len(metadata.get('search_queries', [])) > 5:
        print(f'Market-context search overrun: Gemini exposed {len(metadata.get("search_queries", []))} queries; prompt maximum is 5.')
    now_iso = datetime.now(UTC).isoformat()
    save_json_object_atomic(
        market_context_cache_file,
        {
            'version': cache_version,
            'schema_version': market_context_schema_version,
            'prompt_version': market_context_prompt_version,
            'prompt_hash': market_prompt_hash,
            'model': used_model,
            'created_at': now_iso,
            'timestamp': now_iso,
            'data': market_context,
            'market_context': market_context,
            'research_metadata': {
                'search_queries': metadata.get('search_queries', []),
                'tool_tokens': metadata.get('tool_tokens', 0),
            },
        },
    )

research_prompt_hash = stable_json_hash(
    {'version': research_cache_version, 'prompt': config['prompt_etf_batch']}
)
research_cache = load_json_object(
    research_cache_file,
    {'version': research_cache_version, 'entries': {}, 'deferred_entries': {}},
)
if research_cache.get('version') != research_cache_version:
    research_cache = {
        'version': research_cache_version,
        'entries': {},
        'deferred_entries': {},
    }
research_cache.setdefault('entries', {})
research_cache.setdefault('deferred_entries', {})

# Compact the persistent research cache before reuse/save. GitHub Actions Cache
# restores the previous runtime cache as an opaque directory, while Python
# remains authoritative for TTL freshness. Removing expired entries here keeps
# successive Actions cache generations small without changing reuse semantics.
research_cache['entries'] = {
    key: entry
    for key, entry in research_cache['entries'].items()
    if isinstance(entry, dict)
    and cache_entry_is_fresh(entry, research_cache_hours)
}
research_cache['deferred_entries'] = {
    key: entry
    for key, entry in research_cache['deferred_entries'].items()
    if isinstance(entry, dict)
    and cache_entry_is_fresh(entry, research_cache_hours)
}
save_json_object_atomic(research_cache_file, research_cache)

candidate_by_symbol = {
    str(candidate['Symbol']).upper(): candidate for candidate in candidate_records
}
cache_keys = {}
research_by_symbol = {}
for candidate in candidate_records:
    symbol = str(candidate['Symbol']).upper()
    signature = {
        key: candidate.get(key) for key in (
            'Symbol', 'Name', 'Category', 'ResearchAdmission', 'QVMScore',
            'BenchmarkRelativeScore',
            'ExpenseRatio', 'AUM', '1M Return', '3M Return', '6M Return',
            '9M Return', '1Y Return', 'FundOverview', 'AssetClasses',
            'SectorWeightings', 'TopHoldings', 'CanonicalSector',
            'StyleCategory', 'PortfolioGroup',
        )
    }
    cache_key = stable_json_hash(
        {'prompt_hash': research_prompt_hash, 'candidate': signature}
    )
    cache_keys[symbol] = cache_key
    entry = research_cache['entries'].get(cache_key)
    compatible_cache = False
    if not entry:
        matching_entries = [
            prior_entry
            for prior_entry in research_cache['entries'].values()
            if (
                isinstance(prior_entry, dict)
                and cache_entry_is_fresh(prior_entry, research_cache_hours)
                and str(
                    (prior_entry.get('research') or {}).get('symbol') or ''
                ).strip().upper() == symbol
            )
        ]
        if matching_entries:
            entry = max(
                matching_entries,
                key=lambda item: str(item.get('timestamp') or ''),
            )
            compatible_cache = True
    if entry and cache_entry_is_fresh(entry, research_cache_hours):
        try:
            research_by_symbol[symbol] = validate_etf_result(
                entry.get('research'), candidate,
                config.get('min_etf_sources', 2),
                config.get('max_etf_sources', 5),
            )
            cache_label = 'compatible validated' if compatible_cache else 'validated'
            print(f'Using {cache_label} ETF research cache for {symbol}.')
        except Exception as exc:
            print(f'Ignoring invalid ETF research cache for {symbol}: {exc}')

cursor = 0
pending_retries = []
deferred_excess_candidates = []
deferred_excess_symbols_this_run = set()
sector_capacity_skipped_symbols_this_run = set()
research_attempts_by_symbol = {}
structural_repairs_by_symbol = {}
consistency_reviewed_symbols = set()
researched_this_run = set()
search_attempts = 0
charged_search_attempts = 0
research_quota_exhausted = False
transport_failed_symbols_this_run = set()
research_budget_exhausted = False
rank_by_symbol = {
    str(candidate['Symbol']).upper(): rank
    for rank, candidate in enumerate(candidate_records, start=1)
}
selected = preview_portfolio(
    candidate_records,
    research_by_symbol,
    target_selected_etfs,
    max_etfs_per_sector_group,
    max_moderate_per_risk_event,
)
initial_conflicts = find_actionable_consistency_conflicts(
    candidate_records, research_by_symbol, selected
)
queued_initial_repairs = enqueue_consistency_repairs(
    initial_conflicts,
    candidate_by_symbol,
    research_by_symbol,
    research_cache,
    cache_keys,
    pending_retries,
    consistency_reviewed_symbols,
    structural_repairs_by_symbol,
    max_structural_repairs_per_etf,
)
if queued_initial_repairs:
    save_json_object_atomic(research_cache_file, research_cache)
    selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
    )
if len(selected) >= target_selected_etfs and not pending_retries:
    print('Validated ETF research cache already supports the full portfolio.')

request_budget.release_summary_for_research = (
    summary_reservation_releasable_on_shortfall
    and len(selected) < target_selected_etfs
)
while (
    (len(selected) < target_selected_etfs or bool(pending_retries))
    and request_budget.can_reserve('research')
    and not research_quota_exhausted
):
    if len(selected) >= target_selected_etfs and pending_retries:
        actionable_pending = []
        skipped_pending = []
        for retry in pending_retries:
            candidate = retry['candidate']
            if pending_candidate_can_improve_full_portfolio(
                candidate, selected, rank_by_symbol, max_etfs_per_sector_group
            ):
                actionable_pending.append(retry)
            else:
                skipped_pending.append(str(candidate['Symbol']).upper())
        pending_retries = actionable_pending
        if skipped_pending:
            print(
                'Skipping post-fill ETF repairs that cannot improve the ranked '
                '10-ETF portfolio: ' + ', '.join(skipped_pending)
            )
        if not pending_retries:
            break
    batch = []
    retry_symbols = set()
    retry_fresh_symbols = set()
    transport_retry_symbols = set()
    provisional_counts = {}
    dedicated_basic_repair_batch = False

    # High-ranked basic/evidence-completion failures get a compact repair-only call.
    # Do not mix them with fresh searches: mixed calls are much more likely to
    # hit MAX_TOKENS and omit the exact repair candidates we are trying to save.
    basic_repair_pending = []
    other_pending = []
    for retry in pending_retries:
        candidate = retry['candidate']
        symbol = str(candidate['Symbol']).upper()
        deferred = research_cache['deferred_entries'].get(cache_keys.get(symbol), {})
        is_basic_repair = (
            not retry.get('needs_research')
            and str(deferred.get('error') or '').startswith('BASIC_METADATA_REPAIR:')
        )
        (basic_repair_pending if is_basic_repair else other_pending).append(retry)
    if basic_repair_pending:
        dedicated_basic_repair_batch = True
        pending_retries = basic_repair_pending + other_pending
    max_transport_retries_in_batch = max(
        1, int(round(gemini_batch_size * transport_retry_batch_fraction))
    )
    deferred_transport_retries = []
    while pending_retries and len(batch) < gemini_batch_size:
        retry = pending_retries[0]
        candidate = retry['candidate']
        symbol = str(candidate['Symbol']).upper()
        deferred = research_cache['deferred_entries'].get(cache_keys.get(symbol), {})
        retry_is_basic_repair = (
            not retry.get('needs_research')
            and str(deferred.get('error') or '').startswith('BASIC_METADATA_REPAIR:')
        )
        if dedicated_basic_repair_batch and not retry_is_basic_repair:
            break
        pending_retries.pop(0)
        transport_failure = bool(retry.get('transport_failure'))
        if (
            transport_failure
            and len(transport_retry_symbols) >= max_transport_retries_in_batch
            and cursor < len(candidate_records)
        ):
            deferred_transport_retries.append(retry)
            continue
        if (
            retry['needs_research']
            and research_attempts_by_symbol.get(symbol, 0)
            >= max_fresh_research_attempts_per_etf
        ):
            continue
        if (
            not retry['needs_research']
            and structural_repairs_by_symbol.get(symbol, 0)
            >= max_structural_repairs_per_etf
        ):
            continue
        batch.append(candidate)
        retry_symbols.add(symbol)
        if retry['needs_research']:
            retry_fresh_symbols.add(symbol)
        if transport_failure:
            transport_retry_symbols.add(symbol)
        provisional = provisional_exposure_group(candidate)
        provisional_counts[provisional] = provisional_counts.get(provisional, 0) + 1
    if deferred_transport_retries:
        pending_retries = deferred_transport_retries + pending_retries
    while (
        not dedicated_basic_repair_batch
        and len(selected) < target_selected_etfs
        and cursor < len(candidate_records)
        and len(batch) < gemini_batch_size
    ):
        candidate = candidate_records[cursor]
        cursor += 1
        symbol = str(candidate['Symbol']).upper()
        if symbol in research_by_symbol or symbol in {str(x['Symbol']).upper() for x in batch}:
            continue
        if lower_ranked_candidate_blocked_by_sector_capacity(
            candidate,
            selected,
            rank_by_symbol,
            max_etfs_per_sector_group,
        ):
            sector_capacity_skipped_symbols_this_run.add(symbol)
            continue
        provisional = provisional_exposure_group(candidate)
        selectable_in_provisional = 0
        for researched_symbol, researched in research_by_symbol.items():
            researched_candidate = candidate_by_symbol.get(researched_symbol)
            if not researched_candidate:
                continue
            if provisional_exposure_group(researched_candidate) != provisional:
                continue
            if researched.get('eligible', True) and str(
                researched.get('reversal_risk') or ''
            ).upper() in SELECTABLE_RISKS:
                selectable_in_provisional += 1
        effective_provisional_cap = (
            max_candidates_per_provisional_group
            if selectable_in_provisional > 0
            else max_candidates_per_provisional_group + 2
        )
        if provisional_counts.get(provisional, 0) >= effective_provisional_cap:
            deferred_excess_candidates.append(candidate)
            deferred_excess_symbols_this_run.add(symbol)
            continue
        batch.append(candidate)
        provisional_counts[provisional] = provisional_counts.get(provisional, 0) + 1
    if (
        len(selected) < target_selected_etfs
        and cursor >= len(candidate_records)
        and len(batch) < gemini_batch_size
    ):
        deferred_round = list(deferred_excess_candidates)
        deferred_excess_candidates = []
        for candidate in deferred_round:
            if len(batch) >= gemini_batch_size:
                deferred_excess_candidates.append(candidate)
                continue
            symbol = str(candidate['Symbol']).upper()
            if symbol in research_by_symbol or symbol in {str(x['Symbol']).upper() for x in batch}:
                continue
            if lower_ranked_candidate_blocked_by_sector_capacity(
                candidate,
                selected,
                rank_by_symbol,
                max_etfs_per_sector_group,
            ):
                sector_capacity_skipped_symbols_this_run.add(symbol)
                continue
            provisional = provisional_exposure_group(candidate)
            if provisional_counts.get(provisional, 0) >= max_candidates_per_provisional_group:
                deferred_excess_candidates.append(candidate)
                continue
            batch.append(candidate)
            provisional_counts[provisional] = provisional_counts.get(provisional, 0) + 1
    if (
        len(selected) < target_selected_etfs
        and cursor >= len(candidate_records)
        and len(batch) < gemini_batch_size
    ):
        # Diversity is advisory. Once no ordinary candidates remain, fill every
        # open slot from the deferred ranked stream instead of making a smaller
        # call solely because a provisional group reached its soft cap.
        while deferred_excess_candidates and len(batch) < gemini_batch_size:
            candidate = deferred_excess_candidates.pop(0)
            symbol = str(candidate['Symbol']).upper()
            if symbol in research_by_symbol or symbol in {
                str(item['Symbol']).upper() for item in batch
            }:
                continue
            if lower_ranked_candidate_blocked_by_sector_capacity(
                candidate,
                selected,
                rank_by_symbol,
                max_etfs_per_sector_group,
            ):
                sector_capacity_skipped_symbols_this_run.add(symbol)
                continue
            batch.append(candidate)
    if (
        len(selected) < target_selected_etfs
        and not batch
        and deferred_excess_candidates
    ):
        # All remaining candidates share already-full provisional groups. Relax
        # the advisory batching cap so the ranked stream can still finish.
        remaining_deferred = []
        for candidate in deferred_excess_candidates:
            symbol = str(candidate['Symbol']).upper()
            if lower_ranked_candidate_blocked_by_sector_capacity(
                candidate,
                selected,
                rank_by_symbol,
                max_etfs_per_sector_group,
            ):
                sector_capacity_skipped_symbols_this_run.add(symbol)
                continue
            if len(batch) < gemini_batch_size:
                batch.append(candidate)
            else:
                remaining_deferred.append(candidate)
        deferred_excess_candidates = remaining_deferred
    if not batch:
        remaining_calls = (
            request_budget.research_limit - request_budget.research_used
        )
        print(
            'ETF candidate reserve exhausted: '
            f'{len(candidate_records)} total candidates, '
            f'{len(research_by_symbol)} with validated research, '
            f'{len(selected)}/{target_selected_etfs} selectable, '
            f'{len(sector_capacity_skipped_symbols_this_run)} lower-ranked '
            'candidates skipped behind full sectors, '
            f'{remaining_calls} research calls still available.'
        )
        break

    fresh_symbols = [
        str(candidate['Symbol']).upper() for candidate in batch
        if (
            str(candidate['Symbol']).upper() not in retry_symbols
            or str(candidate['Symbol']).upper() in retry_fresh_symbols
        )
    ]
    repair_only_symbols = retry_symbols - retry_fresh_symbols
    if (
        not dedicated_basic_repair_batch
        and not fresh_symbols
        and len(batch) < min_structural_repair_batch_size
        and len(selected) + len(batch) < target_selected_etfs
    ):
        print(
            'Deferring undersized structural-repair-only batch because even '
            'perfect repairs could not complete the portfolio: '
            + ', '.join(str(candidate['Symbol']) for candidate in batch)
        )
        break
    retry_payload = []
    fresh_retry_payload = []
    prior_grounding_urls = []
    for candidate in batch:
        symbol = str(candidate['Symbol']).upper()
        deferred = research_cache['deferred_entries'].get(cache_keys[symbol], {})
        if symbol in retry_fresh_symbols:
            fresh_retry_payload.append({
                'symbol': symbol,
                'prior_failure': deferred.get('error'),
            })
        if symbol in repair_only_symbols and deferred.get('draft'):
            validation_error = str(deferred.get('error') or '')
            repair_type = (
                'SOURCE_ONLY'
                if is_source_validation_error(validation_error)
                else 'GENERAL'
            )
            retry_payload.append({
                'symbol': symbol,
                'repair_type': repair_type,
                'validation_error': validation_error,
                'prior_draft': deferred.get('draft'),
                'peer_classifications': deferred.get(
                    'peer_classifications', []
                ),
            })
            prior_grounding_urls.extend(deferred.get('grounding_source_urls', []))
    prior_grounding_urls = list(dict.fromkeys(prior_grounding_urls))[
        :config.get('max_grounding_urls_for_repair', 60)
    ]
    ranks = [rank_by_symbol[str(candidate['Symbol']).upper()] for candidate in batch]
    stage = (
        f'ETF evidence completion repair ranks {min(ranks)}-{max(ranks)}'
        if dedicated_basic_repair_batch
        else f'ETF batch ranks {min(ranks)}-{max(ranks)}'
    )
    prompt = (
        config['prompt_etf_batch'].rstrip()
        + '\n\nMARKET_CONTEXT:\n'
        + json.dumps(market_context, ensure_ascii=False)
        + '\n\nCANDIDATES:\n'
        + json.dumps(batch, ensure_ascii=False)
        + '\n\nFRESH_RESEARCH_REQUIRED_FOR:\n'
        + json.dumps(fresh_symbols)
        + '\n\nFRESH_RESEARCH_RETRY_REASONS:\n'
        + json.dumps(fresh_retry_payload, ensure_ascii=False)
        + '\n\nREPAIR_WITHOUT_NEW_SEARCH:\n'
        + json.dumps(retry_payload, ensure_ascii=False)
        + '\n\nPRIOR_GROUNDING_SOURCE_URLS:\n'
        + json.dumps(prior_grounding_urls, ensure_ascii=False)
    )
    print('\n...calling Gemini for ETF batch: '
          + ', '.join(str(candidate['Symbol']) for candidate in batch) + '...\n')
    try:
        data, used_model, metadata = call_gemini_json(
            client,
            model_primary,
            prompt,
            build_gemini_config(
                thinking_budget,
                enable_search=bool(fresh_symbols),
                response_mime_type=(
                    None if fresh_symbols else 'application/json'
                ),
                max_output_tokens=gemini_max_output_tokens,
            ),
            stage,
            request_budget,
            'research',
            max_transient_api_attempts,
            initial_delay,
            max_transient_delay,
            allow_partial=True,
        )
        models_used.append(used_model)
        call_diagnostics.append({'stage': stage, **metadata})
        if fresh_symbols and not metadata.get('search_queries') and metadata.get('tool_tokens', 0) <= 0:
            raise RuntimeError('Gemini returned an ungrounded ETF research batch.')
        search_attempts += len(fresh_symbols)
        data, applied_repair_patches = merge_structural_repair_patches(
            data,
            retry_payload,
        )
        if repair_only_symbols:
            print('ETF structural-repair response inspection:')
            for symbol in sorted(repair_only_symbols):
                fields = applied_repair_patches.get(symbol)
                if fields is not None:
                    repair_request = next(
                        (
                            item for item in retry_payload
                            if str(item.get('symbol') or '').upper() == symbol
                        ),
                        {},
                    )
                    repair_type = repair_request.get('repair_type', 'GENERAL')
                    print(
                        f'  {symbol}: {repair_type} compact patch merged; fields='
                        + (', '.join(fields) or 'none')
                    )
                else:
                    print(
                        f'  {symbol}: no compact patch; checking for a full '
                        'replacement result.'
                    )
        valid, errors, drafts = validate_etf_batch(
            data,
            batch,
            config.get('min_etf_sources', 2),
            config.get('max_etf_sources', 5),
        )
        raw_result_shapes = {}
        for candidate in batch:
            symbol = str(candidate['Symbol']).upper()
            draft = drafts.get(symbol)
            raw_result_shapes[symbol] = {
                'present': isinstance(draft, dict),
                'field_count': len(draft) if isinstance(draft, dict) else 0,
                'sources_shape': source_shape(
                    draft.get('sources') if isinstance(draft, dict) else None
                ),
                'benchmark_assessment': (
                    draft.get('benchmark_assessment') if isinstance(draft, dict) else None
                ),
                'mandate_assessment': (
                    draft.get('mandate_assessment') if isinstance(draft, dict) else None
                ),
                'us_equity_weight_estimate': (
                    draft.get('us_equity_weight_estimate')
                    if isinstance(draft, dict) else None
                ),
                'mandate_basis': (
                    draft.get('mandate_basis') if isinstance(draft, dict) else None
                ),
                'mechanism_status': (
                    draft.get('mechanism_status') if isinstance(draft, dict) else None
                ),
                'evidence_scope': (
                    draft.get('evidence_scope') if isinstance(draft, dict) else None
                ),
                'adverse_change_observed': (
                    draft.get('adverse_change_observed')
                    if isinstance(draft, dict) else None
                ),
                'normalization_probability': (
                    draft.get('normalization_probability')
                    if isinstance(draft, dict) else None
                ),
                'risk_materiality': (
                    draft.get('risk_materiality') if isinstance(draft, dict) else None
                ),
                'reversal_mechanism_chars': (
                    len(str(draft.get('reversal_mechanism') or '').strip())
                    if isinstance(draft, dict) else 0
                ),
            }
        if errors:
            print('ETF response-shape inspection:')
            for symbol in [str(item['Symbol']).upper() for item in batch]:
                if symbol in errors:
                    shape = raw_result_shapes[symbol]
                    print(
                        f'  {symbol}: present={shape["present"]}, '
                        f'fields={shape["field_count"]}, '
                        f'sources={shape["sources_shape"]}, '
                        f'benchmark={shape["benchmark_assessment"]}, '
                        f'mandate={shape["mandate_assessment"]}, '
                        f'us_equity_weight={shape["us_equity_weight_estimate"]}, '
                        f'mandate_basis={shape["mandate_basis"]}, '
                        f'mechanism={shape["mechanism_status"]}, '
                        f'evidence_scope={shape["evidence_scope"]}, '
                        f'adverse_observed={shape["adverse_change_observed"]}, '
                        f'mechanism_chars={shape["reversal_mechanism_chars"]}, '
                        f'probability={shape["normalization_probability"]}, '
                        f'materiality={shape["risk_materiality"]}'
                    )
        for candidate in batch:
            symbol = str(candidate['Symbol']).upper()
            missing_exposed_search = (
                symbol in fresh_symbols
                and bool(metadata.get('search_queries'))
                and not any(
                query_matches_etf(query, candidate)
                for query in metadata.get('search_queries', [])
                )
            )
            if missing_exposed_search:
                valid.pop(symbol, None)
                errors[symbol] = 'Gemini did not expose an ETF-specific search.'
                research_attempts_by_symbol.setdefault(symbol, 0)
                print(f'Refunding unsearched ETF attempt [{symbol}].')
            else:
                if symbol in fresh_symbols:
                    research_attempts_by_symbol[symbol] = (
                        research_attempts_by_symbol.get(symbol, 0) + 1
                    )
                    charged_search_attempts += 1
                    researched_this_run.add(symbol)
                else:
                    structural_repairs_by_symbol[symbol] = (
                        structural_repairs_by_symbol.get(symbol, 0) + 1
                    )
            if symbol in valid:
                research_by_symbol[symbol] = valid[symbol]
                research_cache['entries'][cache_keys[symbol]] = {
                    'timestamp': datetime.now(UTC).isoformat(),
                    'research': valid[symbol],
                }
                research_cache['deferred_entries'].pop(cache_keys[symbol], None)
                print(f'Validated ETF research [{symbol}]: {len(valid[symbol]["sources"])} distinct URLs')
            else:
                error = errors.get(symbol, 'Invalid ETF research.')
                print(f'Deferred ETF research [{symbol}]: {error}')
                prior_deferred = research_cache['deferred_entries'].get(
                    cache_keys[symbol], {}
                )
                grounding_for_repair = (
                    metadata.get('source_urls', [])
                    or prior_deferred.get('grounding_source_urls', [])
                )
                basic_metadata_repair = (
                    rank_by_symbol.get(symbol, float('inf'))
                    <= top_rank_basic_metadata_repair_limit
                    and is_basic_metadata_repair_candidate(
                        error, drafts.get(symbol)
                    )
                    and bool(grounding_for_repair)
                )
                cached_error = (
                    f'BASIC_METADATA_REPAIR: {error}'
                    if basic_metadata_repair else error
                )
                research_cache['deferred_entries'][cache_keys[symbol]] = {
                    'timestamp': datetime.now(UTC).isoformat(),
                    'error': cached_error,
                    'draft': drafts.get(symbol),
                    'grounding_source_urls': grounding_for_repair,
                }
                needs_research = validation_error_requires_fresh_research(
                    error,
                    draft=drafts.get(symbol),
                    grounding_urls=grounding_for_repair,
                )
                if dedicated_basic_repair_batch:
                    # A failed evidence-only repair has already used its cheap
                    # structural attempt. Escalate directly to the ETF's final
                    # permitted fresh search rather than queueing another repair.
                    needs_research = True
                source_error = is_source_validation_error(error)
                has_repair_evidence = bool(drafts.get(symbol)) and bool(
                    grounding_for_repair
                )
                if basic_metadata_repair:
                    needs_research = False
                    print(
                        f'Queueing evidence-only completion repair [{symbol}] '
                        'before spending another search.'
                    )
                elif source_error:
                    needs_research = not has_repair_evidence
                used_attempts = (
                    research_attempts_by_symbol.get(symbol, 0)
                    if needs_research
                    else structural_repairs_by_symbol.get(symbol, 0)
                )
                attempt_limit = (
                    max_fresh_research_attempts_per_etf
                    if needs_research
                    else max_structural_repairs_per_etf
                )
                if used_attempts < attempt_limit:
                    pending_retries.append({
                        'candidate': candidate,
                        'needs_research': needs_research,
                    })
                elif (
                    basic_metadata_repair
                    and not needs_research
                    and research_attempts_by_symbol.get(symbol, 0)
                    < max_fresh_research_attempts_per_etf
                ):
                    print(
                        f'Escalating evidence completion repair [{symbol}] to its '
                        'final fresh-search retry.'
                    )
                    pending_retries.append({
                        'candidate': candidate,
                        'needs_research': True,
                    })
                elif (
                    source_error
                    and not needs_research
                    and research_attempts_by_symbol.get(symbol, 0)
                    < max_fresh_research_attempts_per_etf
                ):
                    print(
                        f'Escalating source repair [{symbol}] to its final fresh-search retry.'
                    )
                    pending_retries.append({
                        'candidate': candidate,
                        'needs_research': True,
                    })
        batch_stats = {
            'stage': stage,
            'batch_size': len(batch),
            'fresh_research_requested': len(fresh_symbols),
            'fresh_retry_requested': len(retry_fresh_symbols),
            'repair_only_requested': len(repair_only_symbols),
            'searches_exposed': len(metadata.get('search_queries', [])),
            'search_overrun': max(0, len(metadata.get('search_queries', [])) - len(fresh_symbols)),
            'validated': len(valid),
            'deferred': len(errors),
            'validated_symbols': sorted(valid),
            'deferred_symbols': sorted(errors),
            'raw_result_shapes': raw_result_shapes,
            'grounding_url_count': len(metadata.get('source_urls', [])),
            'applied_repair_patches': applied_repair_patches,
        }
        batch_research_diagnostics.append(batch_stats)
        print('Batch research summary:')
        print(f'  candidates: {batch_stats["batch_size"]}')
        print(f'  fresh research requested: {batch_stats["fresh_research_requested"]}')
        print(f'  fresh-search retries: {batch_stats["fresh_retry_requested"]}')
        print(f'  repair-only candidates: {batch_stats["repair_only_requested"]}')
        if batch_stats['search_overrun']:
            print(f'  search-query overrun: {batch_stats["search_overrun"]} above the fresh-candidate maximum')
        print(f'  validated: {batch_stats["validated"]}')
        print(f'  deferred: {batch_stats["deferred"]}')
        print_validated_research_inspection(batch, valid)
        save_json_object_atomic(research_cache_file, research_cache)
    except Exception as exc:
        print(f'ETF batch failed without validated output: {exc}')
        research_quota_exhausted = is_daily_quota_error(exc)
        response_level_failure = (
            not research_quota_exhausted and is_transient_gemini_error(exc)
        )
        batch_research_diagnostics.append({
            'stage': stage,
            'batch_size': len(batch),
            'fresh_research_requested': len(fresh_symbols),
            'fresh_retry_requested': len(retry_fresh_symbols),
            'repair_only_requested': len(repair_only_symbols),
            'validated': 0,
            'deferred': len(batch),
            'response_level_failure': response_level_failure,
            'error': str(exc),
        })
        for candidate in batch:
            symbol = str(candidate['Symbol']).upper()
            needs_research = symbol in fresh_symbols
            if response_level_failure:
                # Gemini never produced usable ETF-level research. Do not charge
                # a per-ETF fresh-research or structural-repair attempt. Requeue
                # the candidate and mix transport failures with unseen candidates
                # on the next logical batch.
                transport_failed_symbols_this_run.add(symbol)
                pending_retries.append({
                    'candidate': candidate,
                    'needs_research': needs_research,
                    'transport_failure': True,
                })
                continue
            if needs_research:
                research_attempts_by_symbol[symbol] = (
                    research_attempts_by_symbol.get(symbol, 0) + 1
                )
                used_attempts = research_attempts_by_symbol[symbol]
                attempt_limit = max_fresh_research_attempts_per_etf
            else:
                structural_repairs_by_symbol[symbol] = (
                    structural_repairs_by_symbol.get(symbol, 0) + 1
                )
                used_attempts = structural_repairs_by_symbol[symbol]
                attempt_limit = max_structural_repairs_per_etf
            if used_attempts < attempt_limit:
                pending_retries.append({
                    'candidate': candidate,
                    'needs_research': needs_research,
                })
        if response_level_failure:
            print(
                'Gemini response-level failure did not consume per-ETF research '
                'attempts; failed candidates were requeued for a mixed batch.'
            )
        if research_quota_exhausted:
            print('ETF research daily quota is exhausted; stopping research calls.')

    selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
    )
    consistency_conflicts = find_actionable_consistency_conflicts(
        candidate_records, research_by_symbol, selected
    )
    queued_repairs = enqueue_consistency_repairs(
        consistency_conflicts,
        candidate_by_symbol,
        research_by_symbol,
        research_cache,
        cache_keys,
        pending_retries,
        consistency_reviewed_symbols,
        structural_repairs_by_symbol,
        max_structural_repairs_per_etf,
    )
    if queued_repairs:
        save_json_object_atomic(research_cache_file, research_cache)
        selected = preview_portfolio(
            candidate_records,
            research_by_symbol,
            target_selected_etfs,
            max_etfs_per_sector_group,
            max_moderate_per_risk_event,
        )
    request_budget.release_summary_for_research = (
        summary_reservation_releasable_on_shortfall
        and len(selected) < target_selected_etfs
    )
    print(f'Portfolio preview after batch: {len(selected)}/{target_selected_etfs} selected.')

research_budget_exhausted = (
    len(selected) < target_selected_etfs
    and not request_budget.can_reserve('research')
)
if research_budget_exhausted:
    print(
        'ETF research stopped because the logical Gemini research-call budget '
        'is exhausted. Summary capacity was releasable during the shortfall.'
        if summary_reservation_releasable_on_shortfall
        else 'ETF research stopped because no research call remains after preserving the summary reservation.'
    )
partial_portfolio = len(selected) < target_selected_etfs
if partial_portfolio:
    print(
        'ETF portfolio shortfall: '
        f'{len(selected)} of {target_selected_etfs} requested ETFs passed '
        'validated research and portfolio rules. Publishing the validated '
        'selections without adding unresearched or ineligible ETFs.'
    )
consistency_warnings = audit_etf_consistency(candidate_records, research_by_symbol)
research_disposition = {}
for candidate in candidate_records:
    symbol = str(candidate['Symbol']).upper()
    if symbol in research_by_symbol:
        continue
    attempted = research_attempts_by_symbol.get(symbol, 0) + structural_repairs_by_symbol.get(symbol, 0)
    if attempted:
        research_disposition[symbol] = 'VALIDATION_FAILED'
    elif symbol in transport_failed_symbols_this_run:
        research_disposition[symbol] = 'API_UNAVAILABLE'
    elif symbol in sector_capacity_skipped_symbols_this_run:
        research_disposition[symbol] = 'SECTOR_CAPACITY'
    elif symbol in deferred_excess_symbols_this_run:
        research_disposition[symbol] = 'PROVISIONAL_GROUP_CAP'
    elif partial_portfolio and research_budget_exhausted:
        research_disposition[symbol] = 'RESEARCH_BUDGET_EXHAUSTED'
    else:
        research_disposition[symbol] = 'NOT_NEEDED'
selected, decision_ledger = build_decision_ledger(
    candidate_records,
    research_by_symbol,
    target_selected_etfs,
    max_etfs_per_sector_group,
    max_moderate_per_risk_event,
    research_disposition,
)
print_decision_ledger(decision_ledger)
selected = selected[:target_selected_etfs]
print('\nFinal ETF selections: ' + ', '.join(item['candidate']['Symbol'] for item in selected))
print(
    f'Research efficiency: {len(researched_this_run)} unique ETFs, '
    f'{search_attempts} requested searches, {charged_search_attempts} charged '
    f'search attempts, {max(0, charged_search_attempts - len(researched_this_run))} '
    'repeated searches.'
)
context_review = build_context_review(
    decision_ledger,
    research_by_symbol,
    selected,
    researched_this_run,
    charged_search_attempts,
    structural_repairs_by_symbol,
    batch_research_diagnostics,
)
print('\n' + context_review + '\n')

recommendations_table = build_recommendations_table(selected)
summary_html = build_fallback_summary(market_context, selected)
summary_source = 'PYTHON_FALLBACK'
summary_input_hash = None
run_diagnostics_file = config.get(
    'run_diagnostics_file', 'caches/etf_run_diagnostics.json'
)
previous_diagnostics = load_json_object(run_diagnostics_file)

if config.get('final_summary_enabled', True):
    selected_summary_input = build_summary_input(selected)
    summary_prompt_text = config['prompt_html_summary'].rstrip()
    summary_input_hash = stable_json_hash({
        'prompt': summary_prompt_text,
        'market_context': market_context,
        'selected_etfs': selected_summary_input,
    })
    previous_summary_cache = previous_diagnostics.get('summary_cache', {})
    existing_summary_html = extract_existing_summary_html('etf_index.html')
    has_matching_summary_signature = (
        isinstance(previous_summary_cache, dict)
        and bool(previous_summary_cache.get('reusable_gemini_summary'))
        and previous_summary_cache.get('input_hash') == summary_input_hash
    )
    # Migration/bootstrap path: older successful runs predate summary_cache
    # metadata. If this execution has needed zero Gemini calls so far, the
    # existing page validates against the exact current selected portfolio, and
    # the previous successful diagnostics reference the same selected symbols,
    # adopt that existing Gemini summary as the baseline instead of spending a
    # one-time editorial call solely to seed the new hash. The current run then
    # writes summary_cache metadata, so subsequent reuse uses the strict hash.
    current_selected_symbols = [
        str(item['candidate']['Symbol']).upper() for item in selected
    ]
    previous_selected_symbols = [
        str(symbol).upper()
        for symbol in (
            previous_diagnostics.get('research', {}).get('selected_symbols', [])
            if isinstance(previous_diagnostics.get('research'), dict)
            else []
        )
    ]
    can_bootstrap_existing_summary = (
        not has_matching_summary_signature
        and not previous_summary_cache
        and bool(previous_diagnostics)
        and request_budget.total_used == 0
        and previous_selected_symbols == current_selected_symbols
        and bool(existing_summary_html)
    )
    can_reuse_existing_summary = (
        bool(existing_summary_html)
        and (has_matching_summary_signature or can_bootstrap_existing_summary)
    )
    if can_reuse_existing_summary:
        try:
            summary_html = validate_summary_response(
                {'summary_html': existing_summary_html}, selected
            )
            summary_source = 'REUSED_GEMINI'
            if can_bootstrap_existing_summary:
                print(
                    'Bootstrapping ETF HTML summary cache from the existing '
                    'validated etf_index.html; portfolio is unchanged and this '
                    'run required no market-context or ETF-research Gemini calls.'
                )
            else:
                print(
                    'Reusing validated Gemini ETF HTML summary from existing '
                    'etf_index.html; summary inputs are unchanged.'
                )
        except Exception as exc:
            reason = (
                'bootstrap candidate' if can_bootstrap_existing_summary
                else 'prior input hash match'
            )
            print(
                f'Existing ETF HTML summary ({reason}) no longer validates; '
                f'regenerating it: {exc}'
            )

    if (
        summary_source != 'REUSED_GEMINI'
        and request_budget.total_used < request_budget.total
    ):
        summary_prompt = (
            summary_prompt_text
            + '\n\nMARKET_CONTEXT:\n'
            + json.dumps(market_context, ensure_ascii=False)
            + '\n\nSELECTED_ETFS:\n'
            + json.dumps(selected_summary_input, ensure_ascii=False)
        )
        summary_data = None
        for summary_model in [model_primary, model_fallback]:
            if request_budget.total_used >= request_budget.total:
                break
            try:
                summary_data, used_model, metadata = call_gemini_json(
                    client,
                    summary_model,
                    summary_prompt,
                    build_gemini_config(
                        summary_thinking_budget,
                        enable_search=False,
                        max_output_tokens=gemini_max_output_tokens,
                    ),
                    'ETF HTML summary',
                    request_budget,
                    'summary',
                    1,
                    initial_delay,
                    max_transient_delay,
                )
                models_used.append(used_model)
                call_diagnostics.append({'stage': 'HTML summary', **metadata})
                break
            except Exception as exc:
                print(f'ETF summary unavailable from {summary_model}: {exc}')
        if isinstance(summary_data, dict):
            try:
                summary_html = validate_summary_response(summary_data, selected)
                summary_source = 'GEMINI'
                print('Using Gemini-written ETF HTML summary.')
            except Exception as exc:
                print(
                    'Gemini summary was invalid; using deterministic fallback summary: '
                    f'{exc}'
                )

final_recommendations = recommendations_table + summary_html
model_used = ', '.join(dict.fromkeys(models_used)) or 'validated cache + Python'
print(f'Generated by model(s): {model_used}')

html_columns = [
    col
    for col in [
        'Symbol', 'Name', 'Category', '3M Return', '1Y Return', 'QVMScore',
    ]
    if col in df_gemini.columns
]
df_html = df_gemini[html_columns].copy()
df_html = df_html.sort_values('QVMScore', ascending=False).reset_index(drop=True)
df_html = df_html.rename(
    columns={
        'Name': 'ETF Name',
        'Category': 'Sector',
        '3M Return': '3M Return (%)',
        '1Y Return': '1Y Return (%)',
        'QVMScore': 'QVM Score',
    }
)
if {'Symbol', 'ETF Name'}.issubset(df_html.columns):
    df_html['ETF Name'] = df_html.apply(
        lambda row: (
            f'<a href="https://finance.yahoo.com/quote/{row["Symbol"]}/" '
            f'target="_blank">{row["ETF Name"]}</a>'
        ),
        axis=1,
    )
if 'Symbol' in df_html.columns:
    df_html['Symbol'] = df_html['Symbol'].apply(
        lambda symbol: (
            f'<a href="https://finance.yahoo.com/quote/{symbol}/" '
            f'target="_blank">{symbol}</a>'
        )
    )
df_html_table = df_html.to_html(
    escape=False,
    index=False,
    classes='recommendations-table',
    border=0,
)
page_reused_without_write = bool(
    summary_source == 'REUSED_GEMINI'
    and existing_page_matches_etf_content(
        'etf_index.html',
        final_recommendations,
        df_html_table,
    )
)
if page_reused_without_write:
    print(
        'ETF page content is unchanged; preserving existing etf_index.html '
        'without rewriting its timestamp or model metadata.'
    )
else:
    update_html_page(
        final_recommendations,
        df_html_table,
        'etf_page_template.html',
        'etf_index.html',
        model_used,
    )
current_classifications = {
    symbol: {
        'portfolio_group': candidate_sector_group(candidate_by_symbol[symbol]),
        'canonical_sector': candidate_by_symbol[symbol].get('CanonicalSector'),
        'style_category': candidate_by_symbol[symbol].get('StyleCategory'),
        'research_admission': research.get('research_admission'),
        'python_eligible': research.get('eligible'),
        'eligibility_reason': research.get('eligibility_reason'),
        'reversal_risk': research.get('reversal_risk'),
        'risk_basis': research.get('risk_basis'),
        'mechanism_status': research.get('mechanism_status'),
        'normalization_probability': research.get('normalization_probability'),
        'benchmark_assessment': research.get('benchmark_assessment'),
        'mandate_assessment': research.get('mandate_assessment'),
        'us_equity_weight_estimate': research.get('us_equity_weight_estimate'),
        'mandate_basis': research.get('mandate_basis'),
        'evidence_scope': research.get('evidence_scope'),
        'adverse_change_observed': research.get('adverse_change_observed'),
        'adverse_change_date': research.get('adverse_change_date'),
        'adverse_change_indicator': research.get('adverse_change_indicator'),
        'mechanism_evidence_source': research.get('mechanism_evidence_source'),
        'exposure_group': research.get('exposure_group'),
        'primary_risk_event_id': research.get('primary_risk_event_id'),
        'risk_exposure_group': research.get('risk_exposure_group'),
        'evidence_warnings': research.get('evidence_warnings') or [],
    }
    for symbol, research in research_by_symbol.items()
}
previous_classifications = previous_diagnostics.get('classifications', {})
classification_changes = []
comparison_fields = {
    'portfolio_group', 'canonical_sector', 'style_category',
    'research_admission', 'python_eligible', 'reversal_risk', 'risk_basis',
    'mechanism_status', 'normalization_probability', 'benchmark_assessment',
    'mandate_assessment', 'us_equity_weight_estimate', 'mandate_basis',
    'evidence_scope', 'adverse_change_observed', 'adverse_change_date',
    'adverse_change_indicator', 'mechanism_evidence_source',
    'exposure_group', 'primary_risk_event_id',
    'risk_exposure_group',
}
for symbol in sorted(set(previous_classifications).intersection(current_classifications)):
    changed_fields = {
        field: {'previous': previous_classifications[symbol].get(field), 'current': value}
        for field, value in current_classifications[symbol].items()
        if field in comparison_fields and previous_classifications[symbol].get(field) != value
    }
    if changed_fields:
        classification_changes.append({'symbol': symbol, 'changes': changed_fields})
if classification_changes:
    print('Classification changes versus prior ETF run:')
    for change in classification_changes:
        print(f'  {change["symbol"]}: ' + ', '.join(change['changes']))
else:
    print('Classification consistency: no changes among comparable cached ETFs.')

diagnostics = {
    'generated_at': datetime.now(UTC).isoformat(),
    'status': 'SUCCESS_PARTIAL' if partial_portfolio else 'SUCCESS',
    'selection_target': target_selected_etfs,
    'selection_count': len(selected),
    'selection_shortfall': max(0, target_selected_etfs - len(selected)),
    'models_used': list(dict.fromkeys(models_used)),
    'budget': {
        'total_used': request_budget.total_used,
        'total_limit': request_budget.total,
        'research_used': request_budget.research_used,
        'research_limit': request_budget.research_limit,
        'summary_used': request_budget.summary_used,
        'api_attempts': request_budget.api_attempts,
        'research_api_attempts': request_budget.research_api_attempts,
        'summary_api_attempts': request_budget.summary_api_attempts,
        'context_api_attempts': request_budget.context_api_attempts,
        'summary_reservation_releasable_on_shortfall': summary_reservation_releasable_on_shortfall,
    },
    'research': {
        'unique_etfs_requested': len(researched_this_run),
        'requested_searches': search_attempts,
        'charged_search_attempts': charged_search_attempts,
        'repeated_searches': max(
            0, charged_search_attempts - len(researched_this_run)
        ),
        'fresh_research_attempts_by_symbol': research_attempts_by_symbol,
        'structural_repairs_by_symbol': structural_repairs_by_symbol,
        'sector_capacity_skipped_symbols': sorted(
            sector_capacity_skipped_symbols_this_run
        ),
        'validated_symbols': sorted(research_by_symbol),
        'selected_symbols': [item['candidate']['Symbol'] for item in selected],
        'batches': batch_research_diagnostics,
        'provisional_exposure_deferred_symbols': sorted(
            deferred_excess_symbols_this_run
        ),
        'transport_failed_symbols': sorted(transport_failed_symbols_this_run),
    },
    'classifications': current_classifications,
    'classification_changes': classification_changes,
    'consistency_warnings': consistency_warnings,
    'research_disposition': research_disposition,
    'decision_ledger': decision_ledger,
    'context_review': context_review,
    'summary_cache': {
        'input_hash': summary_input_hash,
        'summary_html_hash': stable_json_hash(summary_html),
        'source': summary_source,
        'reusable_gemini_summary': summary_source in {
            'GEMINI', 'REUSED_GEMINI'
        },
    },
    'calls': call_diagnostics,
}
save_json_object_atomic(
    config.get('run_diagnostics_file', 'caches/etf_run_diagnostics.json'),
    diagnostics,
)
print(
    'Gemini request budget:\n'
    f'  logical calls: {request_budget.total_used}/{request_budget.total}\n'
    f'  ETF research logical calls: {request_budget.research_used}/{request_budget.research_limit}\n'
    f'  summary logical calls: {request_budget.summary_used}\n'
    f'  actual API attempts: {request_budget.api_attempts} '
    f'(research={request_budget.research_api_attempts}, '
    f'context={request_budget.context_api_attempts}, '
    f'summary={request_budget.summary_api_attempts})'
)
end_time = time.perf_counter()
print(f'Elapsed time: {round(end_time - start_time)} seconds\n')
