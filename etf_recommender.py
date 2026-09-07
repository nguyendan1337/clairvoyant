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
TOP_QVM_CACHE_VERSION = 9
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
    def __init__(self, total, research, reserved_summary=1):
        self.total = int(total)
        self.research_limit = int(research)
        self.reserved_summary = int(reserved_summary)
        self.total_used = 0
        self.research_used = 0
        self.summary_used = 0

    def can_reserve(self, category):
        if self.total_used >= self.total:
            return False
        if category == 'research':
            return (
                self.research_used < self.research_limit
                and self.total - self.total_used > self.reserved_summary
            )
        return True

    def reserve(self, category):
        if not self.can_reserve(category):
            if category == 'research' and self.total - self.total_used <= self.reserved_summary:
                raise RuntimeError('Only the reserved summary call remains.')
            if category == 'research':
                raise RuntimeError('ETF research call budget exhausted.')
            raise RuntimeError('Gemini call budget exhausted.')
        if category == 'research':
            self.research_used += 1
        elif category == 'summary':
            self.summary_used += 1
        self.total_used += 1


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
    allow_partial=False,
):
    last_error = None
    for attempt in range(1, int(max_attempts) + 1):
        budget.reserve(category)
        print(f'Gemini request {budget.total_used}/{budget.total}: {stage} '
              f'({model}, attempt {attempt})')
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
            delay = int(initial_delay) * (3 ** (attempt - 1))
            print(f'Retrying in {delay}s...')
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
    timestamp = parse_utc_timestamp(entry.get('timestamp')) if isinstance(entry, dict) else None
    return bool(timestamp and datetime.now(UTC) - timestamp < timedelta(hours=ttl_hours))


def stable_json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)
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


def load_top_qvm_cache():
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


def save_top_qvm_cache(df):
    try:
        cache = {
            'version': TOP_QVM_CACHE_VERSION,
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
        column for column in ('Name', 'Category', 'LegalType')
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
    selected_for_info = list(
        dict.fromkeys(
            list(priority_symbols) + sorted_symbols[:max_info_calls]
        )
    )
    print(
        'Fetching yfinance metadata for top '
        f'{len(selected_for_info)} momentum candidates...'
    )
    cache = load_json_cache(YF_CACHE_FILE, YF_CACHE_EXPIRY_DAYS)
    for symbol in tqdm(selected_for_info):
        try:
            if symbol in cache:
                info = cache[symbol]['info']
            else:
                ticker = yf.Ticker(symbol)
                raw_info = ticker.info or {}
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
                info = {
                    field: raw_info.get(field)
                    for field in info_fields
                }
                cache[symbol] = {
                    'info': info,
                    'timestamp': datetime.now(UTC).isoformat(),
                }
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
    required = ['as_of_date', 'market_summary', 'active_risk_events', 'sources']
    for field in required:
        if field not in data:
            raise ValueError(f'Market context missing {field}.')
    if len(str(data['market_summary']).strip()) < 40:
        raise ValueError('Market summary is too short.')
    if not isinstance(data['active_risk_events'], list):
        raise ValueError('active_risk_events must be a list.')
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


def normalize_etf_result(result):
    result = dict(result)
    result['symbol'] = str(result.get('symbol') or '').strip().upper()
    for field in (
        'research_status', 'benchmark_assessment', 'continuation_outlook',
        'mechanism_status', 'normalization_probability', 'probability_basis',
        'risk_materiality', 'risk_time_horizon', 'risk_basis', 'reversal_risk',
        'driver_dependence', 'mandate_assessment',
    ):
        if result.get(field) is not None:
            result[field] = str(result[field]).strip().upper()
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


def derive_reversal_risk(result):
    status = result['mechanism_status']
    probability = result['normalization_probability']
    materiality = result['risk_materiality']
    dependence = result['driver_dependence']
    if status == 'NONE':
        return 'MINIMAL' if materiality == 'LOW' else 'LOW'
    if status == 'HYPOTHETICAL':
        return 'LOW'
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


def validate_etf_result(result, candidate, minimum_sources=2, maximum_sources=5):
    result = normalize_etf_result(result)
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
    if result.get('benchmark_assessment') not in {'PASS', 'FAIL'}:
        raise ValueError(f'{symbol} has invalid benchmark_assessment.')
    if result.get('mandate_assessment') not in {'US_EQUITY', 'NOT_US_EQUITY'}:
        raise ValueError(f'{symbol} has invalid mandate_assessment.')
    if len(str(result.get('mandate_evidence') or '').strip()) < 10:
        raise ValueError(f'{symbol} missing mandate_evidence.')
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
    reported_risk = result.get('reversal_risk')
    risk = derive_reversal_risk(result)
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
    if risk in {'MODERATE', 'ELEVATED', 'SEVERE'}:
        if result['mechanism_status'] != 'OBSERVED':
            raise ValueError(f'{symbol} {risk} risk lacks an observed mechanism.')
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
            f'Excluded by Python because the evidence-derived reversal risk is {risk}.'
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
    result['sources'] = validate_sources(
        result.get('sources'), symbol, minimum=minimum_sources, maximum=maximum_sources
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
            'mandate_evidence', 'benchmark_assessment',
            'continuation_outlook', 'reversal_mechanism',
            'mechanism_status', 'normalization_probability',
            'probability_basis', 'probability_evidence', 'material_effect',
            'risk_time_horizon', 'risk_materiality', 'driver_dependence',
            'risk_basis', 'primary_risk_event_id', 'risk_exposure_group',
            'explanation',
        }
        allowed_fields = {
            field for field in repairable_fields
            if field.lower() in validation_error
        }
        if 'sources' in validation_error or 'ungrounded' in validation_error:
            allowed_fields.add('sources')
        if 'observed mechanism' in validation_error:
            allowed_fields.update({
                'mechanism_status', 'reversal_mechanism',
                'normalization_probability', 'probability_evidence',
                'risk_materiality', 'driver_dependence', 'material_effect',
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


def validation_error_requires_fresh_research(message):
    """Separate evidence failures from JSON/classification repair failures."""
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
        for field in ('Name', 'Category')
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
    return str(candidate.get('Category') or 'Other').strip() or 'Other'


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
            f'sector={candidate_sector_group(candidate)}, '
            f'exposure={research.get("exposure_group")}; '
            f'{research.get("eligibility_reason")}'
        )


def print_decision_ledger(ledger, heading='ETF portfolio-rule inspection:'):
    print(heading)
    for row in ledger:
        print(
            f'  rank={row["qvm_rank"]} {row["symbol"]}: {row["status"]}; '
            f'sector={row.get("sector_group") or "N/A"}; '
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
        'Sector Selected After', 'Total Selected After', 'Explanation',
    ]
    rows = [
        [
            decision['qvm_rank'],
            decision['symbol'],
            decision['sector_group'],
            decision['status'],
            decision['sector_selected_after'],
            decision['total_selected_after'],
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


def build_decision_ledger(candidates, research_by_symbol, target, max_per_sector, max_moderate_event):
    selected, ledger, sector_counts, event_counts = [], [], {}, {}
    for rank, candidate in enumerate(candidates, start=1):
        if len(selected) >= target:
            break
        symbol = str(candidate['Symbol']).upper()
        research = research_by_symbol.get(symbol)
        status, reason = None, None
        sector = candidate_sector_group(candidate)
        sector_key = sector.casefold()
        if not research:
            status = 'NOT SELECTED — RESEARCH UNAVAILABLE'
            reason = 'No validated current ETF research was available.'
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
max_transient_api_attempts = config.get('max_transient_api_attempts', 2)
max_fresh_research_attempts_per_etf = config.get(
    'max_fresh_research_attempts_per_etf', 2
)
max_structural_repairs_per_etf = config.get(
    'max_structural_repairs_per_etf', 2
)
min_structural_repair_batch_size = config.get(
    'min_structural_repair_batch_size', 3
)
research_cache_file = config.get(
    'etf_research_cache_file', 'caches/gemini_etf_research_cache.json'
)
market_context_cache_file = config.get(
    'market_context_cache_file', 'caches/gemini_etf_market_context_cache.json'
)
research_cache_hours = config.get('gemini_research_cache_hours', 12)
market_context_cache_hours = config.get('market_context_cache_hours', 12)
cache_version = config.get('cache_version', 1)
benchmark_etfs = [
    str(symbol).strip().upper()
    for symbol in config['benchmark_etfs']
]
if not benchmark_etfs:
    raise ValueError('benchmark_etfs must contain at least one symbol.')
max_info_calls = config.get('max_info_calls', 500)
top_n = config.get('qvm_top_n', 50)
qvm_weights = config.get('qvm_weights', {'Quality': 0.25, 'Value': 0.15, 'Momentum': 0.6})
min_quality = config.get('min_quality', 35)
top_etfs = load_top_qvm_cache()
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
    strict_candidate_mask = (
        superiority_mask
        & (scored_etfs['QualityScore'] >= min_quality)
    )
    strict_candidates = scored_etfs[strict_candidate_mask].copy()
    strict_candidates['ResearchAdmission'] = 'STRICT'
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
    research_pool_limit = min(
        top_n,
        config.get('research_candidate_pool_limit', top_n),
    )
    backfill_needed = max(
        0,
        research_pool_limit - len(strict_candidates),
    )
    strict_symbols = set(strict_candidates['Symbol'])
    backfill_candidates = scored_etfs[
        competitive_mask & ~scored_etfs['Symbol'].isin(strict_symbols)
    ].head(backfill_needed)
    backfill_candidates = backfill_candidates.copy()
    backfill_candidates['ResearchAdmission'] = 'BACKFILL'
    candidate_etfs = (
        pd.concat([strict_candidates, backfill_candidates], ignore_index=True)
        .drop_duplicates(subset='Symbol', keep='first')
        .sort_values(['QVMScore', 'MomentumScore'], ascending=[False, False])
        .head(research_pool_limit)
    )
    print(
        f'ETF research candidates: {len(candidate_etfs)} '
        f'({len(strict_candidates)} strict benchmark-superior, '
        f'{len(backfill_candidates)} benchmark-competitive backfill; '
        f'maximum requested: {top_n}).'
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
    save_top_qvm_cache(top_etfs)

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
top_etfs = apply_structured_equity_filter(
    top_etfs,
    benchmark_etfs,
    config.get('min_equity_asset_weight', 0.80),
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
print(top_etfs[display_columns].head(50).to_string(index=False))
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
    raise RuntimeError(
        f'Only {len(candidate_records)} benchmark-superior ETF candidates are available; '
        f'{target_selected_etfs} are required.'
    )

client = initialize_gemini_client()
request_budget = GeminiRequestBudget(
    max_gemini_calls_per_run,
    max_etf_research_calls_per_run,
    reserved_summary_calls,
)
call_diagnostics = []
batch_research_diagnostics = []
models_used = []

# Research the shared market backdrop once and cache it separately from ETFs.
market_prompt = config['prompt_market_context'].rstrip()
market_prompt_hash = stable_json_hash({'version': cache_version, 'prompt': market_prompt})
market_cache = load_json_object(market_context_cache_file)
market_context = None
if (
    market_cache.get('version') == cache_version
    and market_cache.get('prompt_hash') == market_prompt_hash
    and cache_entry_is_fresh(market_cache, market_context_cache_hours)
):
    try:
        market_context = validate_market_context(market_cache.get('data'))
        age = datetime.now(UTC) - parse_utc_timestamp(market_cache['timestamp'])
        print(f'Using validated ETF market context cache ({age.total_seconds() / 3600:.1f}h old).')
    except Exception as exc:
        print(f'Ignoring invalid ETF market context cache: {exc}')
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
    )
    if not metadata.get('search_queries') and metadata.get('tool_tokens', 0) <= 0:
        raise RuntimeError('Gemini returned ungrounded ETF market context.')
    market_context = validate_market_context(data)
    models_used.append(used_model)
    call_diagnostics.append({'stage': 'market context', **metadata})
    save_json_object_atomic(
        market_context_cache_file,
        {
            'version': cache_version,
            'prompt_hash': market_prompt_hash,
            'timestamp': datetime.now(UTC).isoformat(),
            'data': market_context,
        },
    )

research_prompt_hash = stable_json_hash(
    {'version': cache_version, 'prompt': config['prompt_etf_batch']}
)
research_cache = load_json_object(
    research_cache_file,
    {'version': cache_version, 'entries': {}, 'deferred_entries': {}},
)
if research_cache.get('version') != cache_version:
    research_cache = {'version': cache_version, 'entries': {}, 'deferred_entries': {}}
research_cache.setdefault('entries', {})
research_cache.setdefault('deferred_entries', {})

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
            'SectorWeightings', 'TopHoldings',
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

selected = preview_portfolio(
    candidate_records,
    research_by_symbol,
    target_selected_etfs,
    max_etfs_per_sector_group,
    max_moderate_per_risk_event,
)
if len(selected) >= target_selected_etfs:
    print('Validated ETF research cache already supports the full portfolio.')

cursor = 0
pending_retries = []
deferred_excess_candidates = []
deferred_excess_symbols_this_run = set()
sector_capacity_skipped_symbols_this_run = set()
research_attempts_by_symbol = {}
structural_repairs_by_symbol = {}
researched_this_run = set()
search_attempts = 0
charged_search_attempts = 0
research_quota_exhausted = False
rank_by_symbol = {
    str(candidate['Symbol']).upper(): rank
    for rank, candidate in enumerate(candidate_records, start=1)
}
while (
    len(selected) < target_selected_etfs
    and request_budget.can_reserve('research')
    and not research_quota_exhausted
):
    batch = []
    retry_symbols = set()
    retry_fresh_symbols = set()
    provisional_counts = {}
    while pending_retries and len(batch) < gemini_batch_size:
        retry = pending_retries.pop(0)
        candidate = retry['candidate']
        symbol = str(candidate['Symbol']).upper()
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
        provisional = provisional_exposure_group(candidate)
        provisional_counts[provisional] = provisional_counts.get(provisional, 0) + 1
    while cursor < len(candidate_records) and len(batch) < gemini_batch_size:
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
        if provisional_counts.get(provisional, 0) >= max_candidates_per_provisional_group:
            deferred_excess_candidates.append(candidate)
            deferred_excess_symbols_this_run.add(symbol)
            continue
        batch.append(candidate)
        provisional_counts[provisional] = provisional_counts.get(provisional, 0) + 1
    if cursor >= len(candidate_records) and len(batch) < gemini_batch_size:
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
    if cursor >= len(candidate_records) and len(batch) < gemini_batch_size:
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
    if not batch and deferred_excess_candidates:
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
        not fresh_symbols
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
            retry_payload.append({
                'symbol': symbol,
                'validation_error': deferred.get('error'),
                'prior_draft': deferred.get('draft'),
            })
            prior_grounding_urls.extend(deferred.get('grounding_source_urls', []))
    prior_grounding_urls = list(dict.fromkeys(prior_grounding_urls))[
        :config.get('max_grounding_urls_for_repair', 60)
    ]
    ranks = [rank_by_symbol[str(candidate['Symbol']).upper()] for candidate in batch]
    stage = f'ETF batch ranks {min(ranks)}-{max(ranks)}'
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
                    print(
                        f'  {symbol}: compact patch merged; fields='
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
                'mechanism_status': (
                    draft.get('mechanism_status') if isinstance(draft, dict) else None
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
                        f'mechanism={shape["mechanism_status"]}, '
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
                research_cache['deferred_entries'][cache_keys[symbol]] = {
                    'timestamp': datetime.now(UTC).isoformat(),
                    'error': error,
                    'draft': drafts.get(symbol),
                    'grounding_source_urls': grounding_for_repair,
                }
                needs_research = validation_error_requires_fresh_research(error)
                source_error = is_source_validation_error(error)
                has_repair_evidence = bool(drafts.get(symbol)) and bool(
                    grounding_for_repair
                )
                if source_error:
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
        print(f'  validated: {batch_stats["validated"]}')
        print(f'  deferred: {batch_stats["deferred"]}')
        print_validated_research_inspection(batch, valid)
        save_json_object_atomic(research_cache_file, research_cache)
    except Exception as exc:
        print(f'ETF batch failed without validated output: {exc}')
        research_quota_exhausted = is_daily_quota_error(exc)
        batch_research_diagnostics.append({
            'stage': stage,
            'batch_size': len(batch),
            'fresh_research_requested': len(fresh_symbols),
            'fresh_retry_requested': len(retry_fresh_symbols),
            'repair_only_requested': len(repair_only_symbols),
            'validated': 0,
            'deferred': len(batch),
            'error': str(exc),
        })
        for candidate in batch:
            symbol = str(candidate['Symbol']).upper()
            needs_research = symbol in fresh_symbols
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
        if research_quota_exhausted:
            print('ETF research daily quota is exhausted; stopping research calls.')

    selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
    )
    print(f'Portfolio preview after batch: {len(selected)}/{target_selected_etfs} selected.')

if len(selected) < target_selected_etfs and not request_budget.can_reserve('research'):
    print(
        'ETF research stopped before constructing another batch because no '
        'research call remains after preserving the summary reservation.'
    )
if len(selected) < target_selected_etfs:
    failure_selected, failure_ledger = build_decision_ledger(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
    )
    print_decision_ledger(
        failure_ledger,
        heading='Final ETF failure inspection:',
    )
    failure_context_review = build_context_review(
        failure_ledger,
        research_by_symbol,
        failure_selected,
        researched_this_run,
        charged_search_attempts,
        structural_repairs_by_symbol,
        batch_research_diagnostics,
    )
    print('\n' + failure_context_review + '\n')
    failure_classifications = {
        symbol: {
            'sector_group': candidate_sector_group(candidate_by_symbol[symbol]),
            'research_admission': research.get('research_admission'),
            'model_eligible': research.get('model_eligible'),
            'python_eligible': research.get('eligible'),
            'eligibility_reason': research.get('eligibility_reason'),
            'benchmark_assessment': research.get('benchmark_assessment'),
            'mandate_assessment': research.get('mandate_assessment'),
            'reversal_risk': research.get('reversal_risk'),
            'exposure_group': research.get('exposure_group'),
            'evidence_warnings': research.get('evidence_warnings') or [],
        }
        for symbol, research in research_by_symbol.items()
    }
    save_json_object_atomic(
        config.get('run_diagnostics_file', 'caches/etf_run_diagnostics.json'),
        {
            'generated_at': datetime.now(UTC).isoformat(),
            'status': 'FAILED_INSUFFICIENT_SELECTIONS',
            'failure_reason': (
                f'Only {len(failure_selected)} of {target_selected_etfs} ETFs '
                'passed validated research and portfolio rules.'
            ),
            'budget': {
                'total_used': request_budget.total_used,
                'total_limit': request_budget.total,
                'research_used': request_budget.research_used,
                'research_limit': request_budget.research_limit,
            },
            'fresh_research_attempts_by_symbol': research_attempts_by_symbol,
            'structural_repairs_by_symbol': structural_repairs_by_symbol,
            'sector_capacity_skipped_symbols': sorted(
                sector_capacity_skipped_symbols_this_run
            ),
            'batch_research_diagnostics': batch_research_diagnostics,
            'classifications': failure_classifications,
            'decision_ledger': failure_ledger,
            'context_review': failure_context_review,
            'calls': call_diagnostics,
        },
    )
    raise RuntimeError(
        f'Only {len(selected)} ETFs passed validated research and portfolio rules. '
        'Publishing was stopped rather than filling with unresearched ETFs.'
    )
selected, decision_ledger = build_decision_ledger(
    candidate_records,
    research_by_symbol,
    target_selected_etfs,
    max_etfs_per_sector_group,
    max_moderate_per_risk_event,
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
if config.get('final_summary_enabled', True) and request_budget.total_used < request_budget.total:
    selected_summary_input = build_summary_input(selected)
    summary_prompt = (
        config['prompt_html_summary'].rstrip()
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
            )
            models_used.append(used_model)
            call_diagnostics.append({'stage': 'HTML summary', **metadata})
            break
        except Exception as exc:
            print(f'ETF summary unavailable from {summary_model}: {exc}')
    if isinstance(summary_data, dict):
        try:
            summary_html = validate_summary_response(summary_data, selected)
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
update_html_page(
    final_recommendations,
    df_html_table,
    'etf_page_template.html',
    'etf_index.html',
    model_used,
)
previous_diagnostics = load_json_object(
    config.get('run_diagnostics_file', 'caches/etf_run_diagnostics.json')
)
current_classifications = {
    symbol: {
        'sector_group': candidate_sector_group(candidate_by_symbol[symbol]),
        'research_admission': research.get('research_admission'),
        'model_eligible': research.get('model_eligible'),
        'python_eligible': research.get('eligible'),
        'eligibility_reason': research.get('eligibility_reason'),
        'reversal_risk': research.get('reversal_risk'),
        'risk_basis': research.get('risk_basis'),
        'mechanism_status': research.get('mechanism_status'),
        'normalization_probability': research.get('normalization_probability'),
        'benchmark_assessment': research.get('benchmark_assessment'),
        'mandate_assessment': research.get('mandate_assessment'),
        'exposure_group': research.get('exposure_group'),
        'primary_risk_event_id': research.get('primary_risk_event_id'),
        'risk_exposure_group': research.get('risk_exposure_group'),
        'evidence_warnings': research.get('evidence_warnings') or [],
    }
    for symbol, research in research_by_symbol.items()
}
previous_classifications = previous_diagnostics.get('classifications', {})
classification_changes = []
for symbol in sorted(set(previous_classifications).intersection(current_classifications)):
    changed_fields = {
        field: {'previous': previous_classifications[symbol].get(field), 'current': value}
        for field, value in current_classifications[symbol].items()
        if previous_classifications[symbol].get(field) != value
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
    'status': 'SUCCESS',
    'models_used': list(dict.fromkeys(models_used)),
    'budget': {
        'total_used': request_budget.total_used,
        'total_limit': request_budget.total,
        'research_used': request_budget.research_used,
        'research_limit': request_budget.research_limit,
        'summary_used': request_budget.summary_used,
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
    },
    'classifications': current_classifications,
    'classification_changes': classification_changes,
    'decision_ledger': decision_ledger,
    'context_review': context_review,
    'calls': call_diagnostics,
}
save_json_object_atomic(
    config.get('run_diagnostics_file', 'caches/etf_run_diagnostics.json'),
    diagnostics,
)
print(
    'Gemini request budget:\n'
    f'  total calls: {request_budget.total_used}/{request_budget.total}\n'
    f'  ETF research calls: {request_budget.research_used}/{request_budget.research_limit}\n'
    f'  summary calls: {request_budget.summary_used}'
)
end_time = time.perf_counter()
print(f'Elapsed time: {round(end_time - start_time)} seconds\n')
