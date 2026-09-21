"""Discover, enrich, score, and publish high-momentum ETF recommendations.

Yahoo universe discovery and market data use yfinance. Gemini adds current
context to the quantitative results and the final output is rendered to HTML.
"""

import os
import sys
import re
import time
import random
import json
import hashlib
import html
import yaml
import signal
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
FUND_CACHE_EXPIRY_DAYS = 3
TOP_QVM_CACHE_FILE = 'caches/top_qvm_etfs_cache.pkl'
TOP_QVM_CACHE_EXPIRY_HOURS = 6
TOP_QVM_CACHE_VERSION = 14
SCRIPT_DIR = Path(__file__).resolve().parent
TOTAL_RUNTIME_TIMEOUT_SECONDS = 60 * 60
GEMINI_REQUEST_TIMEOUT_MS = 10 * 60 * 1000

# Local semantic matching for dynamic market-event IDs. FastEmbed is loaded
# lazily only when Gemini returns a noncanonical ID. Any import, model-download,
# or runtime failure falls back to a deterministic lexical matcher, so this
# optional reconciliation layer cannot take down the ETF run.
FASTEMBED_MODEL_NAME = os.getenv(
    'CLAIRVOYANT_FASTEMBED_MODEL', 'BAAI/bge-small-en-v1.5'
)
FASTEMBED_CACHE_DIR = Path('caches') / 'fastembed'
FASTEMBED_IDENTITY_MIN_SIMILARITY = 0.64
FASTEMBED_TRANSMISSION_MIN_SIMILARITY = 0.60
FASTEMBED_COMBINED_MIN_SIMILARITY = 0.64
FASTEMBED_UNIQUENESS_MARGIN = 0.06
_fastembed_model = None
_fastembed_unavailable_reason = None
_risk_event_embedding_cache = {}


class TeeStream:
    """Keep console output while recording a complete upload-friendly run log."""
    def __init__(self, original, log):
        self.original = original
        self.log = log
    def write(self, message):
        self.original.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.original.flush()
        self.log.flush()
    def __getattr__(self, name):
        return getattr(self.original, name)

RUN_REPORTS_DIR = Path('run_reports')
RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ETF_RUN_LOG_FILE = RUN_REPORTS_DIR / 'etf_run.log'
_etf_run_log = open(ETF_RUN_LOG_FILE, 'w', encoding='utf-8', buffering=1)
sys.stdout = TeeStream(sys.stdout, _etf_run_log)
sys.stderr = TeeStream(sys.stderr, _etf_run_log)
classification_call_diagnostics = []
classification_logical_calls_used = 0
classification_api_attempts_used = 0
classification_quota_exhausted = False
summary_35_api_attempts_used = 0
classification_validation_rounds = {}
classification_validation_diagnostics = []
classification_scheduling_diagnostics = []
runtime_reconciliation_diagnostics = []
cache_diagnostics = {
    'yfinance_metadata': {'status': 'not_run'},
    'fund_data': {'status': 'not_run'},
}
api_attempt_diagnostics = []
stage_diagnostics = []
active_stage = 'startup'
active_stage_started = time.perf_counter()


class TotalRuntimeTimeout(BaseException):
    """Stop the process when the recommender exceeds its total runtime."""


def handle_total_runtime_timeout(signum, frame):
    raise TotalRuntimeTimeout(
        f'ETF recommender exceeded its '
        f'{TOTAL_RUNTIME_TIMEOUT_SECONDS // 60}-minute total runtime limit.'
    )


def initialize_gemini_client():
    api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent / '.env'
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        raise ValueError('GEMINI_KEY not found in environment or .env')

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
    )


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
    def __init__(
        self, total, research, reserved_summary=1,
        release_summary_for_research=False, max_api_attempts=None,
    ):
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
        self.max_api_attempts = int(
            total if max_api_attempts is None else max_api_attempts
        )

    def can_reserve(self, category):
        if self.total_used >= self.total or self.api_attempts >= self.max_api_attempts:
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
        if self.api_attempts >= self.max_api_attempts:
            raise RuntimeError(
                f'Gemini API-attempt ceiling exhausted '
                f'({self.api_attempts}/{self.max_api_attempts}).'
            )
        self.api_attempts += 1
        if category == 'research':
            self.research_api_attempts += 1
        elif category == 'summary':
            self.summary_api_attempts += 1
        else:
            self.context_api_attempts += 1


class GeminiApiAttemptBudgetExhausted(RuntimeError):
    """Raised before an API call that would exceed a per-model ceiling."""


class GeminiApiAttemptBudget:
    """Enforce actual API-attempt limits by model family and workload."""

    def __init__(self, family_limits, category_limits):
        self.family_limits = {
            str(family): int(limit)
            for family, limit in family_limits.items()
        }
        self.category_limits = {
            (str(family), str(category)): int(limit)
            for (family, category), limit in category_limits.items()
        }
        self.family_used = {family: 0 for family in self.family_limits}
        self.category_used = {key: 0 for key in self.category_limits}

    @staticmethod
    def model_family(model_name):
        normalized = str(model_name or '').lower()
        if 'gemini-3.5' in normalized:
            return '3.5'
        if 'gemini-2.5' in normalized:
            return '2.5'
        return normalized or 'unknown'

    def reserve(self, model_name, category, stage):
        family = self.model_family(model_name)
        family_limit = self.family_limits.get(family)
        category_key = (family, str(category))
        category_limit = self.category_limits.get(category_key)
        family_used = self.family_used.get(family, 0)
        category_used = self.category_used.get(category_key, 0)
        if family_limit is not None and family_used >= family_limit:
            raise GeminiApiAttemptBudgetExhausted(
                f'Gemini {family} API-attempt budget of {family_limit} was '
                f'exhausted before {stage}.'
            )
        if category_limit is not None and category_used >= category_limit:
            raise GeminiApiAttemptBudgetExhausted(
                f'Gemini {family} {category} API-attempt budget of '
                f'{category_limit} was exhausted before {stage}.'
            )
        self.family_used[family] = family_used + 1
        if category_limit is not None:
            self.category_used[category_key] = category_used + 1
        print(
            f'Gemini {family} API budget: '
            f'{self.family_used[family]}/{family_limit or "unlimited"} total; '
            f'{self.category_used.get(category_key, 0)}/'
            f'{category_limit or "unlimited"} {category}.'
        )

    def remaining(self, model_name, category):
        family = self.model_family(model_name)
        family_limit = self.family_limits.get(family)
        family_remaining = (
            float('inf') if family_limit is None
            else max(0, family_limit - self.family_used.get(family, 0))
        )
        category_key = (family, str(category))
        category_limit = self.category_limits.get(category_key)
        category_remaining = (
            float('inf') if category_limit is None
            else max(
                0,
                category_limit - self.category_used.get(category_key, 0),
            )
        )
        return min(family_remaining, category_remaining)

    def snapshot(self):
        return {
            'family_used': dict(self.family_used),
            'family_limits': dict(self.family_limits),
            'category_used': {
                f'{family}:{category}': used
                for (family, category), used in self.category_used.items()
            },
            'category_limits': {
                f'{family}:{category}': limit
                for (family, category), limit in self.category_limits.items()
            },
        }


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
    api_category = 'market' if category == 'context' else category
    if api_attempt_budget.remaining(model, api_category) <= 0:
        raise GeminiApiAttemptBudgetExhausted(
            f'No {model} {api_category} API attempts remain before {stage}.'
        )
    budget.reserve(category)
    logical_request = budget.total_used
    for attempt in range(1, int(max_attempts) + 1):
        api_attempt_budget.reserve(model, api_category, stage)
        budget.record_api_attempt(category)
        attempt_started = time.perf_counter()
        attempt_diag = {
            'stage': stage, 'model': model, 'category': category,
            'logical_call': logical_request, 'api_attempt': budget.api_attempts,
            'success': False,
        }
        api_attempt_diagnostics.append(attempt_diag)
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
            attempt_diag['success'] = True
            return data, model, metadata
        except Exception as exc:
            last_error = exc
            attempt_diag['error'] = str(exc)
            print(f'Gemini {stage} attempt {attempt}/{max_attempts} failed: {exc}')
            if (
                is_daily_quota_error(exc)
                or budget.api_attempts >= budget.max_api_attempts
                or api_attempt_budget.remaining(model, api_category) <= 0
                or attempt >= int(max_attempts)
                or not is_transient_gemini_error(exc)
            ):
                break
            base_delay = min(float(max_delay), float(initial_delay) * (2 ** (attempt - 1)))
            delay = max(1.0, min(float(max_delay), base_delay * random.uniform(0.75, 1.5)))
            print(f'Retrying transient Gemini failure in {delay:.1f}s...')
            time.sleep(delay)
        finally:
            attempt_diag['elapsed_seconds'] = round(time.perf_counter() - attempt_started, 3)
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
        return True
    except Exception as exc:
        print(f'Could not save {path}: {exc}')
        return False


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
    return bool(timestamp and timedelta(0) <= datetime.now(UTC) - timestamp < timedelta(hours=ttl_hours))


def cache_age_hours(value):
    timestamp = parse_utc_timestamp(value)
    return ((datetime.now(UTC) - timestamp).total_seconds() / 3600
            if timestamp else None)


def stable_json_hash(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


RUN_ID = (
    f"{os.environ['GITHUB_RUN_ID']}-{os.getenv('GITHUB_RUN_ATTEMPT', '1')}"
    if os.getenv('GITHUB_RUN_ID') else datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')
)
RUN_PROVENANCE = {
    'run_id': RUN_ID, 'github_sha': os.getenv('GITHUB_SHA'),
    'github_run_attempt': os.getenv('GITHUB_RUN_ATTEMPT'),
}


def write_run_checkpoint(status='RUNNING', error=None):
    """Best-effort diagnostics must never replace an application exception."""
    try:
        state = globals()
        budget = state.get('request_budget')
        snapshot = {
            'schema_version': 2, 'run_id': RUN_ID,
            'updated_at': datetime.now(UTC).isoformat(), 'status': status,
            'active_stage': active_stage,
            'elapsed_seconds': round(time.perf_counter() - state.get('start_time', active_stage_started), 3),
            'active_stage_seconds': round(time.perf_counter() - active_stage_started, 3),
            'provenance': RUN_PROVENANCE, 'stages': stage_diagnostics,
            'cache': cache_diagnostics,
            'budget_25': dict(vars(budget)) if budget else {},
            'per_model_api_attempts': (
                state['api_attempt_budget'].snapshot()
                if state.get('api_attempt_budget') else {}
            ),
            'classification_api_attempts': classification_api_attempts_used,
            'classification_validation': classification_validation_diagnostics,
            'classification_calls': classification_call_diagnostics,
            'api_attempts': api_attempt_diagnostics,
            'batches': state.get('batch_research_diagnostics', []),
            'validated_symbols': sorted(state.get('research_by_symbol', {})),
            'selected_symbols': [str(x['candidate']['Symbol']) for x in state.get('selected', [])],
            'error': error,
        }
        save_json_object_atomic(RUN_REPORTS_DIR / 'etf_run_checkpoint.json', snapshot)
        if status == 'FAILED':
            save_json_object_atomic(RUN_REPORTS_DIR / 'etf_run_failure.json', snapshot)
    except Exception as exc:
        print(f'Warning: ETF checkpoint unavailable: {exc}')


def set_run_stage(stage):
    global active_stage, active_stage_started
    now = time.perf_counter()
    stage_diagnostics.append({
        'stage': active_stage, 'elapsed_seconds': round(now - active_stage_started, 3),
    })
    active_stage, active_stage_started = stage, now
    print(f'ETF STAGE {stage} run_id={RUN_ID}')
    write_run_checkpoint()


def build_coverage_diagnostics(candidates, research, requests, attempts, selected_symbols, actionable_symbols, attempt_limit):
    """Separate work never attempted from work attempted but still unusable."""
    selected_symbols, actionable_symbols = set(selected_symbols), set(actionable_symbols)
    counts, lifecycle = {}, {}
    for candidate in candidates:
        symbol = str(candidate['Symbol']).upper()
        result = research.get(symbol, {})
        if symbol in selected_symbols:
            state = 'SELECTED'
        elif result.get('judgment_model'):
            state = 'JUDGED_NOT_SELECTED'
        elif result:
            state = 'VALIDATED_UNJUDGED'
        elif attempts.get(symbol, 0) >= attempt_limit:
            state = 'VALIDATION_EXHAUSTED'
        elif requests.get(symbol, 0):
            state = 'REQUESTED_UNVALIDATED'
        else:
            state = 'NEVER_REQUESTED'
        counts[state] = counts.get(state, 0) + 1
        lifecycle[symbol] = {
            'state': state, 'requests': requests.get(symbol, 0),
            'charged_research_attempts': attempts.get(symbol, 0),
            'actionable_for_research': symbol in actionable_symbols,
            'judgment_validation_rounds': classification_validation_rounds.get(symbol, 0),
            'judgment_error': result.get('judgment_validation_error'),
        }
    return {
        'counts': counts, 'candidates': lifecycle,
        'actionable_never_requested': [s for s, v in lifecycle.items()
                                       if v['actionable_for_research'] and v['state'] == 'NEVER_REQUESTED'],
        'actionable_retryable': [s for s, v in lifecycle.items()
                                if v['actionable_for_research'] and v['state'] == 'REQUESTED_UNVALIDATED'],
        'validated_unjudged': [s for s, v in lifecycle.items() if v['state'] == 'VALIDATED_UNJUDGED'],
    }


_original_exception_hook = sys.excepthook


def report_uncaught_failure(exc_type, exc_value, traceback):
    write_run_checkpoint('FAILED', f'{exc_type.__name__}: {exc_value}')
    _original_exception_hook(exc_type, exc_value, traceback)


sys.excepthook = report_uncaught_failure


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
    # Leave the prior complete page untouched if rendering/writing fails.
    temporary = f'{display_page}.tmp'
    with open(temporary, 'w', encoding='utf-8') as f:
        f.write(html_output)
    os.replace(temporary, display_page)


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
    if not isinstance(cache, dict):
        return fresh_cache
    now = datetime.now(UTC)
    for key, entry in cache.items():
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if timedelta(0) <= now - timestamp < timedelta(days=expiry_days):
                fresh_cache[key] = entry
        except Exception:
            continue
    return fresh_cache


def save_json_cache(path, cache):
    return save_json_object_atomic(path, cache)


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
    cache = load_json_cache(cache_file, max(YF_CACHE_EXPIRY_DAYS, FUND_CACHE_EXPIRY_DAYS))
    fund_data_by_symbol = {}
    fund_cache_stats = {'status': 'completed', 'hits': 0, 'live_requests': 0, 'incomplete_symbols': []}
    print(f'Fetching structured fund data for {len(df)} ETF research candidates...')
    for symbol in tqdm(df['Symbol'].astype(str).str.upper().tolist()):
        entry = cache.get(symbol, {})
        fund_data = entry.get('fund_data')
        fund_fresh = cache_entry_is_fresh(
            {'timestamp': entry.get('fund_timestamp')}, FUND_CACHE_EXPIRY_DAYS * 24
        )
        if not fund_data or not fund_fresh:
            fund_cache_stats['live_requests'] += 1
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
            if any(fund_data.values()):
                entry['fund_timestamp'] = entry['timestamp']
            else:
                fund_cache_stats['incomplete_symbols'].append(symbol)
            cache[symbol] = entry
            time.sleep(delay + random.uniform(0, 0.15))
        else:
            fund_cache_stats['hits'] += 1
        fund_data_by_symbol[symbol] = fund_data or {}
    fund_cache_stats['save_succeeded'] = save_json_cache(cache_file, cache)
    cache_diagnostics['fund_data'] = fund_cache_stats
    print('ETF FUND CACHE ' + json.dumps(fund_cache_stats))
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
    """Compute cumulative returns plus stock-style price-path quality metrics."""
    if close is None or len(close) < 20:
        return {}
    latest = safe_float(close.iloc[-1])
    daily_returns = close.pct_change().dropna()
    downside = daily_returns[daily_returns < 0]
    running_peak = close.cummax()
    drawdowns = close / running_peak - 1
    trend_window = close.tail(min(126, len(close)))
    trend_r2 = None
    annualized_trend = None
    if len(trend_window) >= 20:
        logs = np.log(trend_window.to_numpy(dtype=float))
        if np.isfinite(logs).all():
            x = np.arange(len(logs), dtype=float)
            slope, intercept = np.polyfit(x, logs, 1)
            fitted = slope * x + intercept
            rss = float(np.square(logs - fitted).sum())
            tss = float(np.square(logs - logs.mean()).sum())
            trend_r2 = 1.0 - rss / tss if tss > 0 else 0.0
            annualized_trend = float((np.exp(slope * 252) - 1) * 100)
    avg50 = safe_float(close.tail(min(50, len(close))).mean())
    avg200 = safe_float(close.tail(min(200, len(close))).mean())
    high52 = safe_float(close.max())
    ret1m = return_from_history(close, 21)
    prior_2m_monthly = None
    if len(close) > 63:
        prior_total = float(close.iloc[-21] / close.iloc[-63])
        prior_2m_monthly = (prior_total ** 0.5 - 1) * 100
    acceleration = float(ret1m - prior_2m_monthly) if ret1m is not None and prior_2m_monthly is not None else None
    rolling5 = close.pct_change(5).dropna()
    return {
        'Price': latest,
        '1M Return': ret1m,
        '3M Return': return_from_history(close, 63),
        '6M Return': return_from_history(close, 126),
        '9M Return': return_from_history(close, 189),
        '1Y Return': return_from_history(close, 252),
        'Volatility 1Y': safe_float(daily_returns.std(ddof=0) * np.sqrt(252) * 100),
        'DownsideVolatility': safe_float(downside.std(ddof=0) * np.sqrt(252) * 100) if len(downside) >= 10 else None,
        'MaxDrawdown': safe_float(drawdowns.min() * 100),
        'PositiveDayPct': safe_float((daily_returns > 0).mean() * 100),
        'TrendR2': safe_float(trend_r2),
        'AnnualizedTrend': safe_float(annualized_trend),
        '50D Average': avg50,
        '200D Average': avg200,
        'Distance50DMA': safe_float((latest / avg50 - 1) * 100) if latest and avg50 else None,
        'Distance200DMA': safe_float((latest / avg200 - 1) * 100) if latest and avg200 else None,
        'Distance52WHigh': safe_float((latest / high52 - 1) * 100) if latest and high52 else None,
        'Largest1DayMove': safe_float(daily_returns.abs().max() * 100) if not daily_returns.empty else None,
        'Largest5DayMove': safe_float(rolling5.abs().max() * 100) if not rolling5.empty else None,
        'MomentumAcceleration': safe_float(acceleration),
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
        'Resolving yfinance metadata (cache first) for '
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
    metadata_stats = {
        'status': 'completed',
        'eligible': len(selected_for_info), 'hits': 0, 'live_requests': 0,
        'failed_symbols': [], 'incomplete_symbols': [],
        'ttl_hours': YF_CACHE_EXPIRY_DAYS * 24,
    }
    for symbol in tqdm(selected_for_info, desc='ETF metadata (cache + live)'):
        try:
            entry = cache.get(symbol) if isinstance(cache.get(symbol), dict) else {}
            info = entry.get('info') if isinstance(entry.get('info'), dict) else {}
            info_is_usable = (
                any(info.get(field) is not None for field in identity_fields)
                and any(info.get(field) is not None for field in scoring_fields)
                and cache_entry_is_fresh(
                    {'timestamp': entry.get('info_timestamp')}, YF_CACHE_EXPIRY_DAYS * 24
                )
            )
            if not info_is_usable:
                metadata_stats['live_requests'] += 1
                ticker = yf.Ticker(symbol)
                raw_info = ticker.info or {}
                info = {field: raw_info.get(field) for field in info_fields}
                if not (any(info.get(field) is not None for field in identity_fields)
                        and any(info.get(field) is not None for field in scoring_fields)):
                    metadata_stats['incomplete_symbols'].append(symbol)
                entry['info'] = info
                entry['info_timestamp'] = datetime.now(UTC).isoformat()
                entry['timestamp'] = entry['info_timestamp']
                cache[symbol] = entry
                time.sleep(delay + random.uniform(0, 0.3))
            else:
                metadata_stats['hits'] += 1
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
            metadata_stats['failed_symbols'].append(symbol)
            print(f'Error retrieving yfinance metadata for {symbol}: {exc}')
    metadata_stats['save_succeeded'] = save_json_cache(YF_CACHE_FILE, cache)
    cache_diagnostics['yfinance_metadata'] = metadata_stats
    print('ETF METADATA CACHE ' + json.dumps(metadata_stats))
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
        'DownsideVolatility',
        'MaxDrawdown',
        'PositiveDayPct',
        'TrendR2',
        'AnnualizedTrend',
        'Distance50DMA',
        'Distance200DMA',
        'Distance52WHigh',
        'Largest1DayMove',
        'Largest5DayMove',
        'MomentumAcceleration',
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
        weights = {'Quality': 0.35, 'Value': 0.10, 'Momentum': 0.55}
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
        'DownsideVolatility',
        'MaxDrawdown',
        'PositiveDayPct',
        'TrendR2',
        'AnnualizedTrend',
        'Distance50DMA',
        'Distance200DMA',
        'Distance52WHigh',
        'Largest1DayMove',
        'Largest5DayMove',
        'MomentumAcceleration',
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
    # Add stock-style price-path quality and an entry-fragility penalty.
    def percentile_score(column, ascending=True):
        if column not in df.columns:
            return pd.Series(50.0, index=df.index)
        values = pd.to_numeric(df[column], errors='coerce')
        return (values.rank(pct=True, ascending=ascending) * 100).fillna(50)

    df['PricePathQuality'] = (
        0.20 * percentile_score('Volatility 1Y', ascending=False)
        + 0.20 * percentile_score('DownsideVolatility', ascending=False)
        + 0.25 * percentile_score('MaxDrawdown', ascending=True)
        + 0.15 * percentile_score('PositiveDayPct', ascending=True)
        + 0.20 * percentile_score('TrendR2', ascending=True)
    ).clip(0, 100)
    df['MomentumScore'] = (
        0.80 * df['MomentumScore'] + 0.20 * df['PricePathQuality']
    ).clip(0, 100)

    entry_components = []
    for column in (
        'Distance50DMA', 'MomentumAcceleration', 'Largest5DayMove',
        'Volatility 1Y',
    ):
        if column in df.columns:
            entry_components.append(
                pd.to_numeric(df[column], errors='coerce').rank(pct=True) * 100
            )
    if entry_components:
        quantitative_entry_risk = pd.concat(
            entry_components, axis=1
        ).mean(axis=1, skipna=True).fillna(50)
        overextension_penalty = (
            (quantitative_entry_risk - 75).clip(lower=0) / 25 * 15
        ).clip(0, 15)
    else:
        quantitative_entry_risk = pd.Series(50.0, index=df.index)
        overextension_penalty = pd.Series(0.0, index=df.index)
    df['QuantitativeEntryRisk'] = quantitative_entry_risk.clip(0, 100)
    df['OverextensionPenalty'] = overextension_penalty
    df['MomentumScore'] = (
        df['MomentumScore'] - df['OverextensionPenalty']
    ).clip(0, 100)

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


CANONICAL_MARKET_SECTORS = (
    'Basic Materials', 'Communication Services', 'Consumer Cyclical',
    'Consumer Defensive', 'Energy', 'Financial Services', 'Healthcare',
    'Industrials', 'Real Estate', 'Technology', 'Utilities',
)
MARKET_CONTEXT_SCHEMA_FIELDS = (
    'as_of_date', 'market_status', 'market_summary', 'market_intro',
    'market_direction', 'major_drivers', 'macro_conditions',
    'strong_sectors', 'weak_sectors', 'strong_exposures', 'weak_exposures',
    'sector_context', 'factor_and_theme_context', 'active_risk_events',
    'sources',
)


def market_cache_contract(prompt_template, model):
    schema_hash = stable_json_hash(MARKET_CONTEXT_SCHEMA_FIELDS)
    contract_hash = stable_json_hash({
        'contract': 'shared_stock_etf_market_context_v1',
        'model': model, 'google_search': True,
        'prompt_template': prompt_template.strip(), 'schema_hash': schema_hash,
    })
    return contract_hash, schema_hash


def read_shared_market_cache(path, prompt_template, model, ttl_hours):
    contract_hash, schema_hash = market_cache_contract(prompt_template, model)
    cache = load_json_object(path)
    entries = cache.get('entries')
    if not isinstance(entries, dict):
        entries = {}
    cache['entries'] = entries
    entry = entries.get(contract_hash)
    diag = {
        'path': str(Path(path).resolve()), 'contract_hash': contract_hash,
        'schema_hash': schema_hash, 'entries': len(entries), 'hit': False,
        'age_hours': None, 'ttl_hours': ttl_hours, 'reason': 'contract_not_found',
    }
    context = None
    if not Path(path).exists():
        diag['reason'] = 'file_missing'
    elif not entries:
        diag['reason'] = 'legacy_empty_or_malformed_cache'
    if isinstance(entry, dict):
        diag['age_hours'] = cache_age_hours(entry.get('created_at') or entry.get('timestamp'))
        if entry.get('model') != model:
            diag['reason'] = 'model_mismatch'
        elif not cache_entry_is_fresh(entry, ttl_hours):
            diag['reason'] = 'expired_or_invalid_timestamp'
        else:
            try:
                context = validate_current_market_context(entry.get('data') or entry.get('market_context'))
                diag.update(hit=True, reason='validated_shared_contract')
            except Exception as exc:
                diag['reason'] = 'validation_failed'
                diag['error'] = str(exc)
    return cache, context, diag


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
    seen_event_ids = set()
    for index, event in enumerate(data['active_risk_events']):
        if not isinstance(event, dict):
            raise ValueError(
                f'Market risk event {index + 1} must be an object.'
            )
        event_id = str(event.get('event_id') or '').strip()
        if not re.fullmatch(r'[A-Z0-9]+(?:_[A-Z0-9]+)*', event_id):
            raise ValueError(
                f'Market risk event {index + 1} has invalid event_id '
                f'{event_id!r}; expected UPPER_SNAKE_CASE.'
            )
        if event_id in seen_event_ids:
            raise ValueError(f'Duplicate market risk event_id {event_id!r}.')
        seen_event_ids.add(event_id)
        if not str(event.get('description') or '').strip():
            raise ValueError(f'Market risk event {event_id} has no description.')
        affected_industries = event.get('affected_industries')
        if (
            not isinstance(affected_industries, list)
            or not affected_industries
            or any(not str(value).strip() for value in affected_industries)
        ):
            raise ValueError(
                f'Market risk event {event_id} must have affected_industries.'
            )
        if not str(event.get('normalization_risk') or '').strip():
            raise ValueError(
                f'Market risk event {event_id} has no normalization_risk.'
            )
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
    data['sources'] = validate_sources(
        data['sources'], 'Market context', minimum=2, maximum=10
    )
    return data


def validate_current_market_context(data):
    """Validate the exact shared stock/ETF market-context contract."""
    # Validate the original contract before normalization can hide omissions,
    # replace malformed fields, or discard source publication dates.
    validate_market_context(data)
    required = set(MARKET_CONTEXT_SCHEMA_FIELDS)
    missing = required.difference(data)
    if missing:
        raise ValueError(
            f'Market context is missing fields: {sorted(missing)}'
        )
    if data.get('market_status') not in {'STRONG', 'MIXED', 'WEAK'}:
        raise ValueError('Market context has an invalid market_status.')
    if not isinstance(data.get('sector_context'), dict):
        raise ValueError('Market context requires sector_context object.')
    normalized_context = {
        str(key).strip().casefold(): str(value or '').strip()
        for key, value in data.get('sector_context', {}).items()
    }
    missing_sectors = [
        sector for sector in CANONICAL_MARKET_SECTORS
        if not normalized_context.get(sector.casefold())
    ]
    if missing_sectors:
        raise ValueError(
            'Market context is missing required sector context: '
            + ', '.join(missing_sectors)
        )
    if not isinstance(data.get('factor_and_theme_context'), dict):
        raise ValueError(
            'Market context requires factor_and_theme_context object.'
        )
    validate_sources(data['sources'], 'Shared market context', minimum=6, maximum=10)
    if not isinstance(data['sources'], list) or any(
        not isinstance(source, dict) or not str(source.get('title') or '').strip()
        or not str(source.get('url') or '').strip().startswith(('https://', 'http://'))
        for source in data['sources']
    ):
        raise ValueError('Shared market sources require explicit title and URL fields.')
    if str(data.get('market_summary') or '').strip() != str(
        data.get('market_intro') or ''
    ).strip():
        raise ValueError(
            'market_summary and market_intro must be identical.'
        )
    return dict(data)


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
    if result.get('mechanism_evidence_source_index') is not None:
        try:
            result['mechanism_evidence_source_index'] = int(
                result['mechanism_evidence_source_index']
            )
        except (TypeError, ValueError):
            pass
    if result.get('adverse_change_date') is not None:
        result['adverse_change_date'] = normalize_adverse_change_date(
            result.get('adverse_change_date')
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
    """Recognize a concrete borderline vulnerability before deterioration begins."""
    if result.get('mechanism_status') != 'HYPOTHETICAL':
        return False
    if result.get('normalization_probability') != 'POSSIBLE':
        return False
    levels = {'LOW': 0, 'MODERATE': 1, 'HIGH': 2}
    materiality = levels.get(result.get('risk_materiality'), -1)
    dependence = levels.get(result.get('driver_dependence'), -1)
    if materiality < 1 or dependence < 1 or max(materiality, dependence) < 2:
        return False
    if result.get('risk_basis') not in {
        'TEMPORARY_DRIVER_NORMALIZATION', 'NORMALIZED_EXPOSURE_DETERIORATION'
    }:
        return False
    if result.get('probability_basis') in {None, 'NONE'}:
        return False
    if result.get('evidence_scope') not in {'FUND_SPECIFIC', 'EXPOSURE_SPECIFIC'}:
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

    return True


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
        result['mechanism_evidence_source_index'] = None
        if result.get('evidence_scope') not in {
            'FUND_SPECIFIC', 'EXPOSURE_SPECIFIC', 'MARKET_ONLY', 'NONE'
        }:
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
        result['mechanism_evidence_source_index'] = None
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


def normalize_adverse_change_date(value):
    """Normalize safely parseable evidence dates without changing their meaning."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Fast path for ISO dates/timestamps, including trailing Z.
    try:
        parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
        return parsed.date().isoformat()
    except (TypeError, ValueError):
        pass
    # Gemini occasionally emits unambiguous human-readable dates. Pandas is
    # already a dependency and gives us a deterministic local normalization.
    try:
        parsed = pd.to_datetime(text, errors='coerce', utc=True)
    except Exception:
        parsed = None
    if parsed is None or pd.isna(parsed):
        return text
    try:
        return parsed.date().isoformat()
    except Exception:
        return text


def canonicalize_source_url(value):
    """Canonicalize harmless URL representation differences for provenance checks."""
    text = str(value or '').strip()
    if not text:
        return ''
    # Do not attempt network resolution. Normalize only deterministic syntax
    # differences that cannot change the underlying resource identity.
    text = re.sub(r'#.*$', '', text).rstrip('/')
    text = re.sub(r'([?&])(utm_[^=&]+|gclid|fbclid)=[^&]*&?', r'\1', text, flags=re.I)
    text = text.replace('?&', '?').rstrip('?&')
    return text


def bind_mechanism_evidence_source(result):
    """Bind OBSERVED evidence to an already validated source deterministically.

    New responses use mechanism_evidence_source_index, eliminating fragile URL
    copying. Older cached responses remain compatible through canonical URL
    matching.
    """
    result = dict(result)
    sources = result.get('sources') or []
    index = result.get('mechanism_evidence_source_index')
    if index is not None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = None
        if index is not None and 0 <= index < len(sources):
            result['mechanism_evidence_source_index'] = index
            result['mechanism_evidence_source'] = str(
                sources[index].get('url') or ''
            ).strip()
            return result

    mechanism_source = canonicalize_source_url(
        result.get('mechanism_evidence_source')
    )
    if mechanism_source:
        matches = [
            idx for idx, source in enumerate(sources)
            if canonicalize_source_url(source.get('url')) == mechanism_source
        ]
        if len(matches) == 1:
            index = matches[0]
            result['mechanism_evidence_source_index'] = index
            result['mechanism_evidence_source'] = str(
                sources[index].get('url') or ''
            ).strip()
    return result


def _risk_event_terms(value):
    """Return stable comparison terms for dynamic event-ID reconciliation."""
    text = ' '.join(str(value or '').upper().split())
    terms = re.findall(r'[A-Z0-9]+', text)
    stopwords = {
        'A', 'AN', 'AND', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CURRENT',
        'EVENT', 'FOR', 'FROM', 'IN', 'IS', 'IT', 'OF', 'ON', 'OR',
        'RISK', 'THE', 'TO', 'WITH', 'WITHOUT', 'COULD', 'WOULD', 'MAY',
        'MATERIAL', 'MATERIALLY', 'NORMALIZATION', 'NORMALIZE', 'NORMALIZING',
        'DISRUPTION', 'DISRUPTIONS', 'TENSION', 'TENSIONS',
    }
    return {
        term for term in terms
        if len(term) >= 3 and term not in stopwords
    }


def _cosine_similarity(left, right):
    """Return bounded cosine similarity, or None for unusable vectors."""
    try:
        left = np.asarray(left, dtype=float).reshape(-1)
        right = np.asarray(right, dtype=float).reshape(-1)
        if left.size == 0 or right.size == 0 or left.size != right.size:
            return None
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if not np.isfinite(denominator) or denominator <= 0:
            return None
        value = float(np.dot(left, right) / denominator)
        if not np.isfinite(value):
            return None
        return max(-1.0, min(1.0, value))
    except Exception:
        return None


def _risk_event_identity_text_from_result(result, raw_event_id):
    return ' '.join(filter(None, [
        f'event label {raw_event_id}',
        str(result.get('reversal_mechanism') or ''),
        str(result.get('current_driver_evidence') or ''),
        str(result.get('probability_evidence') or ''),
        str(result.get('primary_reversal_channel') or ''),
    ])).strip()


def _risk_event_transmission_text_from_result(result):
    return ' '.join(filter(None, [
        f'portfolio group {result.get("_portfolio_group") or ""}',
        f'fund exposure {result.get("exposure_group") or ""}',
        f'risk exposure {result.get("risk_exposure_group") or ""}',
        str(result.get('reversal_mechanism') or ''),
        str(result.get('material_effect') or ''),
        str(result.get('holdings_evidence') or ''),
    ])).strip()


def _risk_event_identity_text_from_catalog(event):
    return ' '.join(filter(None, [
        f'event id {event.get("event_id") or ""}',
        str(event.get('description') or ''),
        str(event.get('normalization_risk') or ''),
    ])).strip()


def _risk_event_transmission_text_from_catalog(event):
    return ' '.join(filter(None, [
        'affected industries ' + ' '.join(
            str(x) for x in (event.get('affected_industries') or [])
        ),
        'beneficiaries ' + ' '.join(
            str(x) for x in (event.get('beneficiaries') or [])
        ),
        str(event.get('normalization_risk') or ''),
        str(event.get('description') or ''),
    ])).strip()


def _get_fastembed_model():
    """Load FastEmbed lazily and disable it for the run after a failure."""
    global _fastembed_model, _fastembed_unavailable_reason
    if _fastembed_model is not None:
        return _fastembed_model
    if _fastembed_unavailable_reason is not None:
        return None
    try:
        from fastembed import TextEmbedding

        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _fastembed_model = TextEmbedding(
            model_name=FASTEMBED_MODEL_NAME,
            cache_dir=str(FASTEMBED_CACHE_DIR),
        )
        print(
            'FastEmbed ETF risk-event matcher ready: '
            f'model={FASTEMBED_MODEL_NAME}, cache={FASTEMBED_CACHE_DIR}.'
        )
        return _fastembed_model
    except Exception as exc:
        _fastembed_unavailable_reason = f'{type(exc).__name__}: {exc}'
        print(
            'FastEmbed ETF risk-event matcher unavailable; using deterministic '
            f'lexical fallback for this run: {_fastembed_unavailable_reason}'
        )
        return None


def _lexical_risk_event_scores(result, raw_event_id, catalog):
    """Deterministic event matcher retained as a technical fallback."""
    id_terms = _risk_event_terms(raw_event_id)
    exposure_terms = _risk_event_terms(' '.join([
        str(result.get('risk_exposure_group') or ''),
        str(result.get('exposure_group') or ''),
    ]))
    portfolio_terms = _risk_event_terms(result.get('_portfolio_group'))
    evidence_terms = _risk_event_terms(' '.join([
        str(result.get('reversal_mechanism') or ''),
        str(result.get('current_driver_evidence') or ''),
        str(result.get('probability_evidence') or ''),
        str(result.get('material_effect') or ''),
        str(result.get('holdings_evidence') or ''),
    ]))

    scored = []
    for event_id, event in catalog.items():
        event_id_terms = _risk_event_terms(event_id)
        event_context_terms = _risk_event_terms(' '.join([
            str(event.get('description') or ''),
            ' '.join(str(x) for x in (event.get('affected_industries') or [])),
            ' '.join(str(x) for x in (event.get('beneficiaries') or [])),
            str(event.get('normalization_risk') or ''),
        ]))
        all_event_terms = event_id_terms | event_context_terms
        score = 0.0
        score += 3.0 * len(id_terms & all_event_terms)
        score += 3.0 * len(exposure_terms & all_event_terms)
        score += 2.0 * len(portfolio_terms & all_event_terms)
        score += 0.5 * min(6, len(evidence_terms & all_event_terms))
        structural_overlap = bool(
            (id_terms & all_event_terms)
            or (exposure_terms & all_event_terms)
            or (portfolio_terms & all_event_terms)
        )
        if structural_overlap:
            scored.append((score, event_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored


def _risk_event_structural_compatibility(result, raw_event_id, event):
    """Require a concrete fund-exposure bridge beyond vector similarity."""
    generic_channel_terms = {
        'MARKET', 'MARKETS', 'PRICE', 'PRICES', 'RATE', 'RATES',
        'STRONG', 'WEAK', 'HIGH', 'LOW', 'GLOBAL', 'SUPPLY', 'DEMAND',
        'CONDITIONS', 'ECONOMIC', 'ECONOMY', 'INDUSTRY', 'INDUSTRIES',
        'FUND', 'FUNDS', 'ETF', 'EQUITY', 'EQUITIES',
    }
    id_terms = _risk_event_terms(raw_event_id)
    exposure_terms = _risk_event_terms(' '.join([
        str(result.get('risk_exposure_group') or ''),
        str(result.get('exposure_group') or ''),
    ]))
    exposure_specific_terms = exposure_terms - generic_channel_terms
    portfolio_terms = _risk_event_terms(result.get('_portfolio_group'))

    event_id_terms = _risk_event_terms(event.get('event_id'))
    event_industry_terms = _risk_event_terms(
        ' '.join(str(x) for x in (event.get('affected_industries') or []))
    )
    event_context_terms = _risk_event_terms(' '.join([
        str(event.get('description') or ''),
        ' '.join(str(x) for x in (event.get('affected_industries') or [])),
        ' '.join(str(x) for x in (event.get('beneficiaries') or [])),
        str(event.get('normalization_risk') or ''),
    ]))
    all_event_terms = event_id_terms | event_context_terms

    event_label_overlap = id_terms & all_event_terms
    exposure_overlap = exposure_specific_terms & all_event_terms
    portfolio_overlap = portfolio_terms & event_industry_terms
    return {
        'compatible': bool(exposure_overlap or portfolio_overlap),
        'event_label_overlap': sorted(event_label_overlap),
        'exposure_overlap': sorted(exposure_overlap),
        'portfolio_overlap': sorted(portfolio_overlap),
    }


def _semantic_risk_event_scores(result, raw_event_id, catalog):
    """Rank current market events by identity and economic transmission."""
    global _fastembed_model, _fastembed_unavailable_reason
    model = _get_fastembed_model()
    if model is None:
        return None
    try:
        ordered_events = list(catalog.items())
        catalog_payload = [
            {
                'event_id': event_id,
                'description': event.get('description'),
                'affected_industries': event.get('affected_industries'),
                'beneficiaries': event.get('beneficiaries'),
                'normalization_risk': event.get('normalization_risk'),
            }
            for event_id, event in ordered_events
        ]
        cache_key = (FASTEMBED_MODEL_NAME, stable_json_hash(catalog_payload))
        catalog_vectors = _risk_event_embedding_cache.get(cache_key)
        if catalog_vectors is None:
            catalog_texts = []
            for _, event in ordered_events:
                catalog_texts.extend([
                    _risk_event_identity_text_from_catalog(event),
                    _risk_event_transmission_text_from_catalog(event),
                ])
            embedded = list(model.embed(catalog_texts))
            if len(embedded) != 2 * len(ordered_events):
                raise ValueError(
                    'FastEmbed returned an unexpected number of event vectors.'
                )
            catalog_vectors = {}
            for index, (event_id, _) in enumerate(ordered_events):
                catalog_vectors[event_id] = {
                    'identity': embedded[index * 2],
                    'transmission': embedded[index * 2 + 1],
                }
            _risk_event_embedding_cache.clear()
            _risk_event_embedding_cache[cache_key] = catalog_vectors

        etf_vectors = list(model.embed([
            _risk_event_identity_text_from_result(result, raw_event_id),
            _risk_event_transmission_text_from_result(result),
        ]))
        if len(etf_vectors) != 2:
            raise ValueError(
                'FastEmbed returned an unexpected number of ETF vectors.'
            )

        scored = []
        for event_id, event in ordered_events:
            identity_similarity = _cosine_similarity(
                etf_vectors[0], catalog_vectors[event_id]['identity']
            )
            transmission_similarity = _cosine_similarity(
                etf_vectors[1], catalog_vectors[event_id]['transmission']
            )
            if identity_similarity is None or transmission_similarity is None:
                continue
            structural = _risk_event_structural_compatibility(
                result, raw_event_id, event
            )
            scored.append({
                'event_id': event_id,
                'identity_similarity': identity_similarity,
                'transmission_similarity': transmission_similarity,
                'combined_similarity': (
                    0.55 * identity_similarity
                    + 0.45 * transmission_similarity
                ),
                'structural': structural,
            })
        scored.sort(
            key=lambda item: (-item['combined_similarity'], item['event_id'])
        )
        return scored
    except Exception as exc:
        _fastembed_model = None
        _fastembed_unavailable_reason = f'{type(exc).__name__}: {exc}'
        _risk_event_embedding_cache.clear()
        print(
            'FastEmbed ETF risk-event scoring failed; using deterministic '
            f'lexical fallback: {type(exc).__name__}: {exc}'
        )
        runtime_reconciliation_diagnostics.append({
            'symbol': str(result.get('symbol') or '').strip().upper(),
            'type': 'dynamic_risk_event_embedding_fallback',
            'reason': f'{type(exc).__name__}: {exc}',
        })
        return None


def reconcile_dynamic_risk_event_id(result, active_risk_events):
    """Map a noncanonical Gemini event label to the shared current catalog."""
    raw_event_id = str(result.get('primary_risk_event_id') or '').strip()
    if not raw_event_id or not isinstance(active_risk_events, list):
        return raw_event_id or None
    catalog = {
        str(event.get('event_id') or '').strip(): event
        for event in active_risk_events
        if isinstance(event, dict) and str(event.get('event_id') or '').strip()
    }
    if raw_event_id in catalog or not catalog:
        return raw_event_id

    semantic_scores = _semantic_risk_event_scores(
        result, raw_event_id, catalog
    )
    if semantic_scores:
        best = semantic_scores[0]
        runner_up = (
            semantic_scores[1]['combined_similarity']
            if len(semantic_scores) > 1 else -1.0
        )
        margin = best['combined_similarity'] - runner_up
        accepted = (
            best['identity_similarity'] >= FASTEMBED_IDENTITY_MIN_SIMILARITY
            and best['transmission_similarity']
            >= FASTEMBED_TRANSMISSION_MIN_SIMILARITY
            and best['combined_similarity']
            >= FASTEMBED_COMBINED_MIN_SIMILARITY
            and margin >= FASTEMBED_UNIQUENESS_MARGIN
            and best['structural']['compatible']
        )
        symbol = str(result.get('symbol') or '').strip().upper()
        if not accepted:
            print(
                f'Semantic ETF risk-event reconciliation declined for {symbol}: '
                f'returned={raw_event_id!r}, best={best["event_id"]!r}, '
                f'identity={best["identity_similarity"]:.3f}, '
                f'transmission={best["transmission_similarity"]:.3f}, '
                f'combined={best["combined_similarity"]:.3f}, '
                f'margin={margin:.3f}, '
                f'structural={best["structural"]["compatible"]}.'
            )
            runtime_reconciliation_diagnostics.append({
                'symbol': symbol,
                'type': 'dynamic_risk_event_id_semantic_decline',
                'field': 'primary_risk_event_id',
                'from': raw_event_id,
                'candidate': best['event_id'],
                'identity_similarity': best['identity_similarity'],
                'transmission_similarity': best['transmission_similarity'],
                'combined_similarity': best['combined_similarity'],
                'runner_up_similarity': runner_up,
                'margin': margin,
                'structural': best['structural'],
            })
            return raw_event_id

        best_event_id = best['event_id']
        print(
            f'Reconciled semantic primary_risk_event_id for ETF {symbol}: '
            f'{raw_event_id!r} -> {best_event_id!r} '
            f'(identity={best["identity_similarity"]:.3f}, '
            f'transmission={best["transmission_similarity"]:.3f}, '
            f'combined={best["combined_similarity"]:.3f}, '
            f'margin={margin:.3f}).'
        )
        runtime_reconciliation_diagnostics.append({
            'symbol': symbol,
            'type': 'dynamic_risk_event_id_semantic_reconciliation',
            'field': 'primary_risk_event_id',
            'from': raw_event_id,
            'to': best_event_id,
            'identity_similarity': best['identity_similarity'],
            'transmission_similarity': best['transmission_similarity'],
            'combined_similarity': best['combined_similarity'],
            'runner_up_similarity': runner_up,
            'margin': margin,
            'structural': best['structural'],
            'model': FASTEMBED_MODEL_NAME,
        })
        return best_event_id

    # Technical fallback only. A semantic near-miss never falls through here.
    scored = _lexical_risk_event_scores(result, raw_event_id, catalog)
    if not scored:
        return raw_event_id
    best_score, best_event_id = scored[0]
    runner_up_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < 6.0 or best_score - runner_up_score < 2.0:
        return raw_event_id

    symbol = str(result.get('symbol') or '').strip().upper()
    print(
        f'Reconciled fallback primary_risk_event_id for ETF {symbol}: '
        f'{raw_event_id!r} -> {best_event_id!r} '
        f'(score={best_score:.1f}, margin={best_score - runner_up_score:.1f}).'
    )
    runtime_reconciliation_diagnostics.append({
        'symbol': symbol,
        'type': 'dynamic_risk_event_id_lexical_fallback',
        'field': 'primary_risk_event_id',
        'from': raw_event_id,
        'to': best_event_id,
        'score': best_score,
        'runner_up_score': runner_up_score,
        'fallback_reason': (
            _fastembed_unavailable_reason or 'embedding_runtime_failure'
        ),
    })
    return best_event_id


def reconcile_etf_risk_event_fields(result, candidate, active_risk_events):
    """Reconcile only event identity; leave Gemini's investment judgment intact."""
    result = dict(result)
    match_context = dict(result)
    match_context['_portfolio_group'] = str(
        candidate.get('PortfolioGroup')
        or candidate.get('CanonicalSector')
        or candidate.get('Category')
        or ''
    ).strip()
    event_id = str(result.get('primary_risk_event_id') or '').strip() or None
    allowed_ids = {
        str(event.get('event_id') or '').strip()
        for event in (active_risk_events or [])
        if isinstance(event, dict) and str(event.get('event_id') or '').strip()
    }
    if event_id and event_id not in allowed_ids:
        match_context['primary_risk_event_id'] = event_id
        event_id = reconcile_dynamic_risk_event_id(
            match_context, active_risk_events
        )
    if event_id and event_id not in allowed_ids:
        raise ValueError(
            f'{result.get("symbol") or candidate.get("Symbol")} has unknown '
            f'primary_risk_event_id {event_id!r}; use an exact event_id from '
            'MARKET_CONTEXT.active_risk_events or null.'
        )
    result['primary_risk_event_id'] = event_id

    exposure_group = result.get('risk_exposure_group')
    if exposure_group is not None:
        exposure_group = re.sub(
            r'[^A-Z0-9]+', '_', str(exposure_group).upper()
        ).strip('_') or None
    if event_id and not exposure_group:
        raise ValueError(
            f'{result.get("symbol") or candidate.get("Symbol")} has a primary '
            'risk event without risk_exposure_group.'
        )
    result['risk_exposure_group'] = exposure_group
    return result


def validate_etf_result(
    result, candidate, minimum_sources=2, maximum_sources=5,
    active_risk_events=None,
):
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
    result = bind_mechanism_evidence_source(result)
    if active_risk_events is not None:
        result = reconcile_etf_risk_event_fields(
            result, candidate, active_risk_events
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
        date_text = normalize_adverse_change_date(
            result.get('adverse_change_date')
        )
        result['adverse_change_date'] = date_text
        try:
            evidence_date = datetime.fromisoformat(
                str(date_text).replace('Z', '+00:00')
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
        source_index = result.get('mechanism_evidence_source_index')
        if not isinstance(source_index, int) or not (
            0 <= source_index < len(result['sources'])
        ):
            raise ValueError(
                f'{symbol} observed mechanism lacks a valid '
                'mechanism_evidence_source_index.'
            )
        result['mechanism_evidence_source'] = str(
            result['sources'][source_index].get('url') or ''
        ).strip()
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
                'cautious-borderline normalization evidence.'
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


def validate_etf_batch(
    data, candidates, minimum_sources=2, maximum_sources=5,
    active_risk_events=None,
):
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
                raw_by_symbol[symbol], candidate, minimum_sources,
                maximum_sources, active_risk_events,
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
            'mechanism_evidence_source_index',
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
            'mechanism_evidence_source', 'mechanism_evidence_source_index',
            'market context alone',
            'etf-specific transmission path', 'material_effect is generic',
        )):
            allowed_fields.update({
                'mechanism_status', 'reversal_mechanism',
                'normalization_probability', 'probability_evidence',
                'risk_materiality', 'driver_dependence', 'material_effect',
                'adverse_change_observed', 'adverse_change_date',
                'adverse_change_indicator', 'mechanism_evidence_source',
                'mechanism_evidence_source_index',
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
                'mechanism_evidence_source_index',
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
            'evidence_timestamp': research_cache['entries'].get(cache_key, {}).get('timestamp'),
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
    """Block a full-sector candidate unless its quantitative challenger score can beat an incumbent."""
    sector_key = candidate_sector_group(candidate).casefold()
    selected_in_sector = [
        item for item in selected
        if candidate_sector_group(item['candidate']).casefold() == sector_key
    ]
    if len(selected_in_sector) < max_per_sector:
        return False
    candidate_score = quantitative_challenger_score(candidate)
    worst_selected_score = min(
        quantitative_challenger_score(item['candidate'])
        for item in selected_in_sector
    )
    return candidate_score + challenger_score_epsilon < worst_selected_score


def lower_ranked_candidate_blocked_by_factor_family_capacity(candidate, selected):
    """Apply the same challenger exception to an already-full economic factor family."""
    family = etf_factor_family(candidate)
    selected_in_family = [
        item for item in selected
        if etf_factor_family(item['candidate'], item.get('research')) == family
    ]
    if len(selected_in_family) < max_etfs_per_factor_family:
        return False
    candidate_score = quantitative_challenger_score(candidate)
    worst_selected_score = min(
        quantitative_challenger_score(item['candidate'])
        for item in selected_in_family
    )
    return candidate_score + challenger_score_epsilon < worst_selected_score


def quantitative_challenger_score(candidate):
    """Pre-research score: QVM dominates; exceptional relative strength may bridge a small QVM gap."""
    qvm = float(candidate.get('QVMScore') or 0.0)
    relative = float(candidate.get('BenchmarkRelativeScore') or 0.0)
    bonus = max(0.0, relative - challenger_relative_bonus_start) * challenger_relative_bonus_per_point
    return qvm + min(challenger_relative_bonus_cap, bonus)


def capacity_challenge_dimensions(candidate, selected):
    """Return full portfolio constraints that this candidate is strong enough to challenge."""
    dimensions = []
    sector_key = candidate_sector_group(candidate).casefold()
    same_sector = [
        item for item in selected
        if candidate_sector_group(item['candidate']).casefold() == sector_key
    ]
    if len(same_sector) >= max_etfs_per_sector_group:
        candidate_score = quantitative_challenger_score(candidate)
        weakest = min(
            same_sector,
            key=lambda item: quantitative_challenger_score(item['candidate'])
        )
        weakest_score = quantitative_challenger_score(weakest['candidate'])
        if candidate_score + challenger_score_epsilon >= weakest_score:
            dimensions.append({
                'dimension': 'SECTOR',
                'group': candidate_sector_group(candidate),
                'incumbent': str(weakest['candidate']['Symbol']).upper(),
                'challenger_score': candidate_score,
                'incumbent_score': weakest_score,
            })
    family = etf_factor_family(candidate)
    same_family = [
        item for item in selected
        if etf_factor_family(item['candidate'], item.get('research')) == family
    ]
    if len(same_family) >= max_etfs_per_factor_family:
        candidate_score = quantitative_challenger_score(candidate)
        weakest = min(
            same_family,
            key=lambda item: quantitative_challenger_score(item['candidate'])
        )
        weakest_score = quantitative_challenger_score(weakest['candidate'])
        if candidate_score + challenger_score_epsilon >= weakest_score:
            dimensions.append({
                'dimension': 'FACTOR_FAMILY',
                'group': family,
                'incumbent': str(weakest['candidate']['Symbol']).upper(),
                'challenger_score': candidate_score,
                'incumbent_score': weakest_score,
            })
    return dimensions


def record_capacity_challenger(candidate, selected, tracker):
    symbol = str(candidate['Symbol']).upper()
    dimensions = capacity_challenge_dimensions(candidate, selected)
    if not dimensions:
        return
    prior = tracker.setdefault(symbol, [])
    seen = {(item['dimension'], item['group'], item['incumbent']) for item in prior}
    for item in dimensions:
        key = (item['dimension'], item['group'], item['incumbent'])
        if key not in seen:
            prior.append(item)
            seen.add(key)
            print(
                f'ETF capacity challenger admitted [{symbol}] against '
                f'{item["incumbent"]} ({item["dimension"]}={item["group"]}): '
                f'quantitative challenger score {item["challenger_score"]:.2f} vs '
                f'{item["incumbent_score"]:.2f}.'
            )


def etf_factor_family(candidate, research=None):
    """Group economically similar categories so labels cannot bypass diversification."""
    group = candidate_sector_group(candidate).upper()
    style = str(candidate.get('StyleCategory') or '').upper()
    driver = normalized_return_driver_key(research or {}) or ''
    text = ' '.join((group, style, driver))
    if 'FREE CASH FLOW' in text or 'QUALITY' in text:
        return 'QUALITY_CASHFLOW'
    if any(token in text for token in ('DIVIDEND', 'LARGE VALUE', 'VALUE')):
        return 'VALUE_INCOME'
    if any(token in text for token in ('TECHNOLOGY', 'MEGA_CAP_TECH', 'AI_CAPEX', 'SEMICONDUCTOR')):
        return 'GROWTH_TECH'
    if any(token in text for token in ('ENERGY', 'OIL', 'GAS')):
        return 'ENERGY_COMMODITY'
    if any(token in text for token in ('HEALTHCARE', 'BIOTECH', 'PHARMA')):
        return 'HEALTHCARE'
    if any(token in text for token in ('MATERIAL', 'METAL', 'MINING')):
        return 'MATERIALS'
    if 'MOMENTUM' in text:
        return 'MOMENTUM'
    return re.sub(r'[^A-Z0-9]+', '_', group).strip('_') or 'OTHER'


def etf_selection_merit(candidate, research):
    """Rank already-qualified ETFs by benchmark ambition without changing eligibility floors."""
    score = etf_final_score(candidate, research)
    relative = float(candidate.get('BenchmarkRelativeScore') or 0.0)
    if relative >= selection_relative_high_threshold:
        score += selection_relative_high_bonus
    elif relative >= selection_relative_good_threshold:
        score += selection_relative_good_bonus
    outlook = str(research.get('benchmark_outperformance_outlook') or 'UNCERTAIN').upper()
    continuation = str(research.get('continuation_strength') or 'ADEQUATE').upper()
    if outlook == 'LIKELY':
        score += selection_likely_bonus
    if continuation == 'STRONG':
        score += selection_strong_bonus
    return score


def pending_candidate_can_improve_full_portfolio(
    candidate, selected, rank_by_symbol, max_per_sector
):
    """Keep bounded benchmark-relative challengers even when raw QVM rank is lower."""
    if not selected:
        return True
    candidate_score = quantitative_challenger_score(candidate)
    sector_key = candidate_sector_group(candidate).casefold()
    same_sector = [
        item for item in selected
        if candidate_sector_group(item['candidate']).casefold() == sector_key
    ]
    family = etf_factor_family(candidate)
    same_family = [
        item for item in selected
        if etf_factor_family(item['candidate'], item.get('research')) == family
    ]
    if len(same_sector) >= max_per_sector:
        comparison_pool = same_sector
    elif len(same_family) >= max_etfs_per_factor_family:
        comparison_pool = same_family
    else:
        comparison_pool = selected
    if not comparison_pool:
        return True
    worst_score = min(quantitative_challenger_score(item['candidate']) for item in comparison_pool)
    return candidate_score + challenger_score_epsilon >= worst_score


def replacement_comparison_record(candidate, research, incumbent_item, dimension):
    """Build a stable audit record for a blocked candidate versus the weakest incumbent."""
    incumbent_candidate = incumbent_item['candidate']
    incumbent_research = incumbent_item['research']
    return {
        'dimension': dimension,
        'challenger_symbol': str(candidate['Symbol']).upper(),
        'incumbent_symbol': str(incumbent_candidate['Symbol']).upper(),
        'challenger_qvm': float(candidate.get('QVMScore') or 0.0),
        'incumbent_qvm': float(incumbent_candidate.get('QVMScore') or 0.0),
        'challenger_benchmark_relative': float(candidate.get('BenchmarkRelativeScore') or 0.0),
        'incumbent_benchmark_relative': float(incumbent_candidate.get('BenchmarkRelativeScore') or 0.0),
        'challenger_final_score': etf_final_score(candidate, research),
        'incumbent_final_score': etf_final_score(incumbent_candidate, incumbent_research),
        'challenger_required_score': etf_required_final_score(research, candidate),
        'incumbent_required_score': etf_required_final_score(incumbent_research, incumbent_candidate),
        'challenger_selection_merit': etf_selection_merit(candidate, research),
        'incumbent_selection_merit': etf_selection_merit(incumbent_candidate, incumbent_research),
        'challenger_risk': combined_etf_reversal_risk(research),
        'incumbent_risk': combined_etf_reversal_risk(incumbent_research),
        'challenger_continuation': str(research.get('continuation_strength') or 'ADEQUATE').upper(),
        'incumbent_continuation': str(incumbent_research.get('continuation_strength') or 'ADEQUATE').upper(),
        'challenger_benchmark_outlook': str(research.get('benchmark_outperformance_outlook') or 'UNCERTAIN').upper(),
        'incumbent_benchmark_outlook': str(incumbent_research.get('benchmark_outperformance_outlook') or 'UNCERTAIN').upper(),
        'challenger_benchmark_confidence': str(research.get('benchmark_outperformance_confidence') or 'MEDIUM').upper(),
        'incumbent_benchmark_confidence': str(incumbent_research.get('benchmark_outperformance_confidence') or 'MEDIUM').upper(),
    }


def build_replacement_diagnostics(candidate_records, research_by_symbol, selected):
    """Explain full-sector/family replacement tests after authoritative selection."""
    selected_symbols = {
        str(item['candidate']['Symbol']).upper() for item in selected
    }
    selected_by_sector = {}
    selected_by_family = {}
    for item in selected:
        sector = candidate_sector_group(item['candidate']).casefold()
        family = etf_factor_family(item['candidate'], item.get('research'))
        selected_by_sector.setdefault(sector, []).append(item)
        selected_by_family.setdefault(family, []).append(item)

    records = []
    for candidate in candidate_records:
        symbol = str(candidate['Symbol']).upper()
        if symbol in selected_symbols:
            continue
        research = research_by_symbol.get(symbol)
        if not research or not research.get('judgment_model'):
            continue
        if not research.get('eligible', True):
            continue
        risk = combined_etf_reversal_risk(research)
        if risk not in SELECTABLE_RISKS:
            continue
        if not etf_passes_benchmark_override(candidate, research):
            continue
        final_score = etf_final_score(candidate, research)
        if final_score + score_comparison_epsilon < etf_required_final_score(research, candidate):
            continue

        sector_key = candidate_sector_group(candidate).casefold()
        family = etf_factor_family(candidate, research)
        pools = []
        if len(selected_by_sector.get(sector_key, [])) >= max_etfs_per_sector_group:
            pools.append(('SECTOR', selected_by_sector[sector_key]))
        if len(selected_by_family.get(family, [])) >= max_etfs_per_factor_family:
            pools.append(('FACTOR_FAMILY', selected_by_family[family]))
        for dimension, pool in pools:
            incumbent = min(
                pool,
                key=lambda item: etf_selection_merit(item['candidate'], item['research'])
            )
            record = replacement_comparison_record(
                candidate, research, incumbent, dimension
            )
            gap = (
                record['challenger_selection_merit']
                - record['incumbent_selection_merit']
            )
            record['selection_merit_gap'] = gap
            record['result'] = (
                'CHALLENGER_WOULD_WIN'
                if gap > replacement_merit_epsilon
                else 'INCUMBENT_RETAINED'
            )
            records.append(record)

    records.sort(
        key=lambda row: row['selection_merit_gap'], reverse=True
    )
    return records[:replacement_diagnostics_top_n]


def print_replacement_diagnostics(records):
    if not records:
        print('ETF replacement audit: no qualified blocked candidates required a head-to-head test.')
        return
    print('ETF replacement audit:')
    for row in records:
        print(
            f'  {row["challenger_symbol"]} vs {row["incumbent_symbol"]} '
            f'[{row["dimension"]}]: {row["result"]}; '
            f'merit={row["challenger_selection_merit"]:.2f} vs '
            f'{row["incumbent_selection_merit"]:.2f} '
            f'(gap={row["selection_merit_gap"]:+.2f}); '
            f'QVM={row["challenger_qvm"]:.2f} vs {row["incumbent_qvm"]:.2f}; '
            f'benchmark_relative={row["challenger_benchmark_relative"]:.2f} vs '
            f'{row["incumbent_benchmark_relative"]:.2f}; '
            f'final={row["challenger_final_score"]:.2f}/{row["challenger_required_score"]:.2f} vs '
            f'{row["incumbent_final_score"]:.2f}/{row["incumbent_required_score"]:.2f}; '
            f'continuation={row["challenger_continuation"]} vs {row["incumbent_continuation"]}; '
            f'benchmark={row["challenger_benchmark_outlook"]}/{row["challenger_benchmark_confidence"]} vs '
            f'{row["incumbent_benchmark_outlook"]}/{row["incumbent_benchmark_confidence"]}; '
            f'risk={row["challenger_risk"]} vs {row["incumbent_risk"]}.'
        )


def assemble_research_batch(
    pending, fresh_candidates, limit, fresh_attempts, repair_attempts,
    fresh_limit, repair_limit, transport_fraction,
):
    """Prune before sizing, fill spare slots, and never send a symbol twice.

    Fresh candidates are already ranked/filtered by portfolio policy. This
    helper only controls work scheduling, not eligibility or selection.
    Transport retries get a soft cap when alternative work exists.
    """
    limit = max(0, int(limit))
    seen, exhausted, duplicates = set(), [], []
    ready = []
    for item in pending:
        symbol = str(item['candidate']['Symbol']).upper()
        attempts = fresh_attempts if item.get('needs_research') else repair_attempts
        ceiling = fresh_limit if item.get('needs_research') else repair_limit
        if attempts.get(symbol, 0) >= ceiling:
            exhausted.append(symbol)
            continue
        if symbol in seen:
            duplicates.append(symbol)
            continue
        seen.add(symbol)
        ready.append(item)
    fresh = []
    for candidate in fresh_candidates:
        symbol = str(candidate['Symbol']).upper()
        if symbol in seen or fresh_attempts.get(symbol, 0) >= fresh_limit:
            continue
        seen.add(symbol)
        fresh.append({'candidate': candidate, 'needs_research': True, 'targeted_backfill': True})
    # Repairs retain priority. Preserve rank/order within each class.
    ready.sort(key=lambda item: bool(item.get('needs_research')))
    transport_cap = max(1, int(round(limit * transport_fraction)))
    batch_items, remaining, transport_deferred = [], [], []
    transport_used = 0
    for item in ready:
        if len(batch_items) >= limit:
            remaining.append(item)
        elif item.get('transport_failure') and fresh and transport_used >= transport_cap:
            transport_deferred.append(item)
        else:
            batch_items.append(item)
            transport_used += int(bool(item.get('transport_failure')))
    fresh_added = fresh[:max(0, limit - len(batch_items))]
    batch_items.extend(fresh_added)
    # The transport cap must not leave otherwise usable capacity idle.
    take = max(0, limit - len(batch_items))
    batch_items.extend(transport_deferred[:take])
    remaining = transport_deferred[take:] + remaining
    return batch_items, remaining, {
        'planned_capacity': limit, 'actual_size': len(batch_items),
        'unused_capacity': max(0, limit - len(batch_items)),
        'exhausted_removed': sorted(set(exhausted)),
        'duplicates_removed': sorted(set(duplicates)),
        'available_fresh_candidates': len(fresh),
        'fresh_added': [str(x['candidate']['Symbol']).upper() for x in fresh_added],
        'transport_deferred': [str(x['candidate']['Symbol']).upper() for x in transport_deferred[take:]],
    }


def queue_portfolio_challengers(candidate_records, research_by_symbol, selected, pending_retries, queued_symbols):
    """Queue the strongest unresearched ETFs that can plausibly displace an incumbent."""
    if not challenger_research_enabled or len(selected) < target_selected_etfs:
        return []
    pending_symbols = {str(item['candidate']['Symbol']).upper() for item in pending_retries}
    candidates = []
    for candidate in candidate_records:
        symbol = str(candidate['Symbol']).upper()
        if symbol in research_by_symbol or symbol in pending_symbols or symbol in queued_symbols:
            continue
        qvm = float(candidate.get('QVMScore') or 0.0)
        relative = float(candidate.get('BenchmarkRelativeScore') or 0.0)
        # Either strong QVM or exceptional benchmark-relative strength is
        # enough to earn the deterministic head-to-head test. Requiring both
        # could leave a plausible incumbent replacement unresearched.
        if qvm < challenger_min_qvm and relative < challenger_min_relative_score:
            continue
        if (
            etf_optimistic_final_score(candidate) + score_comparison_epsilon
            < minimum_final_selection_score
        ):
            continue
        if not pending_candidate_can_improve_full_portfolio(
            candidate, selected, rank_by_symbol, max_etfs_per_sector_group
        ):
            continue
        candidates.append(candidate)
    candidates.sort(key=quantitative_challenger_score, reverse=True)
    remaining_slots = max(0, max_challenger_candidates_per_run - len(queued_symbols))
    if remaining_slots <= 0:
        return []
    queued = []
    for candidate in candidates[:remaining_slots]:
        symbol = str(candidate['Symbol']).upper()
        pending_retries.append({
            'candidate': candidate, 'needs_research': True,
            'portfolio_challenger': True,
        })
        queued_symbols.add(symbol)
        queued.append(symbol)
    if queued:
        print('Queued benchmark-relative portfolio challengers: ' + ', '.join(queued))
    return queued


def combined_etf_reversal_risk(research):
    order = {'MINIMAL': 0, 'LOW': 1, 'MODERATE': 2, 'ELEVATED': 3, 'SEVERE': 4}
    levels = [
        str(research.get(field) or '').upper()
        for field in ('reversal_risk', 'exposure_reversal_risk', 'entry_reversal_risk')
    ]
    valid = [level for level in levels if level in order]
    return max(valid, key=order.get) if valid else str(research.get('reversal_risk') or 'SEVERE').upper()


def normalized_return_driver_key(research):
    raw = research.get('risk_exposure_group') or research.get('exposure_group')
    return re.sub(r'[^A-Z0-9]+', '_', str(raw or '').upper()).strip('_') or None


def etf_final_score(candidate, research):
    """QVM stays dominant; judgment fields provide a bounded continuation overlay."""
    score = float(candidate.get('QVMScore') or 0.0)
    risk = combined_etf_reversal_risk(research)
    score += {'MINIMAL': 5.0, 'LOW': 3.0, 'MODERATE': -3.0,
              'ELEVATED': -1000.0, 'SEVERE': -1000.0}.get(risk, -1000.0)
    outlook = str(research.get('benchmark_outperformance_outlook') or 'UNCERTAIN').upper()
    confidence = str(research.get('benchmark_outperformance_confidence') or 'MEDIUM').upper()
    if outlook == 'LIKELY':
        score += benchmark_likely_bonus
    elif outlook == 'UNCERTAIN':
        score += benchmark_uncertain_penalty
    elif outlook == 'UNLIKELY' and confidence == 'HIGH':
        score += -1000.0
    elif outlook == 'UNLIKELY' and confidence == 'LOW':
        score += benchmark_unlikely_low_penalty
    else:
        score += benchmark_unlikely_medium_penalty
    continuation = str(research.get('continuation_strength') or 'ADEQUATE').upper()
    score += continuation_adjustments.get(continuation, continuation_adjustments['WEAK'])

    # Sector/index concentration is often intentional for ETFs. Penalize it
    # gently and cap the combined effect so a focused mandate is not counted
    # twice as both holdings concentration and construction concentration.
    concentration_penalty = 0.0
    for field in ('holdings_concentration', 'construction_concentration'):
        level = str(research.get(field) or 'LOW').upper()
        concentration_penalty += concentration_adjustments.get(level, -1.0)
    score += max(max_concentration_penalty, concentration_penalty)
    if etf_benchmark_relative_rescue(candidate, research):
        score += benchmark_relative_rescue_bonus
    return score


def etf_benchmark_relative_rescue(candidate, research):
    """Bounded rescue for quantitatively strong ETFs with uncertain, not negative, forward evidence."""
    if not benchmark_relative_rescue_enabled or not research or not research.get('judgment_model'):
        return False
    if str(research.get('benchmark_outperformance_outlook') or '').upper() != 'UNCERTAIN':
        return False
    if str(research.get('continuation_strength') or '').upper() not in {'STRONG', 'ADEQUATE'}:
        return False
    if combined_etf_reversal_risk(research) not in {'MINIMAL', 'LOW', 'MODERATE'}:
        return False
    return (
        float(candidate.get('QVMScore') or 0.0) >= benchmark_relative_rescue_min_qvm
        and float(candidate.get('BenchmarkRelativeScore') or 0.0) >= benchmark_relative_rescue_min_relative_score
    )


def etf_required_final_score(research, candidate=None):
    """Return the calibrated ETF-specific score floor for this judgment."""
    required = minimum_final_selection_score
    if not research or not research.get('judgment_model'):
        return required
    outlook = str(research.get('benchmark_outperformance_outlook') or 'UNCERTAIN').upper()
    risk = combined_etf_reversal_risk(research)
    if outlook == 'UNCERTAIN':
        required = max(required, minimum_uncertain_selection_score)
    if risk == 'MODERATE':
        required = max(required, minimum_moderate_selection_score)
    if candidate is not None and etf_benchmark_relative_rescue(candidate, research):
        required = min(required, benchmark_relative_rescue_floor)
    return required


def normalize_etf_judgment_results(data):
    """Normalize a few harmless JSON envelope variants without weakening field validation."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return None
    for key in ('results', 'classifications', 'judgments', 'items'):
        value = data.get(key)
        if isinstance(value, list):
            return value
    list_values = [value for value in data.values() if isinstance(value, list)]
    if len(list_values) == 1:
        return list_values[0]
    return None


def etf_optimistic_final_score(candidate):
    """Upper bound used to avoid researching ETFs that cannot clear even the baseline floor."""
    return (
        float(candidate.get('QVMScore') or 0.0)
        + 5.0  # best reversal-risk adjustment: MINIMAL
        + max(0.0, benchmark_likely_bonus)
        + max(0.0, continuation_adjustments.get('STRONG', 0.0))
    )


def etf_benchmark_override_required(candidate):
    return (
        require_likely_strong_below_benchmark_qvm
        and benchmark_qvm_floor is not None
        and float(candidate.get('QVMScore') or 0.0) < benchmark_qvm_floor
    )


def etf_passes_benchmark_override(candidate, research):
    if not etf_benchmark_override_required(candidate):
        return True
    return (
        str(research.get('benchmark_outperformance_outlook') or '').upper() == 'LIKELY'
        and str(research.get('continuation_strength') or '').upper() == 'STRONG'
    )


def normalize_etf_judgment_patch(symbol, patch, research):
    """Normalize harmless judge enum aliases and validate one ETF without affecting peers."""
    if not isinstance(patch, dict):
        raise ValueError(f'{symbol} judgment patch is not a JSON object.')

    def enum_value(field, default, allowed, aliases=None):
        raw = patch.get(field)
        if raw is None or str(raw).strip() == '':
            raw = default
        token = normalize_enum_token(raw)
        aliases = aliases or {}
        normalized = aliases.get(token, token)
        if normalized not in allowed:
            raise ValueError(
                f'{symbol} judgment returned invalid {field}={raw!r}; '
                f'expected one of {sorted(allowed)}.'
            )
        if token != normalized:
            print(
                f'Normalized ETF judgment enum [{symbol}] {field}: '
                f'{raw!r} -> {normalized}'
            )
        return normalized

    risk_aliases = {
        'MINIMUM': 'MINIMAL', 'VERY_LOW': 'LOW', 'MEDIUM': 'MODERATE',
        'VERY_HIGH': 'SEVERE',
    }
    continuation_aliases = {
        'HIGH': 'STRONG', 'ROBUST': 'STRONG', 'POSITIVE': 'STRONG',
        'MODERATE': 'ADEQUATE', 'MEDIUM': 'ADEQUATE', 'NEUTRAL': 'ADEQUATE',
        'MIXED': 'ADEQUATE', 'BASE': 'ADEQUATE', 'BASE_CASE': 'ADEQUATE',
        'LOW': 'WEAK', 'FRAGILE': 'WEAK', 'NEGATIVE': 'WEAK',
    }
    outlook_aliases = {
        'POSITIVE': 'LIKELY', 'OUTPERFORM': 'LIKELY', 'OUTPERFORMANCE_LIKELY': 'LIKELY',
        'NEUTRAL': 'UNCERTAIN', 'MIXED': 'UNCERTAIN', 'UNCLEAR': 'UNCERTAIN',
        'NEGATIVE': 'UNLIKELY', 'UNDERPERFORM': 'UNLIKELY', 'OUTPERFORMANCE_UNLIKELY': 'UNLIKELY',
    }
    confidence_aliases = {'MODERATE': 'MEDIUM'}
    concentration_aliases = {'MINIMAL': 'LOW', 'MEDIUM': 'MODERATE', 'ELEVATED': 'HIGH'}

    risk_default = research.get('reversal_risk') or 'LOW'
    exposure_risk = enum_value(
        'exposure_reversal_risk', risk_default,
        {'MINIMAL', 'LOW', 'MODERATE', 'ELEVATED', 'SEVERE'}, risk_aliases,
    )
    entry_risk = enum_value(
        'entry_reversal_risk', risk_default,
        {'MINIMAL', 'LOW', 'MODERATE', 'ELEVATED', 'SEVERE'}, risk_aliases,
    )
    overall = enum_value(
        'reversal_risk', risk_default,
        {'MINIMAL', 'LOW', 'MODERATE', 'ELEVATED', 'SEVERE'}, risk_aliases,
    )
    order = {'MINIMAL': 0, 'LOW': 1, 'MODERATE': 2, 'ELEVATED': 3, 'SEVERE': 4}
    overall = max((overall, exposure_risk, entry_risk), key=order.get)

    continuation = enum_value(
        'continuation_strength', 'ADEQUATE', {'STRONG', 'ADEQUATE', 'WEAK'},
        continuation_aliases,
    )
    benchmark_outlook = enum_value(
        'benchmark_outperformance_outlook', 'UNCERTAIN',
        {'LIKELY', 'UNCERTAIN', 'UNLIKELY'}, outlook_aliases,
    )
    benchmark_confidence = enum_value(
        'benchmark_outperformance_confidence', 'MEDIUM',
        {'LOW', 'MEDIUM', 'HIGH'}, confidence_aliases,
    )
    holdings_conc = enum_value(
        'holdings_concentration', 'LOW', {'LOW', 'MODERATE', 'HIGH'},
        concentration_aliases,
    )
    construction_conc = enum_value(
        'construction_concentration', 'LOW', {'LOW', 'MODERATE', 'HIGH'},
        concentration_aliases,
    )

    normalized = dict(patch)
    normalized.update({
        'exposure_reversal_risk': exposure_risk,
        'entry_reversal_risk': entry_risk,
        'reversal_risk': overall,
        'continuation_strength': continuation,
        'benchmark_outperformance_outlook': benchmark_outlook,
        'benchmark_outperformance_confidence': benchmark_confidence,
        'holdings_concentration': holdings_conc,
        'construction_concentration': construction_conc,
    })
    return normalized


def safe_judge_etf_research_pool(
    client, candidate_records, research_by_symbol, market_context, force=False,
):
    """Contain any unexpected judge failure so one classification bug cannot abort the run."""
    try:
        return judge_etf_research_pool(
            client, candidate_records, research_by_symbol, market_context,
            force=force,
        )
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, TotalRuntimeTimeout)):
            raise
        print(
            'WARNING — ETF judgment stage encountered an unexpected error; '
            'preserving grounded research and continuing with affected ETFs '
            f'unjudged/ineligible rather than aborting the run: {exc}'
        )
        classification_call_diagnostics.append({
            'stage': 'etf_judgment_safety_wrapper',
            'success': False,
            'error': str(exc),
        })
        return research_by_symbol


def judge_etf_research_pool(
    client, candidate_records, research_by_symbol, market_context, force=False,
):
    """Use 3.5 Flash as a no-Search judge over validated 2.5 evidence packets."""
    global classification_logical_calls_used, classification_api_attempts_used
    global classification_quota_exhausted
    if classification_quota_exhausted:
        return research_by_symbol
    candidate_by_symbol = {
        str(candidate['Symbol']).upper(): candidate for candidate in candidate_records
    }
    judged = dict(research_by_symbol)
    symbols = [
        str(candidate['Symbol']).upper() for candidate in candidate_records
        if (
            str(candidate['Symbol']).upper() in research_by_symbol
            and not research_by_symbol[str(candidate['Symbol']).upper()].get('judgment_model')
            and etf_optimistic_final_score(candidate) >= minimum_final_selection_score
            and classification_validation_rounds.get(str(candidate['Symbol']).upper(), 0)
            < max_judgment_validation_rounds_per_etf
        )
    ]
    if not symbols:
        return research_by_symbol
    # Stable sort preserves rank within a round, across scheduler invocations.
    symbols.sort(key=lambda symbol: classification_validation_rounds.get(symbol, 0))
    classification_scheduling_diagnostics.append({
        'pending': len(symbols), 'force': force,
        'target': classification_batch_target,
        'action': 'classify' if force or len(symbols) >= classification_batch_target else 'accumulate',
    })
    if not force and len(symbols) < classification_batch_target:
        print(
            f'Accumulating ETF judgment candidates: {len(symbols)}/'
            f'{classification_batch_target}; no 3.5 call yet.'
        )
        return research_by_symbol
    prior_peer_patches = {
        symbol: {
            key: research.get(key) for key in (
                'reversal_risk', 'continuation_strength',
                'benchmark_outperformance_outlook', 'benchmark_outperformance_confidence', 'risk_exposure_group'
            )
        }
        for symbol, research in judged.items()
        if research.get('judgment_model')
    }
    start = 0
    while symbols and (force or len(symbols) >= classification_batch_target):
        batch_symbols = symbols[:classification_batch_soft_max]
        tail = len(symbols) - len(batch_symbols)
        if 0 < tail < classification_min_intermediate_batch:
            keep = max(
                classification_min_intermediate_batch,
                len(batch_symbols) - (classification_min_intermediate_batch - tail),
            )
            batch_symbols = batch_symbols[:keep]
        if classification_logical_calls_used >= max_classification_logical_calls_per_run:
            print('ETF logical classification-call budget exhausted; remaining ETFs stay unjudged and cannot enter the final portfolio.')
            break
        if classification_api_attempts_used >= max_classification_api_attempts_per_run:
            print('ETF classification API-attempt budget exhausted; remaining ETFs stay unjudged and cannot enter the final portfolio.')
            break
        if api_attempt_budget.remaining(classification_model, 'classification') <= 0:
            print('ETF model-family classification budget exhausted; remaining ETFs stay unjudged.')
            break
        classification_logical_calls_used += 1
        compact = []
        for symbol in batch_symbols:
            candidate = candidate_by_symbol[symbol]
            research = judged[symbol]
            compact.append({
                'symbol': symbol,
                'name': candidate.get('Name'),
                'portfolio_group': candidate_sector_group(candidate),
                'qvm_score': candidate.get('QVMScore'),
                'quality_score': candidate.get('QualityScore'),
                'value_score': candidate.get('ValueScore'),
                'momentum_score': candidate.get('MomentumScore'),
                'price_path_quality': candidate.get('PricePathQuality'),
                'quantitative_entry_risk': candidate.get('QuantitativeEntryRisk'),
                'overextension_penalty': candidate.get('OverextensionPenalty'),
                'returns': {period: candidate.get(f'{period} Return') for period in ('1M','3M','6M','9M','1Y')},
                'price_path': {
                    'volatility': candidate.get('Volatility 1Y'),
                    'downside_volatility': candidate.get('DownsideVolatility'),
                    'max_drawdown': candidate.get('MaxDrawdown'),
                    'positive_day_pct': candidate.get('PositiveDayPct'),
                    'trend_r2': candidate.get('TrendR2'),
                    'distance_50dma': candidate.get('Distance50DMA'),
                    'distance_200dma': candidate.get('Distance200DMA'),
                    'distance_52w_high': candidate.get('Distance52WHigh'),
                    'largest_5d_move': candidate.get('Largest5DayMove'),
                    'momentum_acceleration': candidate.get('MomentumAcceleration'),
                },
                'benchmark_excess_returns': {
                    benchmark: {
                        period: candidate.get(f'{period} Excess vs {benchmark}')
                        for period in ('1M','3M','6M','9M','1Y')
                    }
                    for benchmark in benchmark_etfs
                },
                'grounded_research': research,
                'previous_classification': (previous_diagnostics.get('classifications', {}) or {}).get(symbol),
            })
        prompt = (
            config['prompt_etf_judgment'].rstrip()
            + '\n\nCURRENT_DATE_UTC: ' + datetime.now(UTC).date().isoformat()
            + '\n\nCONFIGURED_BENCHMARKS:\n' + json.dumps(benchmark_etfs)
            + '\n\nMARKET_CONTEXT:\n' + json.dumps(market_context, ensure_ascii=False)
            + '\n\nPEER_CLASSIFICATIONS_FROM_EARLIER_BATCHES:\n' + json.dumps(prior_peer_patches, ensure_ascii=False)
            + '\n\nCANDIDATES_WITH_GROUNDED_RESEARCH:\n' + json.dumps(compact, ensure_ascii=False)
        )
        models = [classification_model]
        if classification_fallback_model not in models:
            models.append(classification_fallback_model)
        patches = None
        used_model = None
        last_error = None
        for model_index, model_name in enumerate(models):
            attempts = classification_attempts if model_index == 0 else 1
            for attempt in range(1, attempts + 1):
                if classification_api_attempts_used >= max_classification_api_attempts_per_run:
                    break
                attempt_started = time.perf_counter()
                stage = f'ETF judgment {start + 1}-{start + len(batch_symbols)} ({model_name}, attempt {attempt}/{attempts})'
                try:
                    api_attempt_budget.reserve(
                        model_name, 'classification', stage
                    )
                except GeminiApiAttemptBudgetExhausted as exc:
                    last_error = exc
                    classification_call_diagnostics.append({
                        'stage': 'etf_judgment', 'model': model_name,
                        'attempt': attempt,
                        'logical_call': classification_logical_calls_used,
                        'api_attempt': classification_api_attempts_used,
                        'symbols': list(batch_symbols),
                        'success': False, 'fallback': model_index > 0,
                        'error': str(exc),
                        'status': 'API_BUDGET_EXHAUSTED',
                    })
                    print(f'ETF API-attempt budget stopped {stage}: {exc}')
                    break
                classification_api_attempts_used += 1
                print(
                    f'Gemini ETF classification logical call '
                    f'{classification_logical_calls_used}/{max_classification_logical_calls_per_run}; '
                    f'API attempt {classification_api_attempts_used}/{max_classification_api_attempts_per_run}: {stage}'
                )
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        config=build_gemini_config(
                            classification_thinking_budget,
                            enable_search=False,
                            response_mime_type='application/json',
                            max_output_tokens=classification_max_output_tokens,
                        ),
                        contents=prompt,
                    )
                    metadata = extract_gemini_metadata(response)
                    metadata['response_text_chars'] = len(getattr(response, 'text', '') or '')
                    print_gemini_metadata(stage, metadata)
                    data = parse_json_response(getattr(response, 'text', None))
                    patches = normalize_etf_judgment_results(data)
                    if not isinstance(patches, list):
                        raise ValueError('ETF judgment response lacks a usable results/classifications list.')
                    patch_by_symbol = {
                        str(item.get('symbol') or '').strip().upper(): item
                        for item in patches if isinstance(item, dict)
                        and str(item.get('symbol') or '').strip().upper() in batch_symbols
                    }
                    # Ambiguous duplicate patches are isolated, not last-write-wins.
                    returned_symbols = [str(p.get('symbol') or '').strip().upper()
                                        for p in patches if isinstance(p, dict)]
                    duplicates = {s for s in batch_symbols if returned_symbols.count(s) > 1}
                    for symbol in duplicates:
                        patch_by_symbol.pop(symbol, None)
                    missing = [symbol for symbol in batch_symbols if symbol not in patch_by_symbol]
                    if missing:
                        print(
                            'Warning: ETF judgment omitted symbols; preserving returned peers '
                            'and leaving omitted ETFs unjudged for a later catch-up: '
                            + ', '.join(missing)
                        )
                    if not patch_by_symbol:
                        raise ValueError('ETF judgment returned no usable symbol patches.')
                    used_model = model_name
                    classification_call_diagnostics.append({
                        'stage': 'etf_judgment', 'model': model_name,
                        'attempt': attempt, 'logical_call': classification_logical_calls_used,
                        'api_attempt': classification_api_attempts_used, 'symbols': list(batch_symbols),
                        'success': True, 'fallback': model_index > 0,
                        'returned_symbols': sorted(patch_by_symbol),
                        'missing_symbols': missing, 'duplicate_symbols': sorted(duplicates),
                        'elapsed_seconds': round(time.perf_counter() - attempt_started, 3),
                        'metadata': metadata,
                    })
                    patches = patch_by_symbol
                    break
                except Exception as exc:
                    patches = None
                    last_error = exc
                    classification_call_diagnostics.append({
                        'stage': 'etf_judgment', 'model': model_name,
                        'attempt': attempt, 'logical_call': classification_logical_calls_used,
                        'api_attempt': classification_api_attempts_used, 'symbols': list(batch_symbols),
                        'success': False, 'fallback': model_index > 0,
                        'elapsed_seconds': round(time.perf_counter() - attempt_started, 3),
                        'error': str(exc),
                    })
                    print(f'Warning: {stage} failed: {exc}')
                    # A valid JSON response with the wrong schema is structural, not
                    # transient. Repeating the same model/prompt usually reproduces it,
                    # so move directly to the fallback model instead of burning budget.
                    if is_daily_quota_error(exc):
                        classification_quota_exhausted = True
                        break
                    if isinstance(exc, ValueError) or not is_transient_gemini_error(exc):
                        break
                    if attempt < attempts:
                        time.sleep(min(max_transient_delay, initial_delay * (2 ** (attempt - 1))) + random.uniform(0, 3))
            if patches is not None or classification_quota_exhausted:
                break
        if patches is None:
            print(f'Warning: ETF judgment unavailable for batch; grounded 2.5 research is preserved, but these ETFs remain ineligible for final selection until judged: {last_error}')
            if classification_quota_exhausted:
                break
            # A wholly malformed response must not starve untouched peers.
            # Treat the batch as missing patches and apply the same bounded
            # validation rounds used for partial responses.
            patches = {}
        valid_patch_count = 0
        invalid_patch_symbols = []
        for symbol in batch_symbols:
            classification_validation_rounds[symbol] = classification_validation_rounds.get(symbol, 0) + 1
            patch = patches.get(symbol)
            if patch is None:
                invalid_patch_symbols.append(symbol)
                research = dict(judged[symbol])
                research['judgment_validation_error'] = (
                    f'Judgment response unavailable: {last_error}'
                    if used_model is None else 'Gemini omitted the ETF judgment patch.'
                )
                judged[symbol] = research
                continue
            research = dict(judged[symbol])
            try:
                patch = normalize_etf_judgment_patch(symbol, patch, research)
            except Exception as exc:
                invalid_patch_symbols.append(symbol)
                research['judgment_validation_error'] = str(exc)
                judged[symbol] = research
                classification_call_diagnostics.append({
                    'stage': 'etf_judgment_field_validation',
                    'model': used_model,
                    'logical_call': classification_logical_calls_used,
                    'api_attempt': classification_api_attempts_used,
                    'symbols': [symbol],
                    'success': False,
                    'error': str(exc),
                    'raw_patch': {
                        key: patch.get(key) for key in (
                            'continuation_strength',
                            'benchmark_outperformance_outlook',
                            'benchmark_outperformance_confidence',
                            'exposure_reversal_risk', 'entry_reversal_risk',
                            'reversal_risk', 'holdings_concentration',
                            'construction_concentration',
                        )
                    } if isinstance(patch, dict) else str(patch)[:500],
                })
                print(
                    f'Warning: invalid ETF judgment patch [{symbol}] was isolated '
                    f'without discarding peer judgments: {exc}'
                )
                continue

            exposure_risk = patch['exposure_reversal_risk']
            entry_risk = patch['entry_reversal_risk']
            overall = patch['reversal_risk']
            continuation = patch['continuation_strength']
            benchmark_outlook = patch['benchmark_outperformance_outlook']
            benchmark_confidence = patch['benchmark_outperformance_confidence']
            holdings_conc = patch['holdings_concentration']
            construction_conc = patch['construction_concentration']
            research.pop('judgment_validation_error', None)
            research.update({
                'judgment_model': used_model,
                'exposure_reversal_risk': exposure_risk,
                'entry_reversal_risk': entry_risk,
                'reversal_risk': overall,
                'holdings_concentration': holdings_conc,
                'construction_concentration': construction_conc,
                'benchmark_outperformance_outlook': benchmark_outlook,
                'benchmark_outperformance_confidence': benchmark_confidence,
                'benchmark_outperformance_basis': patch.get('benchmark_outperformance_basis'),
                'continuation_strength': continuation,
                'primary_reversal_channel': patch.get('primary_reversal_channel'),
                'classification_change_reason': patch.get('classification_change_reason'),
                'material_new_evidence': patch.get('material_new_evidence'),
            })
            for field in ('risk_exposure_group', 'primary_risk_event_id', 'explanation'):
                if field in patch:
                    research[field] = patch[field]
            try:
                research = reconcile_etf_risk_event_fields(
                    research,
                    candidate_by_symbol[symbol],
                    market_context.get('active_risk_events') or [],
                )
            except Exception as exc:
                invalid_patch_symbols.append(symbol)
                prior_research = dict(judged[symbol])
                prior_research['judgment_validation_error'] = str(exc)
                judged[symbol] = prior_research
                classification_call_diagnostics.append({
                    'stage': 'etf_judgment_event_reconciliation',
                    'model': used_model,
                    'logical_call': classification_logical_calls_used,
                    'api_attempt': classification_api_attempts_used,
                    'symbols': [symbol],
                    'success': False,
                    'error': str(exc),
                    'raw_primary_risk_event_id': patch.get(
                        'primary_risk_event_id'
                    ),
                })
                print(
                    f'Warning: ETF judgment event identity [{symbol}] was '
                    f'isolated without discarding peer judgments: {exc}'
                )
                continue
            valid_patch_count += 1
            # Reset only judgment-owned exclusion state before applying the new
            # authoritative patch; grounded mandate/research exclusions remain intact.
            prior_reason = str(research.get('eligibility_reason') or '')
            if prior_reason.startswith('Excluded by authoritative ETF judgment') or \
                    prior_reason.startswith('Excluded because authoritative judgment'):
                research['eligible'] = True
                research['eligibility_reason'] = (
                    'Python eligibility passed: authoritative judgment did not '
                    'trigger a hard exclusion.'
                )
            if overall in {'ELEVATED','SEVERE'}:
                research['eligible'] = False
                research['eligibility_reason'] = f'Excluded by authoritative ETF judgment because reversal risk is {overall}.'
            elif benchmark_outlook == 'UNLIKELY' and benchmark_confidence == 'HIGH':
                research['eligible'] = False
                research['eligibility_reason'] = 'Excluded because authoritative judgment finds benchmark outperformance unlikely over 6-12 months with HIGH confidence.'
            judged[symbol] = research
            prior_peer_patches[symbol] = {
                key: research.get(key) for key in (
                    'reversal_risk','continuation_strength',
                    'benchmark_outperformance_outlook','benchmark_outperformance_confidence','risk_exposure_group'
                )
            }
        print(
            f'ETF judgment batch field validation: {valid_patch_count}/{len(batch_symbols)} '
            f'usable; {len(invalid_patch_symbols)} isolated/unjudged.'
        )
        if invalid_patch_symbols:
            print(
                'ETF judgment catch-up required for: '
                + ', '.join(invalid_patch_symbols)
            )
        retryable = [s for s in invalid_patch_symbols
                     if classification_validation_rounds[s] < max_judgment_validation_rounds_per_etf]
        classification_validation_diagnostics.append({
            'logical_call': classification_logical_calls_used,
            'requested': list(batch_symbols), 'validated_count': valid_patch_count,
            'invalid_or_missing_symbols': invalid_patch_symbols,
            'retryable_symbols': retryable,
            'exhausted_symbols': sorted(set(invalid_patch_symbols) - set(retryable)),
        })
        write_run_checkpoint()
        # Classify untouched peers before retrying malformed/omitted patches.
        symbols = symbols[len(batch_symbols):] + retryable
        start += len(batch_symbols)
        if not force and len(symbols) < classification_batch_target:
            break
    return judged


def preview_portfolio(candidates, research_by_symbol, target, max_per_sector, max_moderate_event, require_judgment=False):
    selected, sector_counts, event_counts, driver_counts, family_counts = [], {}, {}, {}, {}
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            etf_selection_merit(
                candidate,
                research_by_symbol.get(str(candidate['Symbol']).upper(), {}),
            )
            if research_by_symbol.get(str(candidate['Symbol']).upper(), {}).get('judgment_model')
            else float(candidate.get('QVMScore') or 0.0)
        ),
        reverse=True,
    )
    for candidate in ranked_candidates:
        if len(selected) >= target:
            break
        symbol = str(candidate['Symbol']).upper()
        research = research_by_symbol.get(symbol)
        if not research or not research.get('eligible', True):
            continue
        if etf_optimistic_final_score(candidate) < minimum_final_selection_score:
            continue
        if require_judgment and not research.get('judgment_model'):
            continue
        if research.get('judgment_model') and not etf_passes_benchmark_override(candidate, research):
            continue
        risk = combined_etf_reversal_risk(research)
        if risk not in SELECTABLE_RISKS:
            continue
        # Apply the calibrated continuation score only after authoritative
        # judgment exists. Provisional 2.5 research can guide backfill order,
        # but it cannot be treated as though missing 3.5 fields were UNCERTAIN.
        if (
            research.get('judgment_model')
            and etf_final_score(candidate, research) + score_comparison_epsilon < etf_required_final_score(research, candidate)
        ):
            continue
        sector = candidate_sector_group(candidate)
        sector_key = sector.casefold()
        family = etf_factor_family(candidate, research)
        if sector_counts.get(sector_key, 0) >= max_per_sector:
            continue
        if family_counts.get(family, 0) >= max_etfs_per_factor_family:
            continue
        event_key = normalized_risk_event_key(research)
        if risk == 'MODERATE' and event_key and event_counts.get(event_key, 0) >= max_moderate_event:
            continue
        driver_key = normalized_return_driver_key(research)
        if driver_key and driver_counts.get(driver_key, 0) >= max_etfs_per_return_driver:
            continue
        selected.append({'candidate': candidate, 'research': research})
        sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        if risk == 'MODERATE' and event_key:
            event_counts[event_key] = event_counts.get(event_key, 0) + 1
        if driver_key:
            driver_counts[driver_key] = driver_counts.get(driver_key, 0) + 1
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
    selected, ledger, sector_counts, event_counts, driver_counts, family_counts = [], [], {}, {}, {}, {}
    research_disposition = research_disposition or {}
    ranked_candidates = sorted(
        list(candidates),
        key=lambda candidate: etf_selection_merit(
            candidate,
            research_by_symbol.get(str(candidate['Symbol']).upper(), {}),
        ) if str(candidate['Symbol']).upper() in research_by_symbol else float(candidate.get('QVMScore') or 0.0),
        reverse=True,
    )
    qvm_rank = {
        str(candidate['Symbol']).upper(): rank
        for rank, candidate in enumerate(candidates, start=1)
    }
    for candidate in ranked_candidates:
        if len(selected) >= target:
            break
        symbol = str(candidate['Symbol']).upper()
        research = research_by_symbol.get(symbol)
        status, reason = None, None
        sector = candidate_sector_group(candidate)
        sector_key = sector.casefold()
        family = etf_factor_family(candidate, research) if research else etf_factor_family(candidate)
        final_score = None
        if not research:
            disposition = research_disposition.get(symbol, 'NOT_NEEDED')
            if disposition == 'QUANTITATIVE_FLOOR':
                status = 'NOT RESEARCHED — CANNOT CLEAR FINAL SCORE FLOOR'
                reason = (
                    f'Even the optimistic maximum continuation score '
                    f'{etf_optimistic_final_score(candidate):.2f} is below the baseline '
                    f'ETF floor {minimum_final_selection_score:.2f}.'
                )
            elif disposition == 'VALIDATION_FAILED':
                status = 'NOT SELECTED — RESEARCH VALIDATION FAILED'
                reason = 'Research was attempted but did not validate.'
            elif disposition == 'SECTOR_CAPACITY':
                status = 'NOT RESEARCHED — SECTOR CAPACITY'
                reason = f'{max_per_sector} higher-ranked ETFs already use {sector}.'
            elif disposition == 'FACTOR_FAMILY_CAPACITY':
                status = 'NOT RESEARCHED — FACTOR-FAMILY CAPACITY'
                reason = (
                    f'{max_etfs_per_factor_family} higher-ranked ETFs already use '
                    f'the {family} factor family.'
                )
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
            risk = combined_etf_reversal_risk(research)
            event_key = normalized_risk_event_key(research)
            driver_key = normalized_return_driver_key(research)
            final_score = etf_final_score(candidate, research)
            benchmark_outlook = str(
                research.get('benchmark_outperformance_outlook') or 'UNCERTAIN'
            ).upper()
            continuation = str(research.get('continuation_strength') or 'ADEQUATE').upper()
            if not research.get('eligible', True):
                status = 'NOT SELECTED — INELIGIBLE'
                reason = research.get('eligibility_reason') or research.get('explanation')
            elif risk not in SELECTABLE_RISKS:
                status = f'NOT SELECTED — {risk}'
                reason = research.get('explanation')
            elif etf_optimistic_final_score(candidate) < minimum_final_selection_score:
                status = 'NOT SELECTED — CANNOT CLEAR FINAL SCORE FLOOR'
                reason = (
                    f'Even the optimistic maximum continuation score '
                    f'{etf_optimistic_final_score(candidate):.2f} is below the baseline '
                    f'ETF floor {minimum_final_selection_score:.2f}.'
                )
            elif not research.get('judgment_model'):
                status = 'NOT SELECTED — AUTHORITATIVE JUDGMENT REQUIRED'
                reason = (
                    'Grounded 2.5 research validated, but no authoritative no-Search '
                    'judgment completed before the classification budget ended.'
                )
            elif not etf_passes_benchmark_override(candidate, research):
                status = 'NOT SELECTED — BENCHMARK OVERRIDE NOT EARNED'
                reason = (
                    f'QVM {float(candidate.get("QVMScore") or 0.0):.2f} is materially below '
                    f'the weaker configured benchmark threshold {benchmark_qvm_floor:.2f}; '
                    'selection therefore requires authoritative benchmark=LIKELY and '
                    'continuation=STRONG.'
                )
            elif final_score + score_comparison_epsilon < etf_required_final_score(research, candidate):
                required_score = etf_required_final_score(research, candidate)
                status = 'NOT SELECTED — FINAL SCORE BELOW CALIBRATED MINIMUM'
                reason = (
                    f'Final continuation score {final_score:.2f} is below the '
                    f'calibrated ETF floor {required_score:.2f}; '
                    f'continuation={continuation}, benchmark={benchmark_outlook}, confidence={str(research.get("benchmark_outperformance_confidence") or "MEDIUM").upper()}, risk={risk}.'
                )
            elif sector_counts.get(sector_key, 0) >= max_per_sector:
                status = 'SKIPPED — SECTOR CAPACITY'
                reason = f'{max_per_sector} selected ETFs already use the {sector} sector/category.'
            elif family_counts.get(family, 0) >= max_etfs_per_factor_family:
                status = 'SKIPPED — FACTOR-FAMILY CAPACITY'
                reason = (
                    f'{max_etfs_per_factor_family} selected ETFs already use the {family} '
                    'factor family; avoid hidden concentration across differently named categories.'
                )
            elif (
                risk == 'MODERATE' and event_key
                and event_counts.get(event_key, 0) >= max_moderate_event
            ):
                status = 'SKIPPED — RISK-EVENT CAPACITY'
                reason = f'The {event_key[0]} / {event_key[1]} group is already full.'
            elif driver_key and driver_counts.get(driver_key, 0) >= max_etfs_per_return_driver:
                status = 'SKIPPED — RETURN-DRIVER CAPACITY'
                reason = f'{max_etfs_per_return_driver} selected ETFs already depend on {driver_key}.'
            else:
                selected.append({'candidate': candidate, 'research': research})
                sector_counts[sector_key] = sector_counts.get(sector_key, 0) + 1
                family_counts[family] = family_counts.get(family, 0) + 1
                if risk == 'MODERATE' and event_key:
                    event_counts[event_key] = event_counts.get(event_key, 0) + 1
                if driver_key:
                    driver_counts[driver_key] = driver_counts.get(driver_key, 0) + 1
                status = f'SELECTED — {risk}'
                reason = research.get('explanation')
        ledger.append({
            'qvm_rank': qvm_rank.get(symbol),
            'symbol': symbol,
            'sector_group': sector,
            'status': status,
            'final_selection_score': final_score,
            'continuation_strength': research.get('continuation_strength') if research else None,
            'benchmark_outperformance_outlook': research.get('benchmark_outperformance_outlook') if research else None,
            'return_driver_group': normalized_return_driver_key(research) if research else None,
            'factor_family': family,
            'selection_merit_score': (
                etf_selection_merit(candidate, research)
                if research and research.get('judgment_model') else None
            ),
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
    sorted_selected = sorted(
        selected,
        key=lambda item: float(item['candidate'].get('QVMScore') or 0.0),
        reverse=True,
    )
    for item in sorted_selected:
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
            'exposure_reversal_risk': research.get('exposure_reversal_risk'),
            'entry_reversal_risk': research.get('entry_reversal_risk'),
            'continuation_strength': research.get('continuation_strength'),
            'benchmark_outperformance_outlook': research.get('benchmark_outperformance_outlook'),
            'benchmark_outperformance_confidence': research.get('benchmark_outperformance_confidence'),
            'benchmark_outperformance_basis': research.get('benchmark_outperformance_basis'),
            'holdings_concentration': research.get('holdings_concentration'),
            'construction_concentration': research.get('construction_concentration'),
            'primary_reversal_channel': research.get('primary_reversal_channel'),
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
signal.signal(signal.SIGALRM, handle_total_runtime_timeout)
signal.alarm(TOTAL_RUNTIME_TIMEOUT_SECONDS)
config_path = SCRIPT_DIR / 'etf_config.yml'
with config_path.open('r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
YF_CACHE_EXPIRY_DAYS = max(1, int(config.get('etf_yf_cache_expiry_days', 7)))
FUND_CACHE_EXPIRY_DAYS = max(1, int(config.get('etf_fund_cache_expiry_days', 3)))
required_prompt_keys = {
    'prompt_market_context', 'prompt_etf_batch', 'prompt_etf_judgment',
    'prompt_html_summary'
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
classification_model = str(config.get('classification_model', 'gemini-3.5-flash'))
classification_fallback_model = str(config.get('classification_fallback_model', classification_model))
summary_model = str(config.get('summary_model', classification_model))
summary_fallback_model = str(config.get('summary_fallback_model', model_primary))
if any('3.5' not in model for model in (classification_model, classification_fallback_model, summary_model)):
    raise ValueError('Classification and primary summary models must use the separately budgeted Gemini 3.5 family.')
if any('2.5' not in model for model in (model_primary, model_fallback, summary_fallback_model)):
    raise ValueError('Research/context and summary fallback models must use the separately budgeted Gemini 2.5 family.')
classification_thinking_budget = int(config.get('classification_thinking_budget', 8192))
classification_max_output_tokens = int(config.get('classification_max_output_tokens', 65536))
classification_attempts = max(1, int(config.get('classification_attempts', 2)))
max_classification_logical_calls_per_run = max(1, int(config.get('max_classification_logical_calls_per_run', config.get('max_classification_calls_per_run', 8))))
max_classification_api_attempts_per_run = max(max_classification_logical_calls_per_run, int(config.get('max_classification_api_attempts_per_run', max_classification_logical_calls_per_run * 2 + 2)))
max_3_5_api_calls_per_run = int(config.get(
    'max_3_5_api_calls_per_run', config.get('max_gemini_35_calls_per_run', 7)
))
max_3_5_classification_calls_per_run = int(config.get(
    'max_3_5_classification_calls_per_run',
    max_classification_api_attempts_per_run,
))
max_3_5_summary_calls_per_run = int(config.get(
    'max_3_5_summary_calls_per_run', 1
))
max_gemini_35_calls_per_run = max_3_5_api_calls_per_run
if (
    max_3_5_classification_calls_per_run
    != max_classification_api_attempts_per_run
):
    raise ValueError(
        'max_3_5_classification_calls_per_run must equal '
        'max_classification_api_attempts_per_run.'
    )
if (
    max_3_5_classification_calls_per_run + max_3_5_summary_calls_per_run
    > max_3_5_api_calls_per_run
):
    raise ValueError(
        'Gemini 3.5 category limits exceed its total API-call limit.'
    )
classification_batch_target = max(1, int(config.get('classification_batch_target', 25)))
classification_batch_soft_max = max(classification_batch_target, int(config.get('classification_batch_soft_max', 30)))
classification_min_intermediate_batch = max(1, int(config.get('classification_min_intermediate_batch', 15)))
max_judgment_validation_rounds_per_etf = max(1, int(config.get('max_judgment_validation_rounds_per_etf', 2)))
thinking_budget = config.get('thinking_budget', 12288)
summary_thinking_budget = config.get('summary_thinking_budget', 4096)
gemini_max_output_tokens = config.get('gemini_max_output_tokens', 49152)
gemini_batch_size = config.get('gemini_batch_size', 15)
target_selected_etfs = config.get('target_selected_etfs', 10)
max_etfs_per_sector_group = config.get('max_etfs_per_sector_group', 2)
max_etfs_per_return_driver = max(1, int(config.get('max_etfs_per_return_driver', 2)))
max_etfs_per_factor_family = max(1, int(config.get('max_etfs_per_factor_family', 3)))
challenger_research_enabled = bool(config.get('challenger_research_enabled', True))
max_challenger_candidates_per_run = max(0, int(config.get('max_challenger_candidates_per_run', 4)))
challenger_min_qvm = float(config.get('challenger_min_qvm', 60.0))
challenger_min_relative_score = float(config.get('challenger_min_relative_score', 90.0))
challenger_relative_bonus_start = float(config.get('challenger_relative_bonus_start', 80.0))
challenger_relative_bonus_per_point = float(config.get('challenger_relative_bonus_per_point', 0.15))
challenger_relative_bonus_cap = float(config.get('challenger_relative_bonus_cap', 3.0))
challenger_score_epsilon = float(config.get('challenger_score_epsilon', 0.25))
replacement_merit_epsilon = float(config.get('replacement_merit_epsilon', 0.25))
replacement_diagnostics_top_n = max(1, int(config.get('replacement_diagnostics_top_n', 12)))
selection_relative_good_threshold = float(config.get('selection_relative_good_threshold', 80.0))
selection_relative_high_threshold = float(config.get('selection_relative_high_threshold', 90.0))
selection_relative_good_bonus = float(config.get('selection_relative_good_bonus', 1.0))
selection_relative_high_bonus = float(config.get('selection_relative_high_bonus', 2.0))
selection_likely_bonus = float(config.get('selection_likely_bonus', 2.0))
selection_strong_bonus = float(config.get('selection_strong_bonus', 1.0))
minimum_final_selection_score = float(config.get('minimum_final_selection_score', 60.0))
minimum_uncertain_selection_score = float(config.get('minimum_uncertain_selection_score', 66.0))
minimum_moderate_selection_score = float(config.get('minimum_moderate_selection_score', 66.0))
score_comparison_epsilon = max(0.0, float(config.get('score_comparison_epsilon', 0.25)))
benchmark_qvm_override_margin = float(config.get('benchmark_qvm_override_margin', 3.0))
require_likely_strong_below_benchmark_qvm = bool(config.get('require_likely_strong_below_benchmark_qvm', True))
benchmark_qvm_floor = None
benchmark_likely_bonus = float(config.get('benchmark_likely_bonus', 5.0))
benchmark_uncertain_penalty = float(config.get('benchmark_uncertain_penalty', -4.0))
benchmark_unlikely_medium_penalty = float(config.get('benchmark_unlikely_medium_penalty', -8.0))
benchmark_unlikely_low_penalty = float(config.get('benchmark_unlikely_low_penalty', -4.0))
benchmark_relative_rescue_enabled = bool(config.get('benchmark_relative_rescue_enabled', True))
benchmark_relative_rescue_min_qvm = float(config.get('benchmark_relative_rescue_min_qvm', 66.0))
benchmark_relative_rescue_min_relative_score = float(config.get('benchmark_relative_rescue_min_relative_score', 90.0))
benchmark_relative_rescue_bonus = float(config.get('benchmark_relative_rescue_bonus', 1.0))
benchmark_relative_rescue_floor = float(config.get('benchmark_relative_rescue_floor', 60.0))
concentration_adjustments = {
    'LOW': float(config.get('concentration_low_adjustment', 0.0)),
    'MODERATE': float(config.get('concentration_moderate_adjustment', 0.0)),
    'HIGH': float(config.get('concentration_high_adjustment', -1.0)),
}
max_concentration_penalty = float(config.get('max_concentration_penalty', -2.0))
continuation_adjustments = {
    'STRONG': float(config.get('continuation_strong_bonus', 4.0)),
    'ADEQUATE': float(config.get('continuation_adequate_bonus', 0.0)),
    'WEAK': float(config.get('continuation_weak_penalty', -4.0)),
}
repeated_return_driver_penalty = float(config.get('repeated_return_driver_penalty', -3.0))
max_candidates_per_provisional_group = config.get(
    'max_candidates_per_provisional_group', 4
)
max_moderate_per_risk_event = config.get('max_moderate_per_risk_event', 2)
max_gemini_calls_per_run = config.get('max_gemini_calls_per_run', 7)
max_etf_research_calls_per_run = config.get('max_etf_research_calls_per_run', 5)
max_gemini_api_attempts_per_run = int(
    config.get('max_gemini_api_attempts_per_run', max_gemini_calls_per_run)
)
max_2_5_api_calls_per_run = int(config.get(
    'max_2_5_api_calls_per_run', max_gemini_api_attempts_per_run
))
max_2_5_market_calls_per_run = int(config.get(
    'max_2_5_market_calls_per_run', 2
))
max_2_5_research_calls_per_run = int(config.get(
    'max_2_5_research_calls_per_run', max_etf_research_calls_per_run
))
max_2_5_summary_fallback_calls_per_run = int(config.get(
    'max_2_5_summary_fallback_calls_per_run', 1
))
if max_gemini_api_attempts_per_run != max_2_5_api_calls_per_run:
    raise ValueError(
        'max_gemini_api_attempts_per_run must equal '
        'max_2_5_api_calls_per_run.'
    )
for category_name, category_limit in (
    ('market', max_2_5_market_calls_per_run),
    ('research', max_2_5_research_calls_per_run),
    ('summary', max_2_5_summary_fallback_calls_per_run),
):
    if category_limit > max_2_5_api_calls_per_run:
        raise ValueError(
            f'Gemini 2.5 {category_name} limit exceeds the shared total limit.'
        )
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
    'market_context_cache_file',
    'caches/gemini_shared_market_context_cache.json',
)
research_cache_hours = config.get('gemini_research_cache_hours', 30)
market_context_cache_hours = config.get('market_context_cache_hours', 12)
cache_version = config.get('cache_version', 1)
research_cache_version = config.get(
    'etf_research_cache_version', cache_version
)
run_report_file = config.get('run_diagnostics_file', 'run_reports/etf_run_report.json')
previous_diagnostics = load_json_object(run_report_file)
RUN_PROVENANCE.update({
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
})
set_run_stage('universe_and_quantitative_data')
print(f'Run report: {Path(run_report_file).resolve()}')
print(f'Run log: {ETF_RUN_LOG_FILE.resolve()}')
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
qvm_weights = config.get('qvm_weights', {'Quality': 0.35, 'Value': 0.10, 'Momentum': 0.55})
min_quality = config.get('min_quality', 35)
top_etfs = load_top_qvm_cache(top_n, research_pool_limit)
cache_diagnostics['top_qvm'] = {
    'hit': top_etfs is not None, 'path': TOP_QVM_CACHE_FILE,
    'ttl_hours': TOP_QVM_CACHE_EXPIRY_HOURS,
}
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

benchmark_qvm_values = [
    float(value)
    for value in top_etfs.loc[top_etfs['Role'] == 'BENCHMARK', 'QVMScore'].tolist()
    if pd.notna(value)
]
if benchmark_qvm_values:
    benchmark_qvm_floor = min(benchmark_qvm_values) - benchmark_qvm_override_margin
    print(
        f'Benchmark-QVM override floor: {benchmark_qvm_floor:.2f} '
        f'(weaker benchmark QVM {min(benchmark_qvm_values):.2f} minus '
        f'{benchmark_qvm_override_margin:.2f} margin).'
    )

gemini_columns = [
    col
    for col in [
        'Role', 'Symbol', 'Name', 'Category', 'LegalType', 'FundFamily', 'QVMScore',
        'ResearchAdmission',
        'QualityScore', 'ValueScore', 'MomentumScore',
        'BenchmarkRelativeScore', 'AUM', 'ExpenseRatio', 'AverageVolume',
        'Volatility 1Y', 'DownsideVolatility', 'MaxDrawdown', 'PositiveDayPct',
        'TrendR2', 'Distance50DMA', 'Distance200DMA', 'Distance52WHigh',
        'Largest5DayMove', 'MomentumAcceleration', 'PricePathQuality',
        'QuantitativeEntryRisk', 'OverextensionPenalty', 'TrailingPE', 'PriceToBook', '1M Return',
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
    max_api_attempts=max_gemini_api_attempts_per_run,
)
api_attempt_budget = GeminiApiAttemptBudget(
    family_limits={
        '3.5': max_3_5_api_calls_per_run,
        '2.5': max_2_5_api_calls_per_run,
    },
    category_limits={
        ('3.5', 'classification'): max_3_5_classification_calls_per_run,
        ('3.5', 'summary'): max_3_5_summary_calls_per_run,
        ('2.5', 'market'): max_2_5_market_calls_per_run,
        ('2.5', 'research'): max_2_5_research_calls_per_run,
        ('2.5', 'summary'): max_2_5_summary_fallback_calls_per_run,
    },
)
call_diagnostics = []
batch_research_diagnostics = []
models_used = []

# The stock and ETF runs share the exact prompt/schema/model contract.
# The current date is supplied to Gemini but is not part of the cache key.
set_run_stage('market_context')
market_prompt_template = config['prompt_market_context'].strip()
market_cache, market_context, market_cache_diag = read_shared_market_cache(
    market_context_cache_file, market_prompt_template, model_primary,
    market_context_cache_hours,
)
cache_diagnostics['market'] = market_cache_diag
market_contract_hash = market_cache_diag['contract_hash']
print('MARKET CACHE ' + ('HIT ' if market_context is not None else 'MISS ')
      + json.dumps(market_cache_diag))
if market_context is None:
    market_prompt = market_prompt_template + (
        f'\n\nCURRENT_DATE_UTC: {datetime.now(UTC).date().isoformat()}\n'
    )
    data, used_model, metadata = call_gemini_json(
        client, model_primary, market_prompt,
        build_gemini_config(
            thinking_budget, enable_search=True, response_mime_type=None,
            max_output_tokens=gemini_max_output_tokens,
        ),
        'ETF market context', request_budget, 'context',
        max_transient_api_attempts, initial_delay, max_transient_delay,
    )
    if not metadata.get('search_queries') and metadata.get('tool_tokens', 0) <= 0:
        raise RuntimeError('Gemini returned ungrounded ETF market context.')
    market_context = validate_current_market_context(data)
    models_used.append(used_model)
    call_diagnostics.append({'stage': 'market context', **metadata})
    if len(metadata.get('search_queries', [])) > 8:
        print(f'Market-context search overrun: {len(metadata["search_queries"])} exposed; prompt maximum is 8.')
    now_iso = datetime.now(UTC).isoformat()
    market_cache['entries'][market_contract_hash] = {
        'created_at': now_iso, 'contract_hash': market_contract_hash,
        'prompt_hash': stable_json_hash(market_prompt_template),
        'schema_hash': market_cache_diag['schema_hash'],
        'model': used_model, 'data': market_context,
        'market_context': market_context,
        'research_metadata': {
            'search_queries': metadata.get('search_queries', []),
            'tool_tokens': metadata.get('tool_tokens', 0),
        },
    }
    market_cache['file_format_version'] = 1
    market_cache_diag['save_succeeded'] = save_json_object_atomic(
        market_context_cache_file, market_cache
    )
    print('MARKET CACHE SAVE ' + json.dumps({
        'contract': market_contract_hash, 'succeeded': market_cache_diag['save_succeeded'],
    }))
set_run_stage('research_cache')

market_context_hash = stable_json_hash(market_context)
active_risk_events = market_context.get('active_risk_events') or []

research_prompt_hash = stable_json_hash(
    {'version': research_cache_version, 'prompt': config['prompt_etf_batch']}
)
research_contract_hash = stable_json_hash({
    'prompt_hash': research_prompt_hash, 'model': model_primary,
    'minimum_sources': config.get('min_etf_sources', 2),
    'maximum_sources': config.get('max_etf_sources', 5),
})
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
for bucket in ('entries', 'deferred_entries'):
    entries = research_cache.get(bucket)
    research_cache[bucket] = entries if isinstance(entries, dict) else {}
research_cache_diag = {
    'path': research_cache_file, 'ttl_hours': research_cache_hours,
    'contract_hash': research_contract_hash, 'hits': 0, 'misses': 0,
    'symbols': {}, 'save_succeeded': None,
}
cache_diagnostics['etf_research'] = research_cache_diag

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
            'BenchmarkRelativeScore', 'PricePathQuality',
            'QuantitativeEntryRisk', 'OverextensionPenalty',
            'ExpenseRatio', 'AUM', '1M Return', '3M Return', '6M Return',
            '9M Return', '1Y Return', 'FundOverview', 'AssetClasses',
            'SectorWeightings', 'TopHoldings', 'CanonicalSector',
            'StyleCategory', 'PortfolioGroup',
        )
    }
    cache_key = stable_json_hash(
        {
            'prompt_hash': research_prompt_hash,
            'market_context_hash': market_context_hash,
            'candidate': signature,
        }
    )
    cache_keys[symbol] = cache_key
    entry = research_cache['entries'].get(cache_key)
    if not isinstance(entry, dict):
        entry = None
    compatible_cache = False
    if entry and (
        entry.get('research_contract_hash', research_contract_hash) != research_contract_hash
        or entry.get('market_context_hash') != market_context_hash
    ):
        entry = None
    if not entry:
        matching_entries = [
            prior_entry
            for prior_entry in research_cache['entries'].values()
            if (
                isinstance(prior_entry, dict)
                and prior_entry.get('research_contract_hash') == research_contract_hash
                and prior_entry.get('fund_identity') == {
                    field: candidate.get(field) for field in ('Name', 'Category', 'LegalType')
                }
                and cache_entry_is_fresh(prior_entry, research_cache_hours)
                and isinstance(prior_entry.get('research'), dict)
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
    cache_result = {
        'hit': False, 'reason': 'missing_or_incompatible',
        'age_hours': cache_age_hours(entry.get('timestamp')) if isinstance(entry, dict) else None,
    }
    research_cache_diag['symbols'][symbol] = cache_result
    if entry and cache_entry_is_fresh(entry, research_cache_hours):
        try:
            cached_research = validate_etf_result(
                entry.get('research'), candidate,
                config.get('min_etf_sources', 2),
                config.get('max_etf_sources', 5),
                active_risk_events,
            )
            # Evidence is reusable, yesterday's authoritative decision is not.
            cached_research.pop('judgment_model', None)
            cached_research.pop('judgment_validation_error', None)
            cached_research['evidence_collected_at'] = entry.get('timestamp')
            cached_research['evidence_market_context_changed'] = (
                entry.get('market_context_hash') != market_context_hash
            )
            research_by_symbol[symbol] = cached_research
            # Preserve evidence age when a compatible older key is reused.
            # Consistency repairs look up this run's key for provenance.
            research_cache['entries'][cache_key] = dict(entry)
            cache_result.update(
                hit=True, reason='compatible' if compatible_cache else 'exact',
                market_context_changed=entry.get('market_context_hash') != market_context_hash,
                evidence_timestamp=entry.get('timestamp'),
            )
            cache_label = 'compatible validated' if compatible_cache else 'validated'
            print(f'Using {cache_label} ETF research cache for {symbol}.')
        except Exception as exc:
            cache_result.update(reason='validation_failed', error=str(exc))
            print(f'Ignoring invalid ETF research cache for {symbol}: {exc}')
    elif entry:
        cache_result['reason'] = 'expired_or_invalid_timestamp'
    research_cache_diag['hits' if cache_result['hit'] else 'misses'] += 1

print('ETF RESEARCH CACHE ' + json.dumps({
    k: v for k, v in research_cache_diag.items() if k != 'symbols'
}))

cursor = 0
pending_retries = []
deferred_excess_candidates = []
deferred_excess_symbols_this_run = set()
sector_capacity_skipped_symbols_this_run = set()
factor_family_capacity_skipped_symbols_this_run = set()
quantitative_floor_skipped_symbols_this_run = set()
research_attempts_by_symbol = {}
research_requests_by_symbol = {}
last_research_errors = {}
structural_repairs_by_symbol = {}
consistency_reviewed_symbols = set()
researched_this_run = set()
search_attempts = 0
charged_search_attempts = 0
research_quota_exhausted = False
transport_failed_symbols_this_run = set()
research_budget_exhausted = False
judgment_reopened_slots = 0
authoritative_backfill_batches = 0
provisional_portfolio_peak = 0
challenger_queued_symbols = set()
challenger_researched_symbols = set()
challenger_judged_symbols = set()
capacity_challenger_tests = {}
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
    research_cache_diag['save_succeeded'] = save_json_object_atomic(research_cache_file, research_cache)
    selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
    )

# Any validated cache entries must pass the same authoritative judgment stage
# before they are allowed to stop fresh research. The judge only processes
# unjudged symbols, so this does not repeatedly rewrite earlier decisions.
initial_provisional_count = len(selected)
provisional_portfolio_peak = max(provisional_portfolio_peak, initial_provisional_count)
if research_by_symbol:
    research_by_symbol = safe_judge_etf_research_pool(
        client, candidate_records, research_by_symbol, market_context
    )
    for call in classification_call_diagnostics:
        if call.get('success') and call.get('model') and call['model'] not in models_used:
            models_used.append(call['model'])
    selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
        require_judgment=True,
    )
    print(
        f'Cached evidence preview: provisional={initial_provisional_count}, '
        f'authoritative={len(selected)}; unjudged evidence is not a reopened slot.'
    )
if len(selected) >= target_selected_etfs:
    queue_portfolio_challengers(
        candidate_records, research_by_symbol, selected, pending_retries,
        challenger_queued_symbols,
    )
if len(selected) >= target_selected_etfs and not pending_retries:
    print('Validated and judged ETF research cache already supports the full portfolio after challenger review.')

request_budget.release_summary_for_research = (
    summary_reservation_releasable_on_shortfall
    and len(selected) < target_selected_etfs
)
while (
    (len(selected) < target_selected_etfs or bool(pending_retries))
    and request_budget.can_reserve('research')
    and api_attempt_budget.remaining(model_primary, 'research') > 0
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

    # Build every research call from the best currently actionable pool rather
    # than continuing a stale rank cursor after authoritative judgment changes
    # the portfolio. Repairs retain first priority because their evidence has
    # already been paid for. New candidates are ordered by optimistic final
    # merit, with a small tie-break bonus for genuinely open sector/family slots.
    open_slots = max(0, target_selected_etfs - len(selected))
    research_batch_limit = gemini_batch_size
    if len(selected) >= target_selected_etfs - 2:
        research_batch_limit = min(
            gemini_batch_size, max(6, open_slots * 3)
        )
    elif len(selected) >= 6:
        research_batch_limit = min(
            gemini_batch_size, max(8, open_slots * 3)
        )
    selected_sectors = {
        candidate_sector_group(item['candidate']).casefold() for item in selected
    }
    selected_families = {
        etf_factor_family(item['candidate'], item.get('research')) for item in selected
    }

    def scheduling_priority(candidate):
        sector_open = candidate_sector_group(candidate).casefold() not in selected_sectors
        family_open = etf_factor_family(candidate) not in selected_families
        bonus = 2.0 if sector_open and family_open else 0.75 if sector_open else 0.0
        return (
            etf_optimistic_final_score(candidate) + bonus,
            quantitative_challenger_score(candidate),
            -rank_by_symbol[str(candidate['Symbol']).upper()],
        )

    best_unresearched = []
    if len(selected) < target_selected_etfs:
        for candidate in candidate_records:
            symbol = str(candidate['Symbol']).upper()
            if symbol in research_by_symbol:
                continue
            if research_attempts_by_symbol.get(symbol, 0) >= max_fresh_research_attempts_per_etf:
                continue
            if etf_optimistic_final_score(candidate) + score_comparison_epsilon < minimum_final_selection_score:
                quantitative_floor_skipped_symbols_this_run.add(symbol)
                continue
            if lower_ranked_candidate_blocked_by_sector_capacity(
                candidate, selected, rank_by_symbol, max_etfs_per_sector_group
            ):
                sector_capacity_skipped_symbols_this_run.add(symbol)
                continue
            if lower_ranked_candidate_blocked_by_factor_family_capacity(candidate, selected):
                factor_family_capacity_skipped_symbols_this_run.add(symbol)
                continue
            best_unresearched.append(candidate)
    best_unresearched.sort(key=scheduling_priority, reverse=True)
    batch_items, pending_retries, queue_stats = assemble_research_batch(
        pending_retries, best_unresearched, research_batch_limit,
        research_attempts_by_symbol, structural_repairs_by_symbol,
        max_fresh_research_attempts_per_etf, max_structural_repairs_per_etf,
        transport_retry_batch_fraction,
    )
    batch = [item['candidate'] for item in batch_items]
    dedicated_basic_repair_batch = False
    fresh_symbols = [
        str(item['candidate']['Symbol']).upper() for item in batch_items
        if item.get('needs_research')
    ]
    repair_only_symbols = {
        str(item['candidate']['Symbol']).upper() for item in batch_items
        if not item.get('needs_research')
    }
    retry_fresh_symbols = {
        symbol for symbol in fresh_symbols if research_requests_by_symbol.get(symbol, 0) > 0
    }
    first_time_symbols = set(fresh_symbols) - retry_fresh_symbols
    transport_retry_symbols = {
        str(item['candidate']['Symbol']).upper() for item in batch_items
        if item.get('transport_failure')
    }
    queue_stats.update({
        'first_time_symbols': sorted(first_time_symbols),
        'fresh_retry_symbols': sorted(retry_fresh_symbols),
        'repair_only_symbols': sorted(repair_only_symbols),
        'selected_before': len(selected),
        'pending_judgment': sum(not r.get('judgment_model') for r in research_by_symbol.values()),
    })
    print('ETF RESEARCH QUEUE ' + json.dumps(queue_stats))
    if not batch:
        judged_before_tail = sum(bool(r.get('judgment_model')) for r in research_by_symbol.values())
        set_run_stage('classification_empty_queue_tail')
        research_by_symbol = safe_judge_etf_research_pool(
            client, candidate_records, research_by_symbol, market_context, force=True,
        )
        selected = preview_portfolio(
            candidate_records, research_by_symbol, target_selected_etfs,
            max_etfs_per_sector_group, max_moderate_per_risk_event, require_judgment=True,
        )
        if sum(bool(r.get('judgment_model')) for r in research_by_symbol.values()) > judged_before_tail:
            # A newly judged tail may reopen sector/factor capacity. Re-scan it.
            continue
        print('No research candidates remain within current eligibility and per-ETF attempt limits.')
        break
    for candidate in batch:
        record_capacity_challenger(candidate, selected, capacity_challenger_tests)
        symbol = str(candidate['Symbol']).upper()
        research_requests_by_symbol[symbol] = research_requests_by_symbol.get(symbol, 0) + 1
    # A small repair tail is useful even if it cannot fill every open slot.
    # It is sent only after all actionable fresh alternatives have been considered.
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
    set_run_stage(f'research_batch_{request_budget.research_used + 1}')
    batch_started = time.perf_counter()
    search_attempts += len(fresh_symbols)
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
            active_risk_events,
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
                last_research_errors.pop(symbol, None)
                valid[symbol].pop('judgment_model', None)
                valid[symbol].pop('judgment_validation_error', None)
                classification_validation_rounds.pop(symbol, None)
                research_by_symbol[symbol] = valid[symbol]
                prior_evidence = research_cache['deferred_entries'].get(cache_keys[symbol], {})
                evidence_timestamp = (
                    datetime.now(UTC).isoformat() if symbol in fresh_symbols else
                    prior_evidence.get('evidence_timestamp') or prior_evidence.get('timestamp')
                    or datetime.now(UTC).isoformat()
                )
                valid[symbol]['evidence_collected_at'] = evidence_timestamp
                valid[symbol]['evidence_market_context_changed'] = False
                research_cache['entries'][cache_keys[symbol]] = {
                    'timestamp': evidence_timestamp,
                    'research_contract_hash': research_contract_hash,
                    'fund_identity': {
                        field: candidate.get(field) for field in ('Name', 'Category', 'LegalType')
                    },
                    'market_context_hash': market_context_hash,
                    'research': valid[symbol],
                }
                research_cache['deferred_entries'].pop(cache_keys[symbol], None)
                print(f'Validated ETF research [{symbol}]: {len(valid[symbol]["sources"])} distinct URLs')
                if symbol in challenger_queued_symbols:
                    challenger_researched_symbols.add(symbol)
            else:
                error = errors.get(symbol, 'Invalid ETF research.')
                last_research_errors[symbol] = error
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
                    'evidence_timestamp': (
                        datetime.now(UTC).isoformat() if symbol in fresh_symbols else
                        prior_deferred.get('evidence_timestamp') or prior_deferred.get('timestamp')
                    ),
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
            'queue': queue_stats,
            'elapsed_seconds': round(time.perf_counter() - batch_started, 3),
            'first_time_research_requested': len(first_time_symbols),
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
            'validation_errors': dict(errors),
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
        research_cache_diag['save_succeeded'] = save_json_object_atomic(research_cache_file, research_cache)
    except Exception as exc:
        print(f'ETF batch failed without validated output: {exc}')
        research_quota_exhausted = is_daily_quota_error(exc)
        response_level_failure = (
            not research_quota_exhausted and is_transient_gemini_error(exc)
        )
        batch_research_diagnostics.append({
            'stage': stage,
            'queue': queue_stats,
            'elapsed_seconds': round(time.perf_counter() - batch_started, 3),
            'first_time_research_requested': len(first_time_symbols),
            'requested_symbols': [str(x['Symbol']).upper() for x in batch],
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
            last_research_errors[symbol] = str(exc)
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

    provisional_selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
    )
    provisional_count = len(provisional_selected)
    provisional_portfolio_peak = max(provisional_portfolio_peak, provisional_count)
    consistency_conflicts = find_actionable_consistency_conflicts(
        candidate_records, research_by_symbol, provisional_selected
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
        research_cache_diag['save_succeeded'] = save_json_object_atomic(research_cache_file, research_cache)
        provisional_selected = preview_portfolio(
            candidate_records,
            research_by_symbol,
            target_selected_etfs,
            max_etfs_per_sector_group,
            max_moderate_per_risk_event,
        )
        provisional_count = len(provisional_selected)
        provisional_portfolio_peak = max(provisional_portfolio_peak, provisional_count)

    judged_before = sum(
        1 for research in research_by_symbol.values() if research.get('judgment_model')
    )
    authoritative_count_before_judgment = len(selected)
    pending_judgment_count = sum(
        1 for candidate in candidate_records
        if (
            str(candidate['Symbol']).upper() in research_by_symbol
            and not research_by_symbol[str(candidate['Symbol']).upper()].get('judgment_model')
            and etf_optimistic_final_score(candidate) >= minimum_final_selection_score
        )
    )
    remaining_research_calls = max(0, min(
        request_budget.research_limit - request_budget.research_used,
        request_budget.max_api_attempts - request_budget.api_attempts,
        api_attempt_budget.remaining(model_primary, 'research'),
    ))
    force_judgment = (
        pending_judgment_count >= classification_min_intermediate_batch
        and len(selected) + pending_judgment_count >= target_selected_etfs
    ) or (
        pending_judgment_count > 0 and remaining_research_calls <= 1
    ) or not request_budget.can_reserve('research')
    set_run_stage('classification_after_research')
    print('ETF JUDGMENT SCHEDULER ' + json.dumps({
        'pending': pending_judgment_count, 'force': force_judgment,
        'batch_target': classification_batch_target,
        'intermediate_minimum': classification_min_intermediate_batch,
        'remaining_research_calls': remaining_research_calls,
    }))
    research_by_symbol = safe_judge_etf_research_pool(
        client, candidate_records, research_by_symbol, market_context,
        force=force_judgment,
    )
    judged_after = sum(
        1 for research in research_by_symbol.values() if research.get('judgment_model')
    )
    challenger_judged_symbols.update(
        symbol for symbol in challenger_queued_symbols
        if research_by_symbol.get(symbol, {}).get('judgment_model')
    )
    for call in classification_call_diagnostics:
        if call.get('success') and call.get('model') and call['model'] not in models_used:
            models_used.append(call['model'])
    selected = preview_portfolio(
        candidate_records,
        research_by_symbol,
        target_selected_etfs,
        max_etfs_per_sector_group,
        max_moderate_per_risk_event,
        require_judgment=True,
    )
    if len(selected) < authoritative_count_before_judgment:
        reopened = authoritative_count_before_judgment - len(selected)
        judgment_reopened_slots += reopened
        if len(selected) < target_selected_etfs:
            authoritative_backfill_batches += 1
        print(
            f'Authoritative ETF judgment reopened {reopened} portfolio slot(s) '
            f'({authoritative_count_before_judgment}->{len(selected)}); research remains open until '
            f'{target_selected_etfs} authoritative/effective selections are restored '
            'or the research budget is exhausted.'
        )
    print(
        f'Portfolio preview after batch: provisional={provisional_count}/'
        f'{target_selected_etfs}, authoritative/effective={len(selected)}/'
        f'{target_selected_etfs}, newly_judged={max(0, judged_after - judged_before)}.'
    )
    if len(selected) >= target_selected_etfs:
        queue_portfolio_challengers(
            candidate_records, research_by_symbol, selected, pending_retries,
            challenger_queued_symbols,
        )
    request_budget.release_summary_for_research = (
        summary_reservation_releasable_on_shortfall
        and len(selected) < target_selected_etfs
    )

# Final forced catch-up handles a deliberately accumulated tail. Because the
# judge skips judged symbols, this cannot introduce repeated classification drift.
set_run_stage('classification_final_tail')
pre_final_selected_count = len(selected)
research_by_symbol = safe_judge_etf_research_pool(
    client, candidate_records, research_by_symbol, market_context, force=True,
)
for call in classification_call_diagnostics:
    if call.get('success') and call.get('model') and call['model'] not in models_used:
        models_used.append(call['model'])
selected = preview_portfolio(
    candidate_records,
    research_by_symbol,
    target_selected_etfs,
    max_etfs_per_sector_group,
    max_moderate_per_risk_event,
    require_judgment=True,
)
if len(selected) < pre_final_selected_count:
    reopened = pre_final_selected_count - len(selected)
    judgment_reopened_slots += reopened
    print(
        f'Warning: final judgment catch-up reopened {reopened} slot(s). '
        'The accumulated tail was judged only after research closed.'
    )

# Regression invariant: do not silently claim that research was unnecessary if
# authoritative judgment has reopened slots while candidate/call capacity remains.
remaining_unresearched = sum(
    1 for candidate in candidate_records
    if str(candidate['Symbol']).upper() not in research_by_symbol
)
actionable_unresearched = []
exhausted_unvalidated = []
sector_capacity_blocked_unresearched = []
factor_family_blocked_unresearched = []
for candidate in candidate_records:
    symbol = str(candidate['Symbol']).upper()
    if symbol in research_by_symbol:
        continue
    if research_attempts_by_symbol.get(symbol, 0) >= max_fresh_research_attempts_per_etf:
        exhausted_unvalidated.append(symbol)
        continue
    if etf_optimistic_final_score(candidate) + score_comparison_epsilon < minimum_final_selection_score:
        continue
    if lower_ranked_candidate_blocked_by_sector_capacity(
        candidate, selected, rank_by_symbol, max_etfs_per_sector_group
    ):
        sector_capacity_blocked_unresearched.append(symbol)
    elif lower_ranked_candidate_blocked_by_factor_family_capacity(candidate, selected):
        factor_family_blocked_unresearched.append(symbol)
    else:
        actionable_unresearched.append(symbol)
if (
    len(selected) < target_selected_etfs
    and actionable_unresearched
    and request_budget.can_reserve('research')
    and api_attempt_budget.remaining(model_primary, 'research') > 0
    and not research_quota_exhausted
):
    print(
        'WARNING — ETF backfill invariant: portfolio is short while actionable '
        f'unresearched candidates ({len(actionable_unresearched)}) and research-call '
        'capacity remain. Candidates: ' + ', '.join(actionable_unresearched[:20])
    )

research_budget_exhausted = (
    len(selected) < target_selected_etfs
    and (not request_budget.can_reserve('research')
         or api_attempt_budget.remaining(model_primary, 'research') <= 0)
)
if research_budget_exhausted:
    print(
        'ETF research budget stop: '
        f'logical={request_budget.research_used}/{request_budget.research_limit}, '
        f'total_logical={request_budget.total_used}/{request_budget.total}, '
        f'API_attempts={request_budget.api_attempts}/{request_budget.max_api_attempts}; '
        f'research_API_remaining={api_attempt_budget.remaining(model_primary, "research")}.'
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
    elif lower_ranked_candidate_blocked_by_sector_capacity(
        candidate, selected, rank_by_symbol, max_etfs_per_sector_group
    ):
        research_disposition[symbol] = 'SECTOR_CAPACITY'
    elif lower_ranked_candidate_blocked_by_factor_family_capacity(candidate, selected):
        research_disposition[symbol] = 'FACTOR_FAMILY_CAPACITY'
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
replacement_diagnostics = build_replacement_diagnostics(
    candidate_records, research_by_symbol, selected
)
print_replacement_diagnostics(replacement_diagnostics)
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
set_run_stage('summary')
if config.get('final_summary_enabled', True):
    selected_summary_input = build_summary_input(selected)
    summary_prompt = (
        config['prompt_html_summary'].rstrip()
        + '\n\nMARKET_CONTEXT:\n'
        + json.dumps(market_context, ensure_ascii=False)
        + '\n\nSELECTED_ETFS:\n'
        + json.dumps(selected_summary_input, ensure_ascii=False)
    )
    summary_data = None
    if (
        classification_api_attempts_used + summary_35_api_attempts_used
        < max_gemini_35_calls_per_run
        and api_attempt_budget.remaining(summary_model, 'summary') > 0
    ):
        api_attempt_budget.reserve(
            summary_model, 'summary', 'ETF HTML summary'
        )
        summary_35_api_attempts_used += 1
        stage = 'ETF HTML summary'
        total_35_attempt = (
            classification_api_attempts_used + summary_35_api_attempts_used
        )
        print(
            f'Gemini 3.5 summary call {total_35_attempt}/'
            f'{max_gemini_35_calls_per_run}: {stage} ({summary_model})'
        )
        try:
            response = client.models.generate_content(
                model=summary_model,
                config=build_gemini_config(
                    summary_thinking_budget,
                    enable_search=False,
                    response_mime_type='application/json',
                    max_output_tokens=gemini_max_output_tokens,
                ),
                contents=summary_prompt,
            )
            response_text = getattr(response, 'text', None)
            summary_data = parse_json_response(response_text)
            summary_html = validate_summary_response(summary_data, selected)
            metadata = extract_gemini_metadata(response)
            metadata['response_text_chars'] = len(response_text or '')
            print_gemini_metadata(stage, metadata)
            models_used.append(summary_model)
            call_diagnostics.append({
                'stage': 'HTML summary', 'model': summary_model,
                'model_family': '3.5', 'api_attempt': total_35_attempt,
                'success': True, **metadata,
            })
            print('Using Gemini 3.5-written ETF HTML summary.')
        except Exception as exc:
            print(f'ETF summary unavailable from {summary_model}: {exc}')
            call_diagnostics.append({
                'stage': 'HTML summary', 'model': summary_model,
                'model_family': '3.5', 'api_attempt': total_35_attempt,
                'success': False, 'error': str(exc),
            })
            summary_data = None
    else:
        print(
            'ETF 3.5 summary reservation was unavailable; trying the 2.5 fallback.'
        )

    if (summary_data is None and request_budget.can_reserve('summary')
            and api_attempt_budget.remaining(summary_fallback_model, 'summary') > 0):
        try:
            summary_data, used_model, metadata = call_gemini_json(
                client,
                summary_fallback_model,
                summary_prompt,
                build_gemini_config(
                    summary_thinking_budget,
                    enable_search=False,
                    response_mime_type='application/json',
                    max_output_tokens=gemini_max_output_tokens,
                ),
                'ETF HTML summary fallback',
                request_budget,
                'summary',
                1,
                initial_delay,
                max_transient_delay,
            )
            summary_html = validate_summary_response(summary_data, selected)
            models_used.append(used_model)
            call_diagnostics.append({
                'stage': 'HTML summary fallback', 'model_family': '2.5',
                **metadata,
            })
            print('Using Gemini 2.5 fallback ETF HTML summary.')
        except Exception as exc:
            print(
                'Gemini summary fallback was unavailable or invalid; using '
                'deterministic fallback summary: '
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
set_run_stage('publish_html')
update_html_page(
    final_recommendations,
    df_html_table,
    'etf_page_template.html',
    'etf_index.html',
    model_used,
)
# previous_diagnostics was loaded before the run so classification drift
# compares against the prior execution rather than the report being written now.
current_classifications = {
    symbol: {
        'portfolio_group': candidate_sector_group(candidate_by_symbol[symbol]),
        'canonical_sector': candidate_by_symbol[symbol].get('CanonicalSector'),
        'style_category': candidate_by_symbol[symbol].get('StyleCategory'),
        'research_admission': research.get('research_admission'),
        'python_eligible': research.get('eligible'),
        'eligibility_reason': research.get('eligibility_reason'),
        'reversal_risk': research.get('reversal_risk'),
        'exposure_reversal_risk': research.get('exposure_reversal_risk'),
        'entry_reversal_risk': research.get('entry_reversal_risk'),
        'continuation_strength': research.get('continuation_strength'),
        'benchmark_outperformance_outlook': research.get('benchmark_outperformance_outlook'),
            'benchmark_outperformance_confidence': research.get('benchmark_outperformance_confidence'),
        'benchmark_outperformance_basis': research.get('benchmark_outperformance_basis'),
        'holdings_concentration': research.get('holdings_concentration'),
        'construction_concentration': research.get('construction_concentration'),
        'primary_reversal_channel': research.get('primary_reversal_channel'),
        'judgment_model': research.get('judgment_model'),
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
        'mechanism_evidence_source_index': research.get(
            'mechanism_evidence_source_index'
        ),
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
    'research_admission', 'python_eligible', 'reversal_risk',
    'exposure_reversal_risk', 'entry_reversal_risk', 'continuation_strength',
    'benchmark_outperformance_outlook', 'holdings_concentration',
    'construction_concentration', 'primary_reversal_channel', 'judgment_model',
    'risk_basis',
    'mechanism_status', 'normalization_probability', 'benchmark_assessment',
    'mandate_assessment', 'us_equity_weight_estimate', 'mandate_basis',
    'evidence_scope', 'adverse_change_observed', 'adverse_change_date',
    'adverse_change_indicator', 'mechanism_evidence_source',
    'mechanism_evidence_source_index',
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

reconciliation_counts = {}
for item in runtime_reconciliation_diagnostics:
    item_type = str(item.get('type') or 'unknown')
    reconciliation_counts[item_type] = (
        reconciliation_counts.get(item_type, 0) + 1
    )
if reconciliation_counts:
    print('ETF runtime reconciliation summary:')
    for item_type, count in sorted(reconciliation_counts.items()):
        print(f'  {item_type}: {count}')
else:
    print('ETF runtime reconciliation summary: no label repairs were needed.')

diagnostics = {
    'schema_version': 2,
    'run_id': RUN_ID,
    'provenance': RUN_PROVENANCE,
    'cache': cache_diagnostics,
    'stages': stage_diagnostics,
    'api_attempts': api_attempt_diagnostics,
    'semantic_matcher': {
        'model': FASTEMBED_MODEL_NAME,
        'loaded': _fastembed_model is not None,
        'unavailable_reason': _fastembed_unavailable_reason,
    },
    'coverage': build_coverage_diagnostics(
        candidate_records, research_by_symbol, research_requests_by_symbol,
        research_attempts_by_symbol, [x['candidate']['Symbol'] for x in selected],
        actionable_unresearched, max_fresh_research_attempts_per_etf,
    ),
    'generated_at': datetime.now(UTC).isoformat(),
    'status': 'SUCCESS_PARTIAL' if partial_portfolio else 'SUCCESS',
    'selection_target': target_selected_etfs,
    'selection_count': len(selected),
    'selection_shortfall': max(0, target_selected_etfs - len(selected)),
    'operational_health': {
        'research_stop_reason': (
            'daily_quota_exhausted' if research_quota_exhausted else
            'portfolio_target_met' if not partial_portfolio else
            'api_or_logical_budget_exhausted' if research_budget_exhausted else
            'no_actionable_work_within_attempt_limits'
        ),
        'classification_quota_exhausted': classification_quota_exhausted,
        'classification_api_attempts_remaining': api_attempt_budget.remaining(
            classification_model, 'classification'
        ),
        'research_api_attempts_remaining': api_attempt_budget.remaining(
            model_primary, 'research'
        ),
    },
    'selection_control': {
        'provisional_portfolio_peak': provisional_portfolio_peak,
        'judgment_reopened_slots': judgment_reopened_slots,
        'authoritative_backfill_batches': authoritative_backfill_batches,
        'remaining_unresearched_candidates': remaining_unresearched,
        'remaining_unvalidated_candidates': remaining_unresearched,
        'actionable_retryable_or_unrequested': actionable_unresearched,
        'validation_exhausted_symbols': exhausted_unvalidated,
        'minimum_final_selection_score': minimum_final_selection_score,
        'max_classification_logical_calls_per_run': max_classification_logical_calls_per_run,
        'max_classification_api_attempts_per_run': max_classification_api_attempts_per_run,
        'max_gemini_35_calls_per_run': max_gemini_35_calls_per_run,
        'benchmark_qvm_floor': benchmark_qvm_floor,
        'benchmark_qvm_override_margin': benchmark_qvm_override_margin,
        'minimum_uncertain_selection_score': minimum_uncertain_selection_score,
        'minimum_moderate_selection_score': minimum_moderate_selection_score,
    },
    'models_used': list(dict.fromkeys(models_used)),
    'budget': {
        'total_used': request_budget.total_used,
        'total_limit': request_budget.total,
        'research_used': request_budget.research_used,
        'research_limit': request_budget.research_limit,
        'summary_used': request_budget.summary_used,
        'api_attempts': request_budget.api_attempts,
        'api_attempt_limit': request_budget.max_api_attempts,
        'research_api_attempts': request_budget.research_api_attempts,
        'summary_api_attempts': request_budget.summary_api_attempts,
        'summary_35_api_attempts': summary_35_api_attempts_used,
        'total_35_api_attempts': (
            classification_api_attempts_used + summary_35_api_attempts_used
        ),
        'context_api_attempts': request_budget.context_api_attempts,
        'per_model_api_attempts': api_attempt_budget.snapshot(),
        'classification_logical_calls': classification_logical_calls_used,
        'classification_api_attempts': classification_api_attempts_used,
        'summary_reservation_releasable_on_shortfall': summary_reservation_releasable_on_shortfall,
    },
    'research': {
        'unique_etfs_requested': len(research_requests_by_symbol),
        'unique_etfs_with_charged_search': len(researched_this_run),
        'requests_by_symbol': research_requests_by_symbol,
        'last_validation_or_transport_error': last_research_errors,
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
        'factor_family_capacity_skipped_symbols': sorted(
            factor_family_capacity_skipped_symbols_this_run
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
    'runtime_reconciliations': runtime_reconciliation_diagnostics,
    'reconciliation_counts': reconciliation_counts,
    'consistency_warnings': consistency_warnings,
    'research_disposition': research_disposition,
    'decision_ledger': decision_ledger,
    'context_review': context_review,
    'classification_calls': classification_call_diagnostics,
    'classification_validation': classification_validation_diagnostics,
    'classification_scheduling': classification_scheduling_diagnostics,
    'candidate_analysis': {
        str(candidate['Symbol']).upper(): {
            'qvm_score': candidate.get('QVMScore'),
            'quality_score': candidate.get('QualityScore'),
            'value_score': candidate.get('ValueScore'),
            'momentum_score': candidate.get('MomentumScore'),
            'price_path_quality': candidate.get('PricePathQuality'),
            'quantitative_entry_risk': candidate.get('QuantitativeEntryRisk'),
            'overextension_penalty': candidate.get('OverextensionPenalty'),
            'returns': {period: candidate.get(f'{period} Return') for period in ('1M','3M','6M','9M','1Y')},
            'benchmark_excess_returns': {
                benchmark: {period: candidate.get(f'{period} Excess vs {benchmark}') for period in ('1M','3M','6M','9M','1Y')}
                for benchmark in benchmark_etfs
            },
            'final_selection_score': (
                etf_final_score(candidate, research_by_symbol[str(candidate['Symbol']).upper()])
                if str(candidate['Symbol']).upper() in research_by_symbol else None
            ),
            'required_selection_score': (
                etf_required_final_score(research_by_symbol[str(candidate['Symbol']).upper()], candidate)
                if str(candidate['Symbol']).upper() in research_by_symbol else None
            ),
        }
        for candidate in candidate_records
    },
    'replacement_diagnostics': replacement_diagnostics,
    'challenger_summary': {
        'queued': sorted(challenger_queued_symbols),
        'researched': sorted(challenger_researched_symbols),
        'judged': sorted(challenger_judged_symbols),
        'displaced_incumbents': sorted({
            row['incumbent_symbol'] for row in replacement_diagnostics
            if row.get('result') == 'CHALLENGER_WOULD_WIN'
        }),
        'capacity_challenger_tests': capacity_challenger_tests,
    },
    'calls': call_diagnostics,
}
report_saved = save_json_object_atomic(
    run_report_file,
    diagnostics,
)
print('ETF COVERAGE ' + json.dumps({
    'counts': diagnostics['coverage']['counts'],
    'actionable_never_requested': diagnostics['coverage']['actionable_never_requested'],
    'actionable_retryable': diagnostics['coverage']['actionable_retryable'],
    'validated_unjudged': diagnostics['coverage']['validated_unjudged'],
    'report_saved': report_saved,
}))
print(
    'ETF selection-control summary:\n'
    f'  provisional portfolio peak: {provisional_portfolio_peak}/{target_selected_etfs}\n'
    f'  portfolio challengers queued: {len(challenger_queued_symbols)}'
    + (f' ({", ".join(sorted(challenger_queued_symbols))})\n' if challenger_queued_symbols else '\n')
    + f'  portfolio challengers researched: {len(challenger_researched_symbols)}'
    + (f' ({", ".join(sorted(challenger_researched_symbols))})\n' if challenger_researched_symbols else '\n')
    + f'  portfolio challengers judged: {len(challenger_judged_symbols)}'
    + (f' ({", ".join(sorted(challenger_judged_symbols))})\n' if challenger_judged_symbols else '\n')
    + f'  capacity challengers evaluated: {len(capacity_challenger_tests)}'
    + (f' ({", ".join(sorted(capacity_challenger_tests))})\n' if capacity_challenger_tests else '\n')
    + f'  replacement head-to-head tests logged: {len(replacement_diagnostics)}\n'
    + f'  max factor-family exposure: {max_etfs_per_factor_family}\n'
    f'  slots reopened by authoritative judgment: {judgment_reopened_slots}\n'
    f'  authoritative backfill batches: {authoritative_backfill_batches}\n'
    f'  remaining unvalidated candidates: {remaining_unresearched}\n'
    f'  actionable never-requested candidates: {len(diagnostics["coverage"]["actionable_never_requested"])}\n'
    f'  actionable retryable candidates: {len(diagnostics["coverage"]["actionable_retryable"])}\n'
    f'  validation-exhausted candidates: {len(exhausted_unvalidated)}\n'
    f'  sector-capacity-blocked unresearched candidates: '
    f'{len(sector_capacity_blocked_unresearched)}\n'
    f'  factor-family-blocked unresearched candidates: '
    f'{len(factor_family_blocked_unresearched)}'
)
print(
    'Gemini request budget:\n'
    f'  logical calls: {request_budget.total_used}/{request_budget.total}\n'
    f'  ETF research logical calls: {request_budget.research_used}/{request_budget.research_limit}\n'
    f'  summary logical calls: {request_budget.summary_used}\n'
    f'  ETF judgment logical calls: {classification_logical_calls_used}/{max_classification_logical_calls_per_run}\n'
    f'  ETF judgment API attempts: {classification_api_attempts_used}/{max_classification_api_attempts_per_run}\n'
    f'  ETF 3.5 summary API attempts: {summary_35_api_attempts_used}/1\n'
    f'  total Gemini 3.5 API attempts: '
    f'{classification_api_attempts_used + summary_35_api_attempts_used}/'
    f'{max_gemini_35_calls_per_run}\n'
    f'  actual API attempts: {request_budget.api_attempts}/{request_budget.max_api_attempts} '
    f'(research={request_budget.research_api_attempts}, '
    f'context={request_budget.context_api_attempts}, '
    f'summary={request_budget.summary_api_attempts})'
)
end_time = time.perf_counter()
print(f'Elapsed time: {round(end_time - start_time)} seconds\n')
set_run_stage('complete')
write_run_checkpoint(diagnostics['status'])
signal.alarm(0)
