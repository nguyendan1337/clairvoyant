import sys
import re
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm
from pathlib import Path
from google import genai
from bs4 import BeautifulSoup
from google.genai import types
from datetime import datetime, timedelta, UTC
import time, random, json, yaml, os, hashlib
import signal
from html import escape
from yfinance import EquityQuery


class TeeStream:
    """Keep the console output while recording the complete run in the repo root."""

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


# Truncate the previous run before any imports or configuration can fail.
RUN_REPORTS_DIR = Path("run_reports")
RUN_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

STOCK_RUN_LOG_FILE = RUN_REPORTS_DIR / "stock_run.log"

_stock_run_log = open(
    STOCK_RUN_LOG_FILE,
    "w",
    encoding="utf-8",
    buffering=1,
)
sys.stdout = TeeStream(sys.stdout, _stock_run_log)
sys.stderr = TeeStream(sys.stderr, _stock_run_log)

CACHE_DIR = Path("caches")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOP_QVM_STOCKS_MD_FILE = CACHE_DIR / "top_qvm_stocks.md"
TOTAL_RUNTIME_TIMEOUT_SECONDS = 60 * 60
GEMINI_REQUEST_TIMEOUT_MS = 10 * 60 * 1000


class TotalRuntimeTimeout(BaseException):
    """Stop the process when the recommender exceeds its total runtime."""


def handle_total_runtime_timeout(signum, frame):
    raise TotalRuntimeTimeout(
        f"Stock recommender exceeded its "
        f"{TOTAL_RUNTIME_TIMEOUT_SECONDS // 60}-minute total runtime limit."
    )

gemini_call_diagnostics = []
gemini_attempt_diagnostics = []
duplicate_result_diagnostics = []
runtime_reconciliation_diagnostics = []
classification_call_diagnostics = []
classification_calls_used = 0
yfinance_fundamentals_diagnostics = {
    "eligible_symbols": 0,
    "cache_hits": 0,
    "live_fetches": 0,
    "fetch_successes": 0,
    "fetch_failures": [],
    "apply_failures": [],
    "pipeline_used": False,
}


DECISION_DIAGNOSTIC_FIELDS = (
    "research_status",
    "crypto_dependence",
    "business_reversal_risk",
    "entry_reversal_risk",
    "reversal_risk",
    "business_concentration",
    "binary_event_risk",
    "benchmark_outperformance_outlook",
    "continuation_strength",
    "risk_basis",
    "catalyst_dependence",
    "mechanism_status",
    "normalization_probability",
    "continuation_outlook",
    "probability_indicator_type",
    "probability_basis",
    "risk_time_horizon",
    "risk_materiality",
    "primary_risk_event_id",
    "risk_exposure_group",
)
EVIDENCE_DIAGNOSTIC_FIELDS = (
    "current_operating_evidence",
    "material_company_event",
    "durable_drivers",
    "temporary_drivers",
    "reversal_mechanism",
    "current_fact",
    "probability_evidence",
    "material_effect",
    "company_difference",
    "sources",
)
PRESENTATION_DIAGNOSTIC_FIELDS = (
    "business_description",
    "industry_context",
    "explanation",
)

def cache_file_path(filename):
    """Route every relative cache filename through the caches directory."""
    path = Path(filename)
    if path.is_absolute() or (path.parts and path.parts[0] == CACHE_DIR.name):
        return str(path)
    return str(CACHE_DIR / path)



def extract_number_with_suffix(s):
    if s is None:
        return None

    s = str(s).strip().upper()

    if s in ["N/A", "NONE", "-", "", "—"]:
        return None

    s = s.replace(',', '')

    match = re.search(r'-?[\d.]+', s)
    if not match:
        return None

    try:
        num = float(match.group())
    except ValueError:
        return None

    # Handle suffixes (B, M, K, T)
    if 'T' in s:
        num *= 1e12
    elif 'B' in s:
        num *= 1e9
    elif 'M' in s:
        num *= 1e6
    elif 'K' in s:
        num *= 1e3

    return num



def clean_52wk_change(s):
    """Robust cleaner for '52 Wk Change %' values like '+2,734.88%' or '−12.34%' """
    if pd.isna(s) or not isinstance(s, str):
        return None
    s = s.replace(',', '').replace('+', '').replace('%', '').strip()
    try:
        return float(s)
    except ValueError:
        return None



def clean_numeric_columns(df, cols):
    """
    Clean numeric columns properly.
    Use special logic for 52 Wk Change %.
    """
    for col in cols:
        if col not in df.columns:
            continue

        if col == '52 WkChange %':
            # Use the robust percentage cleaner
            df[col] = df[col].apply(clean_52wk_change)
        else:
            # Use suffix extractor for Market Cap, Volume, etc.
            df[col] = (
                df[col]
                .astype(str)
                .apply(extract_number_with_suffix)
            )

    return df



HTML_CACHE_FILE = cache_file_path("stock_pages_cache.json")
HTML_CACHE_EXPIRY_DAYS = 1



def load_html_cache():
    if not os.path.exists(HTML_CACHE_FILE):
        return {}

    try:
        with open(HTML_CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        if not isinstance(cache, dict):
            print(
                f"Warning: {HTML_CACHE_FILE} does not contain a JSON object. "
                "Ignoring it."
            )
            return {}

    except (json.JSONDecodeError, OSError) as e:
        print(
            f"Warning: could not read {HTML_CACHE_FILE}: {e}. "
            "Ignoring the invalid cache and rebuilding it."
        )
        return {}

    fresh_cache = {}
    now = datetime.now(UTC)

    for key, entry in cache.items():
        try:
            ts = datetime.fromisoformat(entry["timestamp"])

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            if now - ts < timedelta(days=HTML_CACHE_EXPIRY_DAYS):
                fresh_cache[key] = entry
        except (KeyError, TypeError, ValueError):
            continue

    return fresh_cache


def save_html_cache(cache):
    temp_file = f"{HTML_CACHE_FILE}.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache, f)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replacement prevents a partially written real cache.
        os.replace(temp_file, HTML_CACHE_FILE)

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def fetch_benchmark_performance(symbols, fallback_52_week_change):
    """Return multi-horizon ETF returns and the strongest 52-week hurdle."""
    normalized_symbols = list(dict.fromkeys(
        str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()
    ))
    if not normalized_symbols:
        raise ValueError("benchmark_symbols must contain at least one symbol.")

    print("Downloading benchmark price history: " + ", ".join(normalized_symbols))
    performance = {}
    try:
        downloaded = yf.download(
            normalized_symbols,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        for symbol in normalized_symbols:
            try:
                frame = (
                    downloaded
                    if len(normalized_symbols) == 1
                    else downloaded[symbol]
                )
                close = frame["Close"].dropna()
                if len(close) < 64:
                    raise ValueError(f"only {len(close)} daily closes")

                def trailing_return(sessions):
                    if len(close) <= sessions:
                        return None
                    return float((close.iloc[-1] / close.iloc[-sessions]) * 100 - 100)

                performance[symbol] = {
                    "1M Return": trailing_return(21),
                    "3M Return": trailing_return(63),
                    "6M Return": trailing_return(126),
                    "9M Return": trailing_return(189),
                    "1Y Return": (
                        float((close.iloc[-1] / close.iloc[0]) * 100 - 100)
                        if len(close) >= 230 else None
                    ),
                }
            except Exception as exc:
                print(f"Could not calculate benchmark returns for {symbol}: {exc}")
    except Exception as exc:
        print(f"Benchmark download failed: {exc}")

    valid_1y = {
        symbol: values["1Y Return"]
        for symbol, values in performance.items()
        if values.get("1Y Return") is not None
    }
    if valid_1y:
        hurdle_symbol = max(valid_1y, key=valid_1y.get)
        hurdle_return = float(valid_1y[hurdle_symbol])
        used_fallback = False
    else:
        hurdle_symbol = "FALLBACK"
        hurdle_return = float(fallback_52_week_change)
        used_fallback = True

    context = {
        "symbols": normalized_symbols,
        "returns": performance,
        "hurdle_symbol": hurdle_symbol,
        "hurdle_52_week_return": hurdle_return,
        "used_fallback": used_fallback,
    }
    print(
        f"Effective 52-week stock hurdle: {hurdle_return:.2f}% "
        f"from {hurdle_symbol}."
    )
    return context



def fetch_stock_universe(
        min_52_week_change,
        min_market_cap,
        min_price,
        min_average_volume,
        max_retries=3,
        page_size=250,
):
    """Fetch US stocks passing cheap Yahoo screener filters via yfinance."""
    query = EquityQuery(
        "and",
        [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gte", ["fiftytwowkpercentchange", min_52_week_change]),
            EquityQuery("gte", ["intradaymarketcap", min_market_cap]),
            EquityQuery("gte", ["intradayprice", min_price]),
            EquityQuery("gte", ["avgdailyvol3m", min_average_volume]),
        ],
    )
    rows, offset = [], 0
    while True:
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = yf.screen(
                    query,
                    offset=offset,
                    size=page_size,
                    sortField="fiftytwowkpercentchange",
                    sortAsc=False,
                )
                break
            except Exception as exc:
                print(
                    f"Stock screener attempt {attempt}/{max_retries} failed "
                    f"at offset {offset}: {exc}"
                )
                if attempt < max_retries:
                    time.sleep(2 + random.uniform(0, 1))
        if result is None:
            raise RuntimeError(f"Yahoo stock screener failed at offset {offset}.")
        page = result.get("quotes") or []
        if not page:
            break
        rows.extend(page)
        print(f"Fetched {len(page)} stock screener rows at offset {offset}.")
        offset += len(page)
        total = result.get("total")
        if len(page) < page_size or (total is not None and offset >= total):
            break
    if not rows:
        raise RuntimeError("Yahoo stock screener returned no matching stocks.")
    df = pd.DataFrame(rows)
    if "symbol" not in df.columns:
        raise RuntimeError("Yahoo stock screener response has no symbol field.")
    df["Name"] = df.get("longName", pd.Series(index=df.index, dtype=object))
    if "shortName" in df.columns:
        df["Name"] = df["Name"].fillna(df["shortName"])
    response_columns = {
        "symbol": "Symbol",
        "fiftyTwoWeekChangePercent": "52 WkChange %",
        "fiftyTwoWeekChange": "52 WkChange %",
        "regularMarketPrice": "Price",
        "marketCap": "Market Cap",
        "averageDailyVolume3Month": "Avg Vol (3M)",
        "trailingPE": "P/E Ratio(TTM)",
    }
    for source, target in response_columns.items():
        if source in df.columns and target not in df.columns:
            df = df.rename(columns={source: target})
    df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
    df = df[df["Symbol"].str.match(r"^[A-Z0-9.-]+$", na=False)]
    if "quoteType" in df.columns:
        qt = df["quoteType"].fillna("").astype(str).str.upper()
        df = df[(qt == "") | (qt == "EQUITY")]
    numeric_floors = {
        "52 WkChange %": min_52_week_change,
        "Market Cap": min_market_cap,
        "Price": min_price,
        "Avg Vol (3M)": min_average_volume,
    }
    for column, floor in numeric_floors.items():
        if column not in df.columns:
            raise RuntimeError(f"Yahoo stock screener response has no required {column} field.")
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df[df[column].notna() & (df[column] >= floor)]
    if "P/E Ratio(TTM)" in df.columns:
        df["P/E Ratio(TTM)"] = pd.to_numeric(df["P/E Ratio(TTM)"], errors="coerce")
        df = df[df["P/E Ratio(TTM)"].isna() | (df["P/E Ratio(TTM)"] > 0)]
    else:
        df["P/E Ratio(TTM)"] = np.nan
    df["Name"] = df["Name"].fillna(df["Symbol"])
    return df.drop_duplicates("Symbol", keep="first").sort_values(
        "52 WkChange %", ascending=False
    ).reset_index(drop=True)


def fetch_single_stock_page(url, start=0, count=100, retries=3, sleep=2, cache=None, force_refresh=False):
    paged_url = f"{url}?start={start}&count={count}"

    # Unique cache key per page
    cache_key = f"{url}|{start}|{count}"

    # ---- CACHE HIT ----
    if not force_refresh and cache is not None and cache_key in cache:
        return cache[cache_key]["html"]

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(paged_url, headers=headers, timeout=10)
            response.raise_for_status()

            html = response.text

            # ---- SAVE TO CACHE ----
            if cache is not None:
                cache[cache_key] = {
                    "html": html,
                    "timestamp": datetime.now(UTC).isoformat()
                }

            return html

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed for {paged_url}: {e}")
            if attempt < retries:
                time.sleep(sleep + random.uniform(0, 1))

    print(f"Failed to fetch {paged_url} after {retries} attempts.")
    return None



def parse_stock_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        print("No <table> found in HTML.")
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    # print("Detected headers:", headers)

    rows = []
    for tr in table.find_all('tr')[1:]:  # skip header row
        tds = [td.get_text(strip=True) for td in tr.find_all('td')]
        if len(tds) == len(headers):
            rows.append(tds)

    if not rows:
        print("No data rows parsed.")
    return pd.DataFrame(rows, columns=headers)



def fetch_all_stock_pages_from_url(url, min_52_week_change=20, force_refresh=False):
    all_pages = []
    start = 0
    count = 100

    # ---- LOAD CACHE ----
    cache = load_html_cache()

    target_col = '52 WkChange %'
    numeric_cols = [
        'Price', 'Change', 'Change %', 'Volume',
        'Avg Vol (3M)', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %'
    ]

    while True:
        html = fetch_single_stock_page(url, start=start, count=count, cache=cache, force_refresh=force_refresh)

        if not html:
            print(f"No HTML returned for start={start}. Stopping.")
            break

        df_page = parse_stock_table(html)
        if df_page.empty:
            print(f"Empty page at start={start}. Stopping.")
            break

        if target_col not in df_page.columns:
            print(f"Column '{target_col}' not found at start={start}. Stopping.")
            break

        df_page = clean_numeric_columns(df_page, numeric_cols)

        df_page = df_page[
            df_page[target_col].notna() &
            (df_page['Avg Vol (3M)'] > 0) &
            (df_page['Market Cap'] > 0)
            ]

        if df_page.empty:
            print(f"No valid rows after cleaning at start={start}. Stopping.")
            break

        max_change_on_page = df_page[target_col].max()
        if max_change_on_page < min_52_week_change:
            print(f"Page at start={start} below threshold. Stopping early.")
            break

        all_pages.append(df_page)

        if len(df_page) < count:
            print(f"Last page reached at start={start}.")
            break

        start += count
        time.sleep(1.0)

    # ---- SAVE CACHE ----
    save_html_cache(cache)

    if not all_pages:
        return pd.DataFrame()

    df = pd.concat(all_pages, ignore_index=True)
    df = df[df[target_col] >= min_52_week_change]
    return df.sort_values(target_col, ascending=False).reset_index(drop=True)



def initialize_gemini_client():
    api_key = os.getenv("GEMINI_KEY")

    if not api_key:
        # fallback to local .env for development
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv("GEMINI_KEY")

    if not api_key:
        raise ValueError("GEMINI_KEY not found in environment or .env")

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
    )


def build_gemini_config(
        thinking_budget, enable_search=True, max_output_tokens=None):
    """Create a low-variance Gemini configuration."""
    tools = None
    if enable_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
    options = {
        "tools": tools,
        "temperature": 0,
        "thinking_config": types.ThinkingConfig(
            thinking_budget=thinking_budget
        ),
    }
    if max_output_tokens is not None:
        options["max_output_tokens"] = int(max_output_tokens)
    return types.GenerateContentConfig(**options)


class GeminiRequestBudget:
    def __init__(
            self, maximum, stock_maximum=None, reserved_summary_calls=0):
        self.maximum = int(maximum)
        self.stock_maximum = (
            None if stock_maximum is None else int(stock_maximum)
        )
        self.reserved_summary_calls = int(reserved_summary_calls)
        self.used = 0
        self.stock_used = 0
        self.api_attempts = 0
        self.stock_api_attempts = 0

    def reserve(self, stage, category="general"):
        if category == "stock" and (
                self.stock_maximum is not None
                and self.stock_used >= self.stock_maximum
        ):
            raise RuntimeError(
                f"Gemini stock-research request budget of "
                f"{self.stock_maximum} was exhausted before {stage}."
            )
        # Stock research is required to build a complete portfolio, whereas
        # the final Gemini-written summary is optional and already has a
        # deterministic Python fallback. Let stock calls use a reserved
        # summary slot when earlier market-context attempts consumed more of
        # the shared budget than expected.
        if category not in {"stock", "summary"} and self.used >= (
                self.maximum - self.reserved_summary_calls
        ):
            raise RuntimeError(
                f"Gemini request budget is reserving "
                f"{self.reserved_summary_calls} final-summary call(s) before "
                f"{stage}."
            )
        if self.used >= self.maximum:
            raise RuntimeError(
                f"Gemini request budget of {self.maximum} was exhausted "
                f"before {stage}."
            )
        self.used += 1
        if category == "stock":
            self.stock_used += 1

    def record_api_attempt(self, stage, category="general"):
        self.api_attempts += 1
        if category == "stock":
            self.stock_api_attempts += 1
        print(
            f"Gemini logical request {self.used}/{self.maximum}: {stage} "
            f"(API attempt {self.api_attempts}; "
            f"stock API attempts {self.stock_api_attempts})"
        )


def stable_json_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_object(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: ignoring invalid {path}: {exc}")
        return {}


def save_json_object_atomic(path, value):
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def parse_utc_timestamp(value):
    timestamp = datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def cache_age_hours(created_at):
    """Report an age only when the cached timestamp is usable."""
    if not created_at:
        return None
    try:
        return round(max(0.0, (datetime.now(UTC) - parse_utc_timestamp(
            created_at
        )).total_seconds() / 3600), 2)
    except (TypeError, ValueError, OverflowError):
        return None


def cache_entry_is_fresh(entry, ttl_hours):
    try:
        created_at = parse_utc_timestamp(entry["created_at"])
    except (KeyError, TypeError, ValueError):
        return False
    return datetime.now(UTC) - created_at < timedelta(hours=ttl_hours)


def extract_gemini_metadata(response):
    usage = getattr(response, "usage_metadata", None)
    tool_tokens = (
        getattr(usage, "tool_use_prompt_token_count", 0) or 0
        if usage else 0
    )

    grounding_metadata = None
    if getattr(response, "candidates", None):
        grounding_metadata = getattr(
            response.candidates[0], "grounding_metadata", None
        )

    search_queries = []
    grounding_chunks = []
    if grounding_metadata:
        search_queries = (
            getattr(grounding_metadata, "web_search_queries", None) or []
        )
        grounding_chunks = (
            getattr(grounding_metadata, "grounding_chunks", None) or []
        )

    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "tool_tokens": tool_tokens,
        "cached_tokens": getattr(usage, "cached_content_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
        "search_queries": search_queries,
        "grounding_chunks": grounding_chunks,
    }


def print_gemini_metadata(stage, metadata):
    gemini_call_diagnostics.append({
        "stage": stage,
        "prompt_tokens": metadata["prompt_tokens"],
        "tool_tokens": metadata["tool_tokens"],
        "cached_tokens": metadata["cached_tokens"],
        "thinking_tokens": metadata["thinking_tokens"],
        "output_tokens": metadata["output_tokens"],
        "total_tokens": metadata["total_tokens"],
        "searches_exposed": len(metadata["search_queries"]),
        "grounding_chunks_exposed": len(metadata["grounding_chunks"]),
    })
    print(
        f"Gemini usage [{stage}]:",
        f"prompt_tokens={metadata['prompt_tokens']},",
        f"tool_tokens={metadata['tool_tokens']},",
        f"cached_tokens={metadata['cached_tokens']},",
        f"thinking_tokens={metadata['thinking_tokens']},",
        f"output_tokens={metadata['output_tokens']},",
        f"total_tokens={metadata['total_tokens']}"
    )
    print(
        f"Google searches performed [{stage}]: "
        f"{len(metadata['search_queries'])}"
    )
    for query in metadata["search_queries"]:
        print(f"  Search: {query}")
    print(
        f"Grounding citation chunks exposed [{stage}]: "
        f"{len(metadata['grounding_chunks'])}"
    )


def search_query_matches_candidate(query, candidate):
    query_text = str(query).upper()
    normalized_query = re.sub(r"[^A-Z0-9]+", " ", query_text).strip()

    symbol = str(candidate["Symbol"]).upper()
    symbol_found = re.search(
        rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])",
        query_text,
    ) is not None

    company_suffixes = {
        "INC", "INCORPORATED", "CORP", "CORPORATION", "LTD",
        "LIMITED", "PLC", "LLC", "CO", "COMPANY",
    }
    name_tokens = [
        token
        for token in re.sub(
            r"[^A-Z0-9]+",
            " ",
            str(candidate.get("Name", "")).upper(),
        ).split()
        if token not in company_suffixes
    ]
    normalized_name = " ".join(name_tokens)
    name_found = bool(normalized_name) and normalized_name in normalized_query
    return symbol_found or name_found


def candidate_search_count(search_queries, candidate):
    return sum(
        1
        for query in search_queries or []
        if search_query_matches_candidate(query, candidate)
    )


def minimum_sources_for_candidate(search_queries, candidate):
    # A single search can expose multiple sources. Require at least one
    # materially used source, while the prompt asks Gemini to return every
    # source actually used rather than targeting a fixed count.
    return 1


def normalized_risk_event_key(research):
    """Return one canonical event-only key for portfolio concentration."""
    event_id = re.sub(
        r"[^A-Z0-9]+", "_",
        str(research.get("primary_risk_event_id") or "").upper(),
    ).strip("_")
    if not event_id:
        return None
    return event_id


def normalized_return_driver_key(research):
    """Identify material shared operating exposure, independently of events."""
    raw = research.get("risk_exposure_group")
    return re.sub(r"[^A-Z0-9]+", "_", str(raw or "").upper()).strip("_") or None


def combined_reversal_risk(research):
    """Return the most conservative overall, business, or entry-risk label."""
    levels = [
        str(research.get(field) or "").upper()
        for field in (
            "reversal_risk", "business_reversal_risk", "entry_reversal_risk"
        )
    ]
    valid = [level for level in levels if level in REVERSAL_RISK_ORDER]
    return max(valid, key=REVERSAL_RISK_ORDER.get) if valid else "SEVERE"


def risk_adjusted_candidate_order(
        ranked_batch,
        research_by_symbol,
        initial_sector_counts,
        initial_event_counts,
        initial_moderate_count,
):
    """Greedily prefer safer entries when their QVM scores are close."""
    remaining = list(ranked_batch)
    ordered = []
    simulated_sectors = dict(initial_sector_counts)
    simulated_events = dict(initial_event_counts)
    simulated_moderate_count = int(initial_moderate_count)

    while remaining:
        scored = []
        for index, candidate in enumerate(remaining):
            symbol = str(candidate["Symbol"]).upper()
            research = research_by_symbol.get(symbol)
            qvm_score = float(candidate.get("QVMScore") or 0.0)
            risk = combined_reversal_risk(research or {})
            benchmark_outlook = str(
                (research or {}).get("benchmark_outperformance_outlook")
                or "UNCERTAIN"
            ).upper()
            event_key = normalized_risk_event_key(research or {})
            effective_score = qvm_score
            if risk == "MODERATE":
                effective_score -= moderate_risk_qvm_penalty
            elif risk in {"ELEVATED", "SEVERE"}:
                effective_score -= 1000
            if event_key and simulated_events.get(event_key, 0) >= 1:
                effective_score -= repeated_risk_event_qvm_penalty
            if benchmark_outlook == "UNCERTAIN":
                effective_score -= benchmark_uncertain_qvm_penalty
            elif benchmark_outlook == "UNLIKELY":
                effective_score -= 1000
            scored.append((effective_score, -int(candidate["QVM Rank"]), -index))

        chosen_index = max(range(len(remaining)), key=lambda i: scored[i])
        chosen = remaining.pop(chosen_index)
        ordered.append(chosen)

        symbol = str(chosen["Symbol"]).upper()
        research = research_by_symbol.get(symbol)
        if not research:
            continue
        sector = chosen["Sector"]
        event_key = normalized_risk_event_key(research)
        risk = combined_reversal_risk(research)
        selectable = bool(
            research.get("eligible", True)
            and not excluded_by_crypto_policy(research, excluded_crypto_dependence)
            and risk not in {"ELEVATED", "SEVERE"}
            and (
                not exclude_unlikely_benchmark_outperformance
                or str(research.get("benchmark_outperformance_outlook")).upper()
                != "UNLIKELY"
            )
            and simulated_sectors.get(sector, 0) < max_stocks_per_sector
            and (
                not event_key
                or simulated_events.get(event_key, 0) < max_stocks_per_risk_event
            )
            and (
                risk != "MODERATE"
                or simulated_moderate_count < max_moderate_risk_selections
            )
        )
        if selectable:
            simulated_sectors[sector] = simulated_sectors.get(sector, 0) + 1
            if event_key:
                simulated_events[event_key] = simulated_events.get(event_key, 0) + 1
            if risk == "MODERATE":
                simulated_moderate_count += 1

    return ordered


def partition_ranked_research_candidates(
        ranked_pool,
        sector_counts,
        sector_limit,
        batch_limit,
        candidates_per_open_slot):
    """Bound same-sector research while carrying every excess candidate."""
    research_candidates = []
    selection_batch = []
    carried_candidates = []
    queued_by_sector = {}

    for candidate in ranked_pool:
        sector = candidate["Sector"]
        open_sector_slots = max(
            0, sector_limit - sector_counts.get(sector, 0)
        )
        if open_sector_slots == 0:
            selection_batch.append(candidate)
            continue
        sector_research_limit = (
            open_sector_slots * candidates_per_open_slot
        )
        if (
            queued_by_sector.get(sector, 0) >= sector_research_limit
            or len(research_candidates) >= batch_limit
        ):
            carried_candidates.append(candidate)
            continue
        research_candidates.append(candidate)
        selection_batch.append(candidate)
        queued_by_sector[sector] = queued_by_sector.get(sector, 0) + 1

    return (
        selection_batch,
        research_candidates,
        carried_candidates,
        queued_by_sector,
    )


def validation_error_requires_fresh_research(message):
    """Return True only when another search can materially repair the result."""
    error = str(message).lower()
    research_markers = (
        "research incomplete",
        "result was missing",
        "no current_operating_evidence",
        "requires at least",
        "without current_fact",
        "without probability_evidence",
        "without material_effect",
        "source without a valid url",
        "source without a title",
        "documented shared event",
        "issuer identity conflict",
        "cross-company evidence contamination",
    )
    return any(marker in error for marker in research_markers)


def incomplete_identity_conflicts_with_current_market_data(reason, candidate):
    """Detect a stale issuer-history claim contradicted by screened market data."""
    identity_markers = (
        "was acquired",
        "no standalone",
        "no longer standalone",
        "no longer publicly traded",
        "not publicly traded",
        "was delisted",
        "has been delisted",
        "ticker is inactive",
        "inactive ticker",
        "company no longer exists",
        "ceased operating",
    )
    normalized_reason = " ".join(str(reason).lower().split())
    if not any(marker in normalized_reason for marker in identity_markers):
        return False

    positive_current_fields = 0
    for field in ("Price", "Avg Vol (3M)", "Market Cap", "HistoryDays"):
        try:
            value = float(candidate.get(field))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            positive_current_fields += 1

    # Multiple current fields avoid treating one stale quote as proof that an
    # issuer is active.
    return positive_current_fields >= 2


def parse_json_response(text):
    """Parse a JSON-only response, tolerating an accidental Markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def extract_partial_stock_results(text):
    """Recover only independently valid objects from a malformed results array."""
    cleaned = str(text).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = re.search(r'"results"\s*:\s*\[', cleaned)
    if not match:
        return []

    results = []
    index = match.end()
    text_length = len(cleaned)

    while index < text_length:
        while index < text_length and (
                cleaned[index].isspace() or cleaned[index] == ","
        ):
            index += 1

        if index >= text_length or cleaned[index] == "]":
            break

        if cleaned[index] != "{":
            index += 1
            continue

        object_start = index
        depth = 0
        in_string = False
        escaped = False

        while index < text_length:
            character = cleaned[index]

            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            else:
                if character == '"':
                    in_string = True
                elif character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        object_text = cleaned[object_start:index + 1]
                        try:
                            result = json.loads(object_text)
                        except json.JSONDecodeError:
                            pass
                        else:
                            if isinstance(result, dict):
                                results.append(result)
                        index += 1
                        break

            index += 1
        else:
            break

    return results


def extract_delimited_stock_results(text):
    """Recover independently valid stock objects between explicit markers."""
    pattern = re.compile(
        r"BEGIN_STOCK_RESULT\s*(.*?)\s*END_STOCK_RESULT",
        flags=re.IGNORECASE | re.DOTALL,
    )
    results = []
    for match in pattern.finditer(str(text)):
        object_text = match.group(1).strip()
        object_text = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", object_text,
            flags=re.IGNORECASE,
        ).strip()
        try:
            result = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            results.append(result)
    return results


def judge_stock_research_batch(
        client,
        research_results,
        candidates,
        market_context,
        previous_classifications,
        peer_classifications,
):
    """Use 3.5 Flash as the primary no-Search judge, with 2.5 Flash fallback.

    Research facts and sources remain owned by the grounded 2.5 research stage.
    The judge returns only decision-field patches, which are merged into the
    grounded drafts. This keeps classification quality independent from search
    behavior and prevents a judgment failure from discarding valid research.
    """
    global classification_calls_used
    if not research_results:
        return research_results, None
    if classification_calls_used >= max_classification_calls_per_run:
        print(
            "Classification-call budget exhausted; validating grounded drafts "
            "with Python safeguards."
        )
        return research_results, None

    expected = [str(c["Symbol"]).upper() for c in candidates]
    by_symbol = {
        str(item.get("symbol") or "").upper(): item
        for item in research_results
        if isinstance(item, dict)
    }
    ordered_drafts = [by_symbol[s] for s in expected if s in by_symbol]
    if not ordered_drafts:
        return research_results, None

    compact_candidates = []
    for candidate in candidates:
        symbol = str(candidate["Symbol"]).upper()
        if symbol not in by_symbol:
            continue
        compact_candidates.append({
            "symbol": symbol,
            "name": candidate.get("Name"),
            "sector": candidate.get("Sector"),
            "qvm_rank": candidate.get("QVM Rank"),
            "qvm_score": candidate.get("QVMScore"),
            "quality_score": candidate.get("QualityScore"),
            "value_score": candidate.get("ValueScore"),
            "momentum_score": candidate.get("MomentumScore"),
            "quantitative_entry_risk": candidate.get("QuantitativeEntryRisk"),
            "overextension_penalty": candidate.get("OverextensionPenalty"),
            "returns": {
                "1m": candidate.get("1M Return"),
                "3m": candidate.get("3M Return"),
                "6m": candidate.get("6M Return"),
                "9m": candidate.get("9M Return"),
                "1y": candidate.get("52 WkChange %"),
            },
            "price_path": {
                "annualized_volatility": candidate.get("AnnualizedVolatility"),
                "downside_volatility": candidate.get("DownsideVolatility"),
                "max_drawdown": candidate.get("MaxDrawdown"),
                "positive_day_pct": candidate.get("PositiveDayPct"),
                "trend_r2": candidate.get("TrendR2"),
                "annualized_trend": candidate.get("AnnualizedTrend"),
                "distance_50dma": candidate.get("Distance50DMA"),
                "distance_200dma": candidate.get("Distance200DMA"),
                "distance_52w_high": candidate.get("Distance52WHigh"),
                "largest_1d_move": candidate.get("Largest1DayMove"),
                "largest_5d_move": candidate.get("Largest5DayMove"),
                "momentum_acceleration": candidate.get("MomentumAcceleration"),
            },
            "benchmark_excess_returns": {
                "1m": candidate.get("BenchmarkExcess1M"),
                "3m": candidate.get("BenchmarkExcess3M"),
                "6m": candidate.get("BenchmarkExcess6M"),
                "9m": candidate.get("BenchmarkExcess9M"),
                "1y": candidate.get("BenchmarkExcess1Y"),
            },
            "research": by_symbol[symbol],
            "previous_classification": previous_classifications.get(symbol),
        })

    prompt = (
        config["prompt_stock_judgment"]
        + "\n\nCURRENT_DATE_UTC: "
        + datetime.now(UTC).date().isoformat()
        + "\n\nMARKET_CONTEXT:\n"
        + json.dumps(market_context, ensure_ascii=False)
        + "\n\nBENCHMARK_CONTEXT:\n"
        + json.dumps(benchmark_context, ensure_ascii=False)
        + "\n\nPEER_CLASSIFICATIONS_FROM_EARLIER_BATCHES:\n"
        + json.dumps(peer_classifications, ensure_ascii=False)
        + "\n\nCANDIDATES_WITH_GROUNDED_RESEARCH:\n"
        + json.dumps(compact_candidates, ensure_ascii=False)
    )

    models = [classification_model]
    if classification_fallback_model not in models:
        models.append(classification_fallback_model)
    last_error = None
    for model_index, model_name in enumerate(models):
        attempts = classification_attempts if model_index == 0 else 1
        for attempt in range(1, attempts + 1):
            if classification_calls_used >= max_classification_calls_per_run:
                break
            classification_calls_used += 1
            stage = (
                f"stock judgment for {len(compact_candidates)} stocks "
                f"({model_name}, attempt {attempt}/{attempts})"
            )
            print(
                f"Gemini classification request {classification_calls_used}/"
                f"{max_classification_calls_per_run}: {stage}"
            )
            try:
                response = client.models.generate_content(
                    model=model_name,
                    config=build_gemini_config(
                        classification_thinking_budget,
                        enable_search=False,
                        max_output_tokens=classification_max_output_tokens,
                    ),
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if not text or not text.strip():
                    raise ValueError("Empty classification response.")
                data = parse_json_response(text)
                patches = data.get("results") if isinstance(data, dict) else None
                if not isinstance(patches, list):
                    raise ValueError("Classification response lacks results array.")
                patch_by_symbol = {
                    str(item.get("symbol") or "").upper(): item
                    for item in patches if isinstance(item, dict)
                }
                missing = [s for s in expected if s in by_symbol and s not in patch_by_symbol]
                if not patch_by_symbol or len(patch_by_symbol) != len(patches):
                    raise ValueError(
                        "Classification returned no distinct, symbol-keyed patches."
                    )
                unexpected = sorted(set(patch_by_symbol).difference(expected))
                if unexpected:
                    raise ValueError(
                        "Classification returned unexpected symbols: "
                        + ", ".join(unexpected)
                    )
                if missing:
                    print(
                        "Classification response omitted "
                        + ", ".join(missing)
                        + "; preserving the returned judgments and queuing only "
                        "the missing research for later classification."
                    )

                allowed = {
                    "business_reversal_risk", "entry_reversal_risk",
                    "reversal_risk", "risk_basis", "catalyst_dependence",
                    "business_concentration", "binary_event_risk",
                    "benchmark_outperformance_outlook", "benchmark_outperformance_basis",
                    "continuation_strength",
                    "mechanism_status", "normalization_probability",
                    "continuation_outlook", "probability_indicator_type",
                    "probability_basis", "risk_time_horizon",
                    "risk_materiality", "primary_risk_event_id",
                    "risk_exposure_group", "reversal_mechanism",
                    "probability_evidence", "material_effect", "explanation",
                    "primary_reversal_channel", "classification_change_reason",
                    "material_new_evidence",
                }
                merged = []
                for symbol in expected:
                    draft = by_symbol.get(symbol)
                    if draft is None or symbol not in patch_by_symbol:
                        continue
                    result = dict(draft)
                    patch = patch_by_symbol[symbol]
                    for field in allowed:
                        if field in patch:
                            result[field] = patch[field]
                    if "risk_exposure_group" not in patch:
                        result["_missing_return_driver_group"] = True
                    merged.append(result)
                classification_call_diagnostics.append({
                    "stage": "stock_judgment",
                    "model": model_name,
                    "attempt": attempt,
                    "symbols": [x["symbol"] for x in compact_candidates],
                    "returned_symbols": [x["symbol"] for x in merged],
                    "missing_symbols": missing,
                    "fallback": model_index > 0,
                    "success": True,
                    "status": "PARTIAL" if missing else "SUCCESS",
                })
                gemini_attempt_diagnostics.append({
                    "stage": "stock_judgment",
                    "model": model_name,
                    "attempt": attempt,
                    "category": "classification",
                    "search_enabled": False,
                    "symbols": [x["symbol"] for x in compact_candidates],
                    "status": "PARTIAL" if missing else "SUCCESS",
                })
                if model_index > 0:
                    print(
                        "Gemini 3.5 judgment unavailable; successfully used "
                        "2.5 Flash no-Search judgment fallback."
                    )
                return merged, model_name
            except Exception as exc:
                last_error = exc
                classification_call_diagnostics.append({
                    "stage": "stock_judgment",
                    "model": model_name,
                    "attempt": attempt,
                    "symbols": [x["symbol"] for x in compact_candidates],
                    "fallback": model_index > 0,
                    "success": False,
                    "status": "ERROR",
                    "error": str(exc),
                })
                gemini_attempt_diagnostics.append({
                    "stage": "stock_judgment",
                    "model": model_name,
                    "attempt": attempt,
                    "category": "classification",
                    "search_enabled": False,
                    "symbols": [x["symbol"] for x in compact_candidates],
                    "status": "ERROR",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(f"Warning: {stage} failed: {exc}")
                if attempt < attempts:
                    delay = min(
                        max_transient_backoff_seconds,
                        initial_transient_backoff_seconds * (2 ** (attempt - 1)),
                    ) + random.uniform(0, transient_backoff_jitter_seconds)
                    print(f"Retrying classification in {delay:.1f}s...")
                    time.sleep(delay)

    print(
        "Warning: all judgment models were unavailable; validating the grounded "
        f"2.5 research drafts directly: {last_error}"
    )
    return research_results, None


def compact_diagnostic_value(value, maximum=120):
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return rendered if len(rendered) <= maximum else rendered[:maximum - 3] + "..."


def stock_result_field_differences(first, repeated):
    """Classify duplicate-block differences without changing either result."""
    changed_fields = sorted(
        field
        for field in set(first).union(repeated)
        if field != "symbol" and first.get(field) != repeated.get(field)
    )
    decision_fields = [
        field for field in changed_fields
        if field in DECISION_DIAGNOSTIC_FIELDS
    ]
    evidence_fields = [
        field for field in changed_fields
        if field in EVIDENCE_DIAGNOSTIC_FIELDS
    ]
    presentation_fields = [
        field for field in changed_fields
        if field in PRESENTATION_DIAGNOSTIC_FIELDS
    ]
    categorized = set(
        decision_fields + evidence_fields + presentation_fields
    )
    return {
        "changed_fields": changed_fields,
        "decision_fields": decision_fields,
        "evidence_fields": evidence_fields,
        "presentation_fields": presentation_fields,
        "other_fields": [
            field for field in changed_fields if field not in categorized
        ],
        "decision_changes": {
            field: {
                "retained": first.get(field),
                "repeated": repeated.get(field),
            }
            for field in decision_fields
        },
    }


def deduplicate_stock_results(results, expected_candidates, stage=None):
    """Keep one recovered object per symbol in authoritative input order."""
    expected_order = [
        str(candidate["Symbol"]).strip().upper()
        for candidate in expected_candidates
    ]
    first_by_symbol = {}
    unsymbolized = []
    duplicate_symbols = []
    conflicting_symbols = []
    duplicate_counts = {}
    field_differences_by_symbol = {}

    for result in results:
        symbol = str(result.get("symbol", "")).strip().upper()
        if not symbol:
            # Preserve malformed objects so structural validation still fails
            # explicitly instead of silently hiding the model error.
            unsymbolized.append(result)
            continue
        if symbol not in first_by_symbol:
            first_by_symbol[symbol] = result
            continue

        duplicate_symbols.append(symbol)
        duplicate_counts[symbol] = duplicate_counts.get(symbol, 0) + 1
        if result != first_by_symbol[symbol]:
            conflicting_symbols.append(symbol)
            differences = stock_result_field_differences(
                first_by_symbol[symbol], result
            )
            existing = field_differences_by_symbol.get(symbol)
            if existing is None:
                field_differences_by_symbol[symbol] = differences
            else:
                for category in (
                    "changed_fields", "decision_fields", "evidence_fields",
                    "presentation_fields", "other_fields",
                ):
                    existing[category] = sorted(set(
                        existing[category] + differences[category]
                    ))
                existing["decision_changes"].update(
                    differences["decision_changes"]
                )

    if duplicate_symbols:
        print(
            "Ignored repeated recovered stock-result blocks for: "
            + ", ".join(dict.fromkeys(duplicate_symbols))
        )
    if conflicting_symbols:
        print(
            "Warning: repeated blocks differed for these symbols; retaining "
            "the first complete block: "
            + ", ".join(dict.fromkeys(conflicting_symbols))
        )
        for symbol in dict.fromkeys(conflicting_symbols):
            differences = field_differences_by_symbol[symbol]
            print(f"Duplicate result differences [{symbol}]:")
            for category, label in (
                ("decision_fields", "decision-critical"),
                ("evidence_fields", "evidence"),
                ("presentation_fields", "presentation-only"),
                ("other_fields", "other"),
            ):
                fields = differences[category]
                if fields:
                    print(f"  {label}: {', '.join(fields)}")
            for field, values in differences["decision_changes"].items():
                print(
                    f"    {field}: retained "
                    f"{compact_diagnostic_value(values['retained'])} -> "
                    f"repeated {compact_diagnostic_value(values['repeated'])}"
                )

    for symbol in dict.fromkeys(duplicate_symbols):
        differences = field_differences_by_symbol.get(symbol, {
            "changed_fields": [],
            "decision_fields": [],
            "evidence_fields": [],
            "presentation_fields": [],
            "other_fields": [],
            "decision_changes": {},
        })
        duplicate_result_diagnostics.append({
            "stage": stage,
            "symbol": symbol,
            "extra_blocks": duplicate_counts.get(symbol, 0),
            **differences,
        })

    ordered = [
        first_by_symbol.pop(symbol)
        for symbol in expected_order
        if symbol in first_by_symbol
    ]
    # Retain unexpected symbols so the existing validator reports them.
    ordered.extend(first_by_symbol.values())
    ordered.extend(unsymbolized)
    return ordered


def normalized_evidence_text(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def detect_cross_company_evidence_contamination(results):
    """Flag likely issuer-fact contamination without rejecting shared reasoning.

    Generic mechanism/effect wording is common among peers and is only a weak
    signal. Fresh research is required only when duplicated issuer-specific
    evidence or multiple matching specific financial figures creates a strong
    contamination signal.
    """
    items = [item for item in results if isinstance(item, dict)]
    conflicts = {}
    high_specificity_fields = (
        "current_operating_evidence",
        "current_fact",
        "probability_evidence",
    )
    medium_specificity_fields = ("business_description",)
    low_specificity_fields = (
        "material_effect",
        "reversal_mechanism",
        "explanation",
    )
    figure_fields = (
        *high_specificity_fields,
        *medium_specificity_fields,
        *low_specificity_fields,
    )
    money_pattern = re.compile(
        r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion|m|b)\b",
        flags=re.IGNORECASE,
    )

    for left_index, left in enumerate(items):
        left_symbol = str(left.get("symbol") or "").strip().upper()
        if not left_symbol:
            continue
        for right in items[left_index + 1:]:
            right_symbol = str(right.get("symbol") or "").strip().upper()
            if not right_symbol or right_symbol == left_symbol:
                continue

            identical_high = []
            identical_medium = []
            identical_low = []
            for field, bucket in (
                *((field, identical_high) for field in high_specificity_fields),
                *((field, identical_medium) for field in medium_specificity_fields),
                *((field, identical_low) for field in low_specificity_fields),
            ):
                left_text = normalized_evidence_text(left.get(field))
                right_text = normalized_evidence_text(right.get(field))
                if len(left_text) >= 35 and left_text == right_text:
                    bucket.append(field)

            left_figures = set()
            right_figures = set()
            for field in figure_fields:
                left_figures.update(
                    match.group(0).lower()
                    for match in money_pattern.finditer(str(left.get(field) or ""))
                )
                right_figures.update(
                    match.group(0).lower()
                    for match in money_pattern.finditer(str(right.get(field) or ""))
                )
            shared_figures = sorted(left_figures.intersection(right_figures))

            suspicion_score = 0.0
            suspicion_score += 3.0 * len(identical_high)
            suspicion_score += 2.0 * len(identical_medium)
            suspicion_score += 0.5 * len(identical_low)
            suspicion_score += min(4.0, 2.0 * len(shared_figures))

            # A generic shared material effect or reversal mechanism is normal
            # for peers. Require a strong issuer-specific signal before forcing
            # another company search.
            if suspicion_score < 3.0:
                if identical_low or shared_figures:
                    print(
                        f"Cross-company similarity noted for {left_symbol}/"
                        f"{right_symbol} but below retry threshold: "
                        f"score={suspicion_score:.1f}, "
                        f"low_specificity_fields={identical_low or 'none'}, "
                        f"shared_specific_financial_figures="
                        f"{shared_figures or 'none'}."
                    )
                continue

            detail = (
                f"possible cross-company evidence contamination between "
                f"{left_symbol} and {right_symbol}: score={suspicion_score:.1f}, "
                f"high_specificity_fields={identical_high or 'none'}, "
                f"medium_specificity_fields={identical_medium or 'none'}, "
                f"low_specificity_fields={identical_low or 'none'}, "
                f"shared_specific_financial_figures={shared_figures or 'none'}"
            )
            conflicts[left_symbol] = detail
            conflicts[right_symbol] = detail

    return conflicts

def validate_final_research_state(research_by_symbol):
    """Fail closed if deterministic reconciliation leaves contradictory risk state."""
    errors = []
    for symbol, research in research_by_symbol.items():
        stored = str(research.get("reversal_risk") or "").upper()
        effective = combined_reversal_risk(research)
        if stored != effective:
            errors.append(
                f"{symbol}: reversal_risk={stored or 'MISSING'}, "
                f"business={research.get('business_reversal_risk')}, "
                f"entry={research.get('entry_reversal_risk')}, effective={effective}"
            )
    if errors:
        raise RuntimeError(
            "Final research-state consistency failure:\n" + "\n".join(errors)
        )


def prioritize_stock_sources(sources, maximum):
    """Deduplicate sources in model-provided relevance order."""
    if not isinstance(sources, list):
        return sources

    unique_sources = []
    seen_urls = set()
    for source in sources:
        if not isinstance(source, dict):
            unique_sources.append(source)
            continue
        url = str(source.get("url", "")).strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        # Older cached responses may contain this now-unused optional metadata.
        source.pop("supports", None)
        unique_sources.append(source)

    return unique_sources[:maximum]


def validate_sources(sources, label, minimum=1, maximum=None):
    if not isinstance(sources, list) or len(sources) < minimum:
        raise ValueError(
            f"{label} requires at least {minimum} research sources."
        )

    unique_urls = set()
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"{label} contains an invalid source entry.")

        if not str(source.get("title", "")).strip():
            raise ValueError(f"{label} has a source without a title.")

        url = str(source.get("url", "")).strip()
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"{label} has a source without a valid URL.")

        unique_urls.add(url)
        source.pop("supports", None)

    if len(unique_urls) < minimum:
        raise ValueError(
            f"{label} requires at least {minimum} distinct source URLs."
        )
    if maximum is not None and len(sources) > maximum:
        raise ValueError(
            f"{label} returned {len(sources)} source entries; "
            f"the maximum is {maximum}."
        )

    return {}


def validate_stock_research_evidence(
        data, expected_candidates, minimum_sources=1):
    """Validate the 2.5 evidence packet without requiring 3.5 judgment fields."""
    results = data.get("results")
    if not isinstance(results, list) or len(results) != len(expected_candidates):
        raise ValueError("Research results do not match the supplied candidates.")
    for candidate, result in zip(expected_candidates, results):
        symbol = str(candidate["Symbol"]).upper()
        if not isinstance(result, dict) or str(result.get("symbol") or "").upper() != symbol:
            raise ValueError(f"Research identity/order mismatch for {symbol}.")
        status = str(result.get("research_status") or "COMPLETE").strip().upper()
        if status == "INCOMPLETE":
            reason = str(result.get("research_incomplete_reason") or "").strip()
            if incomplete_identity_conflicts_with_current_market_data(reason, candidate):
                reason = (
                    "issuer identity conflict: verify the current issuer using "
                    "an official source, including possible relisting or spin-off"
                )
            raise ValueError(
                f"{symbol} research incomplete: {reason or 'current evidence missing'}"
            )
        if status != "COMPLETE":
            raise ValueError(f"{symbol} has invalid research_status {status!r}.")
        result["research_status"] = status
        result["qvm_rank"] = int(candidate["QVM Rank"])
        result["eligible"] = True
        result["eligibility_reason"] = None
        for field in (
                "business_description", "industry_context",
                "current_operating_evidence"):
            value = " ".join(str(result.get(field) or "").split())
            if not value:
                raise ValueError(f"{symbol} research incomplete: missing {field}.")
            result[field] = value
        drivers = result.get("durable_drivers")
        if not isinstance(drivers, list) or not any(str(x).strip() for x in drivers):
            raise ValueError(f"{symbol} research incomplete: missing durable drivers.")
        result["crypto_dependence"] = normalize_crypto_dependence(result)
        sources = result.get("sources")
        if isinstance(sources, list):
            result["sources"] = prioritize_stock_sources(sources, max_stock_sources)
        validate_sources(
            result.get("sources"), symbol,
            minimum=max(1, minimum_sources), maximum=max_stock_sources,
        )
        print(
            f"Validated grounded research [{symbol}]: "
            f"{len({str(x['url']).strip() for x in result['sources']})} "
            "distinct source URLs; awaiting judgment."
        )


def validate_market_context(data):
    required = {
        "as_of_date", "market_status", "market_intro", "market_direction",
        "major_drivers", "macro_conditions", "strong_sectors",
        "weak_sectors", "sector_context", "active_risk_events", "sources"
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Market context is missing fields: {sorted(missing)}")
    if data["market_status"] not in {"STRONG", "MIXED", "WEAK"}:
        raise ValueError("Market context has an invalid market_status.")
    if not str(data["market_intro"]).strip():
        raise ValueError("Market context has an empty market_intro.")
    if not re.search(r"\d+(?:\.\d+)?\s*%", str(data["market_intro"])):
        raise ValueError("Market context market_intro has no S&P 500 percentage.")
    if not isinstance(data["sector_context"], dict):
        raise ValueError("Market context sector_context must be an object.")
    active_risk_events = data["active_risk_events"]
    if not isinstance(active_risk_events, list):
        raise ValueError("Market context active_risk_events must be an array.")
    seen_event_ids = set()
    for index, event in enumerate(active_risk_events):
        if not isinstance(event, dict):
            raise ValueError(
                f"Market risk event {index + 1} must be an object."
            )
        event_id = str(event.get("event_id", "")).strip()
        if not re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)*", event_id):
            raise ValueError(
                f"Market risk event {index + 1} has invalid event_id "
                f"{event_id!r}; expected UPPER_SNAKE_CASE."
            )
        if event_id in seen_event_ids:
            raise ValueError(f"Duplicate market risk event_id {event_id!r}.")
        seen_event_ids.add(event_id)
        if not str(event.get("description", "")).strip():
            raise ValueError(f"Market risk event {event_id} has no description.")
        affected_industries = event.get("affected_industries")
        if (
                not isinstance(affected_industries, list)
                or not affected_industries
                or any(not str(value).strip() for value in affected_industries)
        ):
            raise ValueError(
                f"Market risk event {event_id} must have affected_industries."
            )
        if not str(event.get("normalization_risk", "")).strip():
            raise ValueError(
                f"Market risk event {event_id} has no normalization_risk."
            )
    validate_sources(data["sources"], "Market context")


ALLOWED_REVERSAL_RISKS = {
    "MINIMAL",
    "LOW",
    "MODERATE",
    "ELEVATED",
    "SEVERE",
}
REVERSAL_RISK_ORDER = {
    "MINIMAL": 0,
    "LOW": 1,
    "MODERATE": 2,
    "ELEVATED": 3,
    "SEVERE": 4,
}


def validate_stock_batch_structure(data, expected_candidates):
    """Allow partial batches while rejecting duplicate or unexpected symbols."""
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Stock batch response is missing a results array.")

    expected_symbols = {
        str(candidate["Symbol"]).upper() for candidate in expected_candidates
    }
    returned_symbols = [
        str(result.get("symbol", "")).upper() for result in results
    ]
    if any(not symbol for symbol in returned_symbols):
        raise ValueError("Stock batch contains a result without a symbol.")
    if len(returned_symbols) != len(set(returned_symbols)):
        raise ValueError("Stock batch contains duplicate symbols.")

    unexpected = sorted(set(returned_symbols).difference(expected_symbols))
    if unexpected:
        raise ValueError(
            "Stock batch contains unexpected symbols: " + ", ".join(unexpected)
        )


NON_PROBABILITY_INDICATOR_ALIASES = {
    "ESTIMATE_MISS",
    "ANALYST_FORECAST",
    "HISTORICAL_CYCLICALITY",
    "ORDINARY_VOLATILITY",
    "DIFFICULT_COMPARISON",
    "PENDING_NEGOTIATION",
    "INCREASED_GROWTH_INVESTMENT",
}

NONRECURRING_TEMPORARY_ITEM_PATTERN = re.compile(
    r"\b(?:refunds?|settlements?|recover(?:y|ies)|asset[ -]sale gains?|"
    r"tax benefits?|tax credits?|accounting gains?|"
    r"one[ -](?:time|off) gains?)\b",
    re.IGNORECASE,
)

OPERATING_SUPPORT_ATTRIBUTION_PATTERN = re.compile(
    r"\b(?:(?:driven|supported|boosted|aided)\s+by|benefited\s+from)\b",
    re.IGNORECASE,
)

UNUSUALLY_FAVORABLE_MARKET_PATTERN = re.compile(
    r"\b(?:elevated|exceptional(?:ly)?(?:[ -](?:high|strong))?|"
    r"record(?:[ -]high)?|surging|"
    r"unusually[ -](?:high|strong))\s+"
    r"(?:(?:commodity|crude|oil|gas|gold|silver|refining|freight|tanker|"
    r"memory|spot|tce|contract)\s+){0,3}"
    r"(?:prices?|rates?|spreads?|margins?|demand)\b|"
    r"\brobust\s+(?:(?:refining|freight|tanker|spot|tce)\s+){1,2}"
    r"(?:rates?|spreads?|margins?)\b|"
    r"\b(?:tight supply|supply shortages?|capacity constraints?)\b",
    re.IGNORECASE,
)


def has_nonrecurring_temporary_item(temporary_drivers):
    """Detect comparison adjustments incorrectly modeled as operating drivers."""
    return any(
        NONRECURRING_TEMPORARY_ITEM_PATTERN.search(str(driver))
        for driver in temporary_drivers or []
    )


def describes_unmodeled_temporary_market_support(result):
    """Detect explicit temporary market support omitted from risk fields."""
    text = " ".join(
        str(result.get(field) or "")
        for field in (
            "current_operating_evidence",
            "industry_context",
            "explanation",
        )
    )
    return bool(
        OPERATING_SUPPORT_ATTRIBUTION_PATTERN.search(text)
        and UNUSUALLY_FAVORABLE_MARKET_PATTERN.search(text)
    )


DEPENDENCE_LEVELS = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
CRYPTO_DEPENDENCE_LEVELS = {"NONE", "INCIDENTAL", "MATERIAL", "PRIMARY"}


def cautious_exposure_floor_applies(result, enabled, minimum_dependence):
    """Identify material temporary-driver exposure even before probability."""
    dependence = str(result.get("catalyst_dependence") or "").upper()
    minimum = str(minimum_dependence or "MODERATE").upper()
    return bool(
        enabled
        and result.get("risk_basis") == "TEMPORARY_DRIVER_NORMALIZATION"
        and DEPENDENCE_LEVELS.get(dependence, -1)
        >= DEPENDENCE_LEVELS.get(minimum, 1)
        and result.get("temporary_drivers")
        and not has_nonrecurring_temporary_item(result.get("temporary_drivers"))
        and str(result.get("material_effect") or "").strip()
        and result.get("risk_time_horizon") in {
            "0_3_MONTHS", "3_6_MONTHS", "6_12_MONTHS"
        }
    )


def normalize_crypto_dependence(result):
    """Normalize Gemini's field, with a conservative no-call fallback."""
    raw_value = re.sub(
        r"[^A-Z0-9]+", "_",
        str(result.get("crypto_dependence") or "").upper(),
    ).strip("_")
    aliases = {
        "NO": "NONE", "NOT_APPLICABLE": "NONE", "MINIMAL": "INCIDENTAL",
        "MODERATE": "MATERIAL", "HIGH": "PRIMARY",
    }
    value = aliases.get(raw_value, raw_value)
    if value in CRYPTO_DEPENDENCE_LEVELS:
        return value

    # Avoid spending a search solely on a newly added enum while failing
    # conservatively for a clearly crypto-centered business description.
    text = " ".join(str(result.get(field) or "") for field in (
        "business_description", "industry_context", "current_operating_evidence"
    ))
    if re.search(
        r"\b(?:digital[- ]asset treasury|crypto(?:currency)? treasury|"
        r"bitcoin treasury|crypto mining|bitcoin mining|token staking|"
        r"crypto exchange|crypto lending)\b", text, re.IGNORECASE,
    ):
        return "PRIMARY"
    if re.search(
        r"\b(?:crypto(?:currency)?|bitcoin|digital assets?|tokens?|blockchain)\b",
        text, re.IGNORECASE,
    ):
        return "MATERIAL"
    return "NONE"


def excluded_by_crypto_policy(research, excluded_levels):
    return str(research.get("crypto_dependence") or "NONE").upper() in {
        str(level).upper() for level in excluded_levels
    }


def align_component_risks_to_authoritative(result, authoritative_risk):
    """Keep component risks consistent after a deterministic Python reconciliation."""
    authoritative_risk = str(authoritative_risk).upper()
    if authoritative_risk not in REVERSAL_RISK_ORDER:
        return {}
    changed = {}
    for field in ("business_reversal_risk", "entry_reversal_risk"):
        current = str(result.get(field) or authoritative_risk).upper()
        if (
                current in REVERSAL_RISK_ORDER
                and REVERSAL_RISK_ORDER[current] > REVERSAL_RISK_ORDER[authoritative_risk]
        ):
            changed[field] = {"from": current, "to": authoritative_risk}
            result[field] = authoritative_risk
    return changed


def downgrade_unproven_material_risk(result, symbol, reason):
    """Apply the prompt's LOW mapping when probability was not established."""
    prior_risk = str(result.get("reversal_risk", "")).upper()
    prior_components = {
        field: str(result.get(field) or prior_risk).upper()
        for field in ("business_reversal_risk", "entry_reversal_risk")
    }
    evidence_fields = (
        "mechanism_status", "normalization_probability", "risk_materiality",
        "current_fact", "probability_evidence", "material_effect",
        "risk_time_horizon", "probability_indicator_type", "probability_basis",
        "catalyst_dependence", "temporary_drivers", "continuation_strength",
    )
    evidence_before = {field: result.get(field) for field in evidence_fields}
    result["reversal_risk"] = "LOW"
    component_changes = align_component_risks_to_authoritative(result, "LOW")
    result["mechanism_status"] = "HYPOTHETICAL"
    result["normalization_probability"] = (
        "NOT_ESTABLISHED"
        if result.get("temporary_drivers")
        else "NOT_APPLICABLE"
    )
    result["continuation_outlook"] = "CONTINUATION_MORE_LIKELY"
    result["probability_indicator_type"] = "NONE"
    if result.get("probability_basis") not in {
        "EXTERNAL_EXPECTATION_ONLY", "NONE"
    }:
        result["probability_basis"] = "NONE"
    result["risk_materiality"] = "LOW"
    result["current_fact"] = None
    result["probability_evidence"] = None
    result["material_effect"] = None
    result["reversal_evidence"] = None
    result["risk_time_horizon"] = None
    result["reversal_mechanism"] = (
        "No qualifying current reversal mechanism established"
    )
    result["normalization_effect"] = (
        "No probable normalization effect established"
    )

    operating = " ".join(
        str(result.get("current_operating_evidence") or "").split()
    ).rstrip(".")
    drivers = result.get("durable_drivers") or []
    driver = (
        " ".join(str(drivers[0]).split()).rstrip(".")
        if drivers else "Current operations provide continuation support"
    )
    result["explanation"] = (
        f"{operating}. {driver}. Potential reversal exposure exists, but the "
        "available current evidence does not establish a measurable adverse "
        "indicator making a material business reversal probable within 6-12 "
        "months. Continuation therefore remains more likely."
    )
    print(
        f"Reconciled {symbol} {prior_risk} to LOW without another Gemini call: "
        f"{reason}."
    )
    evidence_after = {field: result.get(field) for field in evidence_fields}
    print(f"  Reconciliation evidence: before={evidence_before}; after={evidence_after}")
    runtime_reconciliation_diagnostics.append({
        "symbol": symbol,
        "type": "risk_downgrade",
        "from": prior_risk,
        "to": "LOW",
        "component_risks_before": prior_components,
        "component_risk_changes": component_changes,
        "evidence_before": evidence_before,
        "evidence_after": evidence_after,
        "reason": reason,
    })


def validate_stock_batch(
        data,
        expected_candidates,
        minimum_sources=2,
        allowed_risk_event_ids=None,
):
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("Stock batch response is missing a results array.")

    expected_symbols = [str(row["Symbol"]).upper() for row in expected_candidates]
    returned_symbols = [str(item.get("symbol", "")).upper() for item in results]
    if returned_symbols != expected_symbols:
        raise ValueError(
            f"Stock batch symbols/order mismatch: expected {expected_symbols}, "
            f"received {returned_symbols}"
        )

    for expected, result in zip(expected_candidates, results):
        symbol = str(result.get("symbol", "")).upper()

        # QVM rank is authoritative Python-owned input metadata.
        # Do not reject otherwise valid research if Gemini reproduces it incorrectly.
        result["qvm_rank"] = int(expected["QVM Rank"])
        result["eligible"] = True
        result["eligibility_reason"] = None

        research_status = str(
            result.get("research_status", "COMPLETE")
        ).strip().upper()
        if research_status not in {"COMPLETE", "INCOMPLETE"}:
            raise ValueError(
                f"{symbol} has invalid research_status {research_status!r}."
            )
        result["research_status"] = research_status
        if research_status == "INCOMPLETE":
            reason = str(
                result.get("research_incomplete_reason")
                or "required current evidence was not established"
            ).strip()
            if incomplete_identity_conflicts_with_current_market_data(
                reason,
                expected,
            ):
                reason = (
                    "issuer identity conflict: Python has current market data "
                    "for this security; verify its current issuer status, "
                    "including any spin-off, relisting, or reorganization, "
                    "using an official current source"
                )
            raise ValueError(f"{symbol} research incomplete: {reason}")

        reversal_risk = str(result.get("reversal_risk", "")).upper()
        if reversal_risk not in ALLOWED_REVERSAL_RISKS:
            raise ValueError(f"{symbol} has invalid reversal_risk {reversal_risk!r}.")
        component_risks = {}
        for field in ("business_reversal_risk", "entry_reversal_risk"):
            component = str(result.get(field) or reversal_risk).strip().upper()
            if component not in ALLOWED_REVERSAL_RISKS:
                raise ValueError(f"{symbol} has invalid {field} {component!r}.")
            component_risks[field] = component
            result[field] = component

        for field in ("business_concentration", "binary_event_risk"):
            level = str(result.get(field) or "LOW").strip().upper()
            if level == "MEDIUM":
                level = "MODERATE"
            if level not in {"LOW", "MODERATE", "HIGH"}:
                raise ValueError(f"{symbol} has invalid {field} {level!r}.")
            result[field] = level

        benchmark_outlook = str(
            result.get("benchmark_outperformance_outlook") or "UNCERTAIN"
        ).strip().upper()
        if benchmark_outlook not in {"LIKELY", "UNCERTAIN", "UNLIKELY"}:
            raise ValueError(
                f"{symbol} has invalid benchmark_outperformance_outlook "
                f"{benchmark_outlook!r}."
            )
        result["benchmark_outperformance_outlook"] = benchmark_outlook

        continuation_strength = str(
            result.get("continuation_strength") or ""
        ).strip().upper()
        if continuation_strength not in {"STRONG", "ADEQUATE", "WEAK"}:
            raise ValueError(
                f"{symbol} has invalid continuation_strength "
                f"{continuation_strength!r}."
            )
        result["continuation_strength"] = continuation_strength

        # The combined selection label is conservatively floored by both
        # component judgments, preventing a low overall label from hiding a
        # material business or entry-specific vulnerability.
        reversal_risk = max(
            [reversal_risk, *component_risks.values()],
            key=lambda level: REVERSAL_RISK_ORDER[level],
        )
        result["reversal_risk"] = reversal_risk
        catalyst_dependence = str(
            result.get("catalyst_dependence", "")
        ).strip().upper()

        catalyst_aliases = {
            "MEDIUM": "MODERATE",
        }

        catalyst_dependence = catalyst_aliases.get(
            catalyst_dependence,
            catalyst_dependence
        )

        if catalyst_dependence not in {"LOW", "MODERATE", "HIGH"}:
            raise ValueError(
                f"{symbol} has invalid catalyst_dependence: "
                f"{result.get('catalyst_dependence')!r}"
            )

        result["catalyst_dependence"] = catalyst_dependence

        mechanism_status = str(result.get("mechanism_status", "")).upper()
        if mechanism_status not in {
            "NONE", "HYPOTHETICAL", "ACTIVE", "UNUSUALLY_PROBABLE"
        }:
            raise ValueError(
                f"{symbol} has invalid mechanism_status {mechanism_status!r}."
            )
        result["mechanism_status"] = mechanism_status

        normalization_probability = str(
            result.get("normalization_probability", "")
        ).upper()
        if normalization_probability not in {
            "NOT_APPLICABLE", "NOT_ESTABLISHED", "REASONABLY_PROBABLE",
            "AT_LEAST_AS_LIKELY",
        }:
            raise ValueError(
                f"{symbol} has invalid normalization_probability "
                f"{normalization_probability!r}."
            )
        result["normalization_probability"] = normalization_probability

        raw_continuation_outlook = str(
            result.get("continuation_outlook", "")
        ).strip().upper()

        continuation_outlook = re.sub(
            r"[^A-Z0-9]+",
            "_",
            raw_continuation_outlook,
        ).strip("_")

        allowed_continuation_outlooks = {
            "CONTINUATION_MORE_LIKELY",
            "REVERSAL_AT_LEAST_AS_LIKELY",
            "THESIS_BROKEN",
        }

        negative_continuation_markers = {
            "NOT",
            "UNLIKELY",
            "LESS_LIKELY",
            "NO_CONTINUATION",
        }

        if (
                continuation_outlook not in allowed_continuation_outlooks
                and "CONTINUATION" in continuation_outlook
                and not any(
            marker in continuation_outlook
            for marker in negative_continuation_markers
        )
        ):
            print(
                f"Normalized continuation_outlook for {symbol}: "
                f"{raw_continuation_outlook!r} -> "
                "'CONTINUATION_MORE_LIKELY'."
            )
            runtime_reconciliation_diagnostics.append({
                "symbol": symbol,
                "type": "enum_alias_normalization",
                "field": "continuation_outlook",
                "from": raw_continuation_outlook,
                "to": "CONTINUATION_MORE_LIKELY",
            })
            continuation_outlook = "CONTINUATION_MORE_LIKELY"

        if continuation_outlook not in allowed_continuation_outlooks:
            raise ValueError(
                f"{symbol} has invalid continuation_outlook "
                f"{raw_continuation_outlook!r}."
            )

        result["continuation_outlook"] = continuation_outlook

        risk_basis = str(result.get("risk_basis", "")).upper()
        if risk_basis not in {
            "NONE", "NORMALIZED_OPERATING_DETERIORATION",
            "TEMPORARY_DRIVER_NORMALIZATION",
            "NONRECURRING_COMPARISON_ONLY",
        }:
            raise ValueError(f"{symbol} has invalid risk_basis {risk_basis!r}.")
        result["risk_basis"] = risk_basis

        industry_group = str(result.get("industry_group", "")).strip().upper()
        if not industry_group:
            raise ValueError(f"{symbol} has no industry_group.")
        result["industry_group"] = industry_group

        primary_risk_event_id = result.get("primary_risk_event_id")
        if primary_risk_event_id is not None:
            primary_risk_event_id = str(primary_risk_event_id).strip() or None
        if (
                primary_risk_event_id
                and allowed_risk_event_ids is not None
                and primary_risk_event_id not in allowed_risk_event_ids
        ):
            raise ValueError(
                f"{symbol} has unknown primary_risk_event_id "
                f"{primary_risk_event_id!r}; use an exact event_id from "
                "MARKET_CONTEXT.active_risk_events or null."
            )
        result["primary_risk_event_id"] = primary_risk_event_id

        risk_exposure_group = result.get("risk_exposure_group")
        if risk_exposure_group is not None:
            risk_exposure_group = re.sub(
                r"[^A-Z0-9]+", "_", str(risk_exposure_group).upper()
            ).strip("_") or None
        if primary_risk_event_id and not risk_exposure_group:
            raise ValueError(
                f"{symbol} has a primary risk event without risk_exposure_group."
            )
        result["risk_exposure_group"] = risk_exposure_group

        operating_evidence = str(
            result.get("current_operating_evidence", "")
        ).strip()
        if bool(result.get("eligible", True)) and not operating_evidence:
            raise ValueError(f"{symbol} has no current_operating_evidence.")
        result["current_operating_evidence"] = operating_evidence

        for field in ("business_description", "industry_context"):
            value = " ".join(str(result.get(field, "")).split())
            if not value:
                raise ValueError(f"{symbol} has no {field}.")
            result[field] = value

        crypto_dependence = normalize_crypto_dependence(result)
        if crypto_dependence != str(
                result.get("crypto_dependence") or ""
        ).strip().upper():
            print(
                f"Normalized crypto_dependence for {symbol} to "
                f"{crypto_dependence} without another Gemini call."
            )
        result["crypto_dependence"] = crypto_dependence

        temporary_drivers = result.get("temporary_drivers") or []
        if not isinstance(temporary_drivers, list):
            raise ValueError(f"{symbol} temporary_drivers must be an array.")
        if temporary_drivers:
            if (
                    normalization_probability == "NOT_APPLICABLE"
                    and risk_basis != "TEMPORARY_DRIVER_NORMALIZATION"
            ):
                # Temporary-driver metadata is irrelevant when another risk
                # basis owns the classification. Clear it locally instead of
                # asking Gemini to rewrite the same researched result.
                temporary_drivers = []
                result["temporary_drivers"] = []
                print(
                    f"Cleared inapplicable temporary_drivers for {symbol}."
                )
            elif (
                    normalization_probability == "NOT_APPLICABLE"
                    and risk_basis == "TEMPORARY_DRIVER_NORMALIZATION"
                    and reversal_risk in {"MINIMAL", "LOW"}
            ):
                normalization_probability = "NOT_ESTABLISHED"
                result["normalization_probability"] = normalization_probability
                if mechanism_status == "NONE":
                    mechanism_status = "HYPOTHETICAL"
                    result["mechanism_status"] = mechanism_status
                print(
                    f"Normalized hypothetical temporary-driver fields for "
                    f"{symbol}."
                )

        explanation = re.sub(
            r"\s*\[(?:cite|source|citation):[^\]]+\]",
            "",
            str(result.get("explanation", "")),
            flags=re.IGNORECASE,
        ).strip()

        if not explanation:
            raise ValueError(f"{symbol} has no explanation.")

        result["explanation"] = explanation
        if not str(result.get("reversal_mechanism", "")).strip():
            raise ValueError(f"{symbol} has no reversal mechanism.")

        if (
                risk_basis == "NONE"
                and not temporary_drivers
                and catalyst_dependence == "LOW"
                and describes_unmodeled_temporary_market_support(result)
        ):
            raise ValueError(
                f"{symbol} describes current results as supported by an "
                "unusually favorable operating-market condition but omits "
                "the corresponding temporary-driver risk fields."
            )

        raw_risk_materiality = result.get("risk_materiality")

        # LOW and MINIMAL already establish that no qualifying material
        # reversal was found. Gemini frequently emits null here even though
        # the schema asks for LOW. This is a safe deterministic normalization,
        # not a new research conclusion, and avoids repeating all searches just
        # to repair one enum field.
        if raw_risk_materiality is None and reversal_risk in {"MINIMAL", "LOW"}:
            raw_risk_materiality = "LOW"
            print(
                f"Normalized null risk_materiality to LOW for {symbol}."
            )

        risk_materiality = str(raw_risk_materiality or "").strip().upper()

        risk_materiality = {
            "MEDIUM": "MODERATE",
        }.get(risk_materiality, risk_materiality)

        if risk_materiality not in {"LOW", "MODERATE", "HIGH"}:
            raise ValueError(
                f"{symbol} has invalid risk_materiality: "
                f"{raw_risk_materiality!r}"
            )

        result["risk_materiality"] = risk_materiality

        raw_horizon = result.get("risk_time_horizon")
        if raw_horizon is None:
            risk_time_horizon = None
        else:
            risk_time_horizon = re.sub(
                r"[^A-Z0-9]+",
                "_",
                str(raw_horizon).strip().upper()
            ).strip("_")

            risk_time_horizon = {
                "0_3_MONTH": "0_3_MONTHS",
                "3_6_MONTH": "3_6_MONTHS",
                "6_12_MONTH": "6_12_MONTHS",
                "OVER_12_MONTHS": "LONGER",
                "MORE_THAN_12_MONTHS": "LONGER",
            }.get(risk_time_horizon, risk_time_horizon)

        allowed_horizons = {
            None,
            "0_3_MONTHS",
            "3_6_MONTHS",
            "6_12_MONTHS",
            "LONGER",
        }

        if risk_time_horizon not in allowed_horizons:
            raise ValueError(
                f"{symbol} has invalid risk_time_horizon: "
                f"{result.get('risk_time_horizon')!r}"
            )

        result["risk_time_horizon"] = risk_time_horizon

        # reversal_evidence is derived output, not another model obligation.
        # Build it from the three independently validated causal-chain fields.
        reversal_evidence = None
        if reversal_risk in {
            "MODERATE", "ELEVATED", "SEVERE"
        }:
            evidence_parts = [
                str(result.get(field) or "").strip()
                for field in (
                    "current_fact", "probability_evidence", "material_effect"
                )
            ]
            if all(evidence_parts):
                # Keep the synthesized field within the prompt's 35-word cap
                # while retaining content from every required component.
                reversal_evidence = " ".join(
                    " ".join(part.split()[:11])
                    for part in evidence_parts
                )
        result["reversal_evidence"] = reversal_evidence

        evidence_required_risks = {
            "MODERATE",
            "ELEVATED",
            "SEVERE",
        }

        probability_indicator_type = re.sub(
            r"[^A-Z0-9]+", "_",
            str(result.get("probability_indicator_type") or "NONE").upper(),
        ).strip("_")
        allowed_probability_indicators = {
            "NONE", "GUIDANCE_REDUCTION", "ORDER_CONTRACTION",
            "UTILIZATION_DECLINE", "PRICE_OR_MARGIN_COMPRESSION",
            "CAPACITY_INCREASE", "CONTRACT_EXPIRY", "INVENTORY_CHANGE",
            "REGULATORY_ACTION", "FORWARD_MARKET_CHANGE",
            "OTHER_CURRENT_INDICATOR",
        }
        if probability_indicator_type in NON_PROBABILITY_INDICATOR_ALIASES:
            print(
                f"Normalized non-probability indicator for {symbol}: "
                f"{probability_indicator_type}->NONE."
            )
            probability_indicator_type = "NONE"
        if probability_indicator_type not in allowed_probability_indicators:
            raise ValueError(
                f"{symbol} has invalid probability_indicator_type "
                f"{probability_indicator_type!r}."
            )
        result["probability_indicator_type"] = probability_indicator_type

        probability_basis = re.sub(
            r"[^A-Z0-9]+", "_",
            str(result.get("probability_basis") or "").upper(),
        ).strip("_")
        probability_basis = {
            "COMPANY_REPORTED_OPERATING_CHANGE": "COMPANY_REPORTED_CHANGE",
            "CURRENT_OPERATING_CHANGE": "COMPANY_REPORTED_CHANGE",
            "FORWARD_MARKET_CHANGE": "OBSERVABLE_MARKET_CHANGE",
            "OBSERVABLE_FORWARD_MARKET_CHANGE": "OBSERVABLE_MARKET_CHANGE",
            "REGULATORY_ACTION": "REGULATORY_OR_CONTRACT_ACTION",
            "CONTRACT_ACTION": "REGULATORY_OR_CONTRACT_ACTION",
            "ANALYST_FORECAST": "EXTERNAL_EXPECTATION_ONLY",
            "EXTERNAL_FORECAST": "EXTERNAL_EXPECTATION_ONLY",
            "NO_CURRENT_EVIDENCE": "NONE",
        }.get(probability_basis, probability_basis)
        if (
                probability_basis == "NONRECURRING_COMPARISON_ONLY"
                and reversal_risk in {"MINIMAL", "LOW"}
        ):
            # This is a risk_basis value, not probability evidence. For an
            # already LOW/MINIMAL result it unambiguously means no qualifying
            # probability basis was established, so repair the misplaced enum
            # locally instead of spending a Gemini structural-repair call.
            probability_basis = "NONE"
            print(
                f"Normalized misplaced nonrecurring probability_basis to "
                f"NONE for {symbol}."
            )
        if (
                not probability_basis
                and probability_indicator_type == "NONE"
                and reversal_risk in {"MINIMAL", "LOW"}
        ):
            probability_basis = "NONE"
            print(f"Defaulted missing probability_basis to NONE for {symbol}.")
        allowed_probability_bases = {
            "COMPANY_REPORTED_CHANGE",
            "OBSERVABLE_MARKET_CHANGE",
            "REGULATORY_OR_CONTRACT_ACTION",
            "EXTERNAL_EXPECTATION_ONLY",
            "NONE",
        }
        if probability_basis not in allowed_probability_bases:
            raise ValueError(
                f"{symbol} has invalid probability_basis "
                f"{probability_basis!r}."
            )
        result["probability_basis"] = probability_basis

        exposure_floor = bool(
            cautious_exposure_floor_applies(
                result,
                cautious_exposure_floor_enabled,
                cautious_exposure_floor_min_dependence,
            )
            and (
                probability_indicator_type == "NONE"
                or probability_basis in {"EXTERNAL_EXPECTATION_ONLY", "NONE"}
                or normalization_probability == "NOT_ESTABLISHED"
            )
        )
        if exposure_floor:
            prior_risk = reversal_risk
            reversal_risk = "MODERATE"
            mechanism_status = "HYPOTHETICAL"
            normalization_probability = "NOT_ESTABLISHED"
            continuation_outlook = "CONTINUATION_MORE_LIKELY"
            probability_indicator_type = "NONE"
            probability_basis = "NONE"
            if risk_materiality == "LOW":
                risk_materiality = "MODERATE"
            result.update({
                "reversal_risk": reversal_risk,
                "mechanism_status": mechanism_status,
                "normalization_probability": normalization_probability,
                "continuation_outlook": continuation_outlook,
                "probability_indicator_type": probability_indicator_type,
                "probability_basis": probability_basis,
                "risk_materiality": risk_materiality,
                "current_fact": None,
                "probability_evidence": None,
            })
            reversal_evidence = " ".join(
                (" ".join(str(temporary_drivers[0]).split()[:12]),
                 " ".join(str(result.get("material_effect")).split()[:18]))
            ).strip()
            result["reversal_evidence"] = reversal_evidence
            print(
                f"Applied cautious temporary-dependence floor to {symbol}: "
                f"{prior_risk}->MODERATE without another Gemini call."
            )
            runtime_reconciliation_diagnostics.append({
                "symbol": symbol,
                "type": "cautious_temporary_dependence_floor",
                "from": prior_risk,
                "to": "MODERATE",
            })

        primary_reversal_channel = re.sub(
            r"[^A-Z0-9]+", "_",
            str(result.get("primary_reversal_channel") or "").upper(),
        ).strip("_")
        result["primary_reversal_channel"] = primary_reversal_channel or None
        entry_or_concentration_floor = bool(
            REVERSAL_RISK_ORDER.get(reversal_risk, 0)
            >= REVERSAL_RISK_ORDER["MODERATE"]
            and (
                REVERSAL_RISK_ORDER.get(
                    result.get("entry_reversal_risk"), 0
                ) >= REVERSAL_RISK_ORDER["MODERATE"]
                or result.get("business_concentration") in {"MODERATE", "HIGH"}
                or result.get("binary_event_risk") in {"MODERATE", "HIGH"}
            )
            and primary_reversal_channel in {
                "CATALYST_EXHAUSTION", "VALUATION_RERATING", "BINARY_EVENT",
                "MOMENTUM_FRAGILITY", "FUNDAMENTAL_DETERIORATION",
            }
            and str(result.get("material_effect") or "").strip()
        )
        if entry_or_concentration_floor and not reversal_evidence:
            reversal_evidence = " ".join(
                part for part in (
                    str(result.get("reversal_mechanism") or "").strip(),
                    str(result.get("material_effect") or "").strip(),
                ) if part
            )
            result["reversal_evidence"] = reversal_evidence
            if risk_materiality == "LOW":
                risk_materiality = "MODERATE"
                result["risk_materiality"] = risk_materiality

        reconciliation_reason = None
        if (
                reversal_risk in evidence_required_risks
                and not exposure_floor
                and not entry_or_concentration_floor
        ):
            if (
                    risk_basis == "TEMPORARY_DRIVER_NORMALIZATION"
                    and has_nonrecurring_temporary_item(temporary_drivers)
            ):
                # Refunds, settlements and similar comparison adjustments are
                # not operating catalysts. This correction uses the existing
                # researched fields and must not trigger another search.
                risk_basis = "NONRECURRING_COMPARISON_ONLY"
                temporary_drivers = []
                result["risk_basis"] = risk_basis
                result["temporary_drivers"] = temporary_drivers
                reconciliation_reason = (
                    "a nonrecurring comparison item was treated as a temporary "
                    "operating driver"
                )
            elif probability_indicator_type == "NONE":
                reconciliation_reason = (
                    "no qualifying current probability indicator"
                )
            elif probability_basis in {"EXTERNAL_EXPECTATION_ONLY", "NONE"}:
                reconciliation_reason = (
                    "probability basis does not establish a current adverse change"
                )
            elif risk_basis == "NONRECURRING_COMPARISON_ONLY":
                reconciliation_reason = (
                    "risk is based only on a nonrecurring comparison"
                )
            elif mechanism_status not in {"ACTIVE", "UNUSUALLY_PROBABLE"}:
                reconciliation_reason = (
                    "mechanism is not active or unusually probable"
                )
            elif (
                    risk_basis == "TEMPORARY_DRIVER_NORMALIZATION"
                    and normalization_probability not in {
                        "REASONABLY_PROBABLE", "AT_LEAST_AS_LIKELY"
                    }
            ):
                reconciliation_reason = (
                    "temporary-driver normalization probability is not established"
                )

        if reconciliation_reason:
            downgrade_unproven_material_risk(
                result, symbol, reconciliation_reason
            )
            reversal_risk = result["reversal_risk"]
            mechanism_status = result["mechanism_status"]
            normalization_probability = result["normalization_probability"]
            continuation_outlook = result["continuation_outlook"]
            probability_indicator_type = result["probability_indicator_type"]
            risk_materiality = result["risk_materiality"]
            risk_time_horizon = result["risk_time_horizon"]
            reversal_evidence = result["reversal_evidence"]

        if (
                reversal_risk == "ELEVATED"
                and not reconciliation_reason
                and (
                    continuation_outlook != "REVERSAL_AT_LEAST_AS_LIKELY"
                    or (
                        risk_basis == "TEMPORARY_DRIVER_NORMALIZATION"
                        and normalization_probability != "AT_LEAST_AS_LIKELY"
                    )
                )
        ):
            result["reversal_risk"] = "MODERATE"
            component_changes = align_component_risks_to_authoritative(
                result, "MODERATE"
            )
            if component_changes:
                runtime_reconciliation_diagnostics.append({
                    "symbol": symbol,
                    "type": "component_risk_alignment",
                    "to": "MODERATE",
                    "component_risk_changes": component_changes,
                    "reason": "overall ELEVATED risk was reconciled to MODERATE",
                })
            result["continuation_outlook"] = "CONTINUATION_MORE_LIKELY"
            if risk_basis == "TEMPORARY_DRIVER_NORMALIZATION":
                result["normalization_probability"] = "REASONABLY_PROBABLE"
                normalization_probability = "REASONABLY_PROBABLE"
            reversal_risk = "MODERATE"
            continuation_outlook = "CONTINUATION_MORE_LIKELY"
            print(
                f"Reconciled {symbol} ELEVATED to MODERATE because the "
                "at-least-as-likely threshold was not established."
            )

        if reversal_risk in evidence_required_risks:
            required_fields = (
                ("material_effect",)
                if exposure_floor or entry_or_concentration_floor
                else ("current_fact", "probability_evidence", "material_effect")
            )
            missing_evidence_fields = [
                field for field in required_fields
                if not str(result.get(field) or "").strip()
            ]
            if missing_evidence_fields:
                raise ValueError(
                    f"{symbol} assessed as {reversal_risk} without "
                    + ", ".join(missing_evidence_fields)
                    + "."
                )
            if not reversal_evidence:
                raise ValueError(
                    f"{symbol} assessed as {reversal_risk} reversal risk without "
                    "specific reversal_evidence."
                )

            if risk_materiality not in {"MODERATE", "HIGH"}:
                raise ValueError(
                    f"{symbol} assessed as {reversal_risk} reversal risk with "
                    f"risk_materiality={risk_materiality}."
                )

            if risk_time_horizon not in {
                "0_3_MONTHS",
                "3_6_MONTHS",
                "6_12_MONTHS",
            }:
                raise ValueError(
                    f"{symbol} assessed as {reversal_risk} reversal risk with "
                    f"risk_time_horizon={risk_time_horizon!r}."
                )

        if (
                risk_basis == "NONRECURRING_COMPARISON_ONLY"
                and reversal_risk not in {"MINIMAL", "LOW"}
                and not entry_or_concentration_floor
        ):
            raise ValueError(
                f"{symbol} uses only a non-recurring comparison to justify "
                f"{reversal_risk} risk."
            )

        if reversal_risk == "MINIMAL":
            if not (
                mechanism_status == "NONE"
                and catalyst_dependence == "LOW"
                and continuation_outlook == "CONTINUATION_MORE_LIKELY"
                and risk_basis == "NONE"
            ):
                raise ValueError(f"{symbol} MINIMAL fields violate risk mapping.")
        elif reversal_risk == "LOW":
            if mechanism_status not in {"NONE", "HYPOTHETICAL"}:
                raise ValueError(f"{symbol} LOW fields violate mechanism mapping.")
            if continuation_outlook != "CONTINUATION_MORE_LIKELY":
                raise ValueError(f"{symbol} LOW fields violate outlook mapping.")
        elif reversal_risk == "MODERATE":
            if (
                    not exposure_floor
                    and mechanism_status not in {"ACTIVE", "UNUSUALLY_PROBABLE"}
            ):
                raise ValueError(f"{symbol} MODERATE fields violate mechanism mapping.")
            if continuation_outlook != "CONTINUATION_MORE_LIKELY":
                result["continuation_outlook"] = "CONTINUATION_MORE_LIKELY"
                continuation_outlook = "CONTINUATION_MORE_LIKELY"
                print(
                    f"Normalized MODERATE continuation_outlook for {symbol}."
                )
            if risk_basis not in {
                "NORMALIZED_OPERATING_DETERIORATION",
                "TEMPORARY_DRIVER_NORMALIZATION",
            }:
                raise ValueError(f"{symbol} MODERATE has an invalid risk basis.")
        elif reversal_risk == "ELEVATED":
            if mechanism_status not in {"ACTIVE", "UNUSUALLY_PROBABLE"}:
                raise ValueError(f"{symbol} ELEVATED fields violate mechanism mapping.")
            if continuation_outlook != "REVERSAL_AT_LEAST_AS_LIKELY":
                raise ValueError(f"{symbol} ELEVATED fields violate outlook mapping.")
        elif reversal_risk == "SEVERE":
            if continuation_outlook != "THESIS_BROKEN":
                raise ValueError(f"{symbol} SEVERE fields violate outlook mapping.")

        if risk_basis == "TEMPORARY_DRIVER_NORMALIZATION":
            if not temporary_drivers:
                raise ValueError(
                    f"{symbol} normalization risk has no temporary driver."
                )
            if (
                reversal_risk in evidence_required_risks
                and not exposure_floor
                and (
                normalization_probability not in {
                    "REASONABLY_PROBABLE", "AT_LEAST_AS_LIKELY"
                }
                )
            ):
                raise ValueError(
                    f"{symbol} material normalization risk lacks sufficient "
                    "normalization probability."
                )

        sources = result.get("sources")
        if isinstance(sources, list):
            original_source_count = len(sources)
            result["sources"] = prioritize_stock_sources(
                sources, max_stock_sources
            )
            sources = result["sources"]
        else:
            original_source_count = 0
        if original_source_count > len(sources or []):
            print(
                f"Trimmed {symbol} sources to the {max_stock_sources} "
                "most relevant distinct entries."
            )

        required_source_count = max(
            minimum_sources,
            2 if reversal_risk in evidence_required_risks else 1,
        )
        validate_sources(
            sources,
            symbol,
            minimum=required_source_count,
            maximum=max_stock_sources,
        )

        unique_source_urls = {
            str(source["url"]).strip()
            for source in result["sources"]
        }
        print(
            f"Validated JSON sources [{symbol}]: "
            f"{len(unique_source_urls)} distinct URLs"
        )


def validate_shared_event_consistency(result, comparison_results):
    """Require an explicit structural reason for a shared-event divergence."""
    normalize_group = lambda value: re.sub(
        r"[^A-Z0-9]+", "_", str(value or "").upper()
    ).strip("_")
    event_id = normalize_group(result.get("primary_risk_event_id"))
    exposure_group = normalize_group(result.get("risk_exposure_group"))
    industry_group = normalize_group(result.get("industry_group"))
    result_risk = str(result.get("reversal_risk") or "").upper()
    baseline_fields = (
        "mechanism_status",
        "normalization_probability",
    )
    for other in comparison_results:
        if other is result or other.get("symbol") == result.get("symbol"):
            continue
        other_exposure_group = normalize_group(other.get("risk_exposure_group"))
        other_industry_group = normalize_group(other.get("industry_group"))
        same_explicit_exposure = bool(
            exposure_group and exposure_group == other_exposure_group
        )
        missing_group_but_same_industry = bool(
            industry_group
            and industry_group == other_industry_group
            and (not exposure_group or not other_exposure_group)
        )
        if not (same_explicit_exposure or missing_group_but_same_industry):
            continue
        other_event_id = normalize_group(other.get("primary_risk_event_id"))
        comparison_group = exposure_group or other_exposure_group or industry_group

        if (
            not event_id
            and other_event_id
            and result_risk in {"MINIMAL", "LOW"}
            and not str(result.get("company_difference") or "").strip()
        ):
            raise ValueError(
                f"{result.get('symbol')} omits documented shared event "
                f"{other_event_id} for comparable exposure {comparison_group}; "
                "current research must establish the event treatment or a "
                "sourced company_difference."
            )

        if not event_id or other_event_id != event_id:
            continue
        differences = [
            field for field in baseline_fields
            if other.get(field) != result.get(field)
        ]
        if differences and not (
            str(result.get("company_difference") or "").strip()
            or str(other.get("company_difference") or "").strip()
        ):
            decision_fields = (
                "reversal_risk", "risk_basis", "catalyst_dependence"
            )
            decision_differences = [
                field for field in decision_fields
                if other.get(field) != result.get(field)
            ]
            if decision_differences:
                raise ValueError(
                    f"{result.get('symbol')} differs from "
                    f"{other.get('symbol')} on shared event {event_id} "
                    f"({', '.join(differences + decision_differences)}) "
                    "without a sourced company_difference."
                )
            print(
                f"Accepted non-decision shared-event label difference for "
                f"{result.get('symbol')} versus {other.get('symbol')} on "
                f"{event_id}: {', '.join(differences)}."
            )
            runtime_reconciliation_diagnostics.append({
                "symbol": result.get("symbol"),
                "type": "accepted_nondecision_shared_event_difference",
                "compared_with": other.get("symbol"),
                "event_id": event_id,
                "fields": differences,
            })


def call_gemini_json(
        client,
        model_primary,
        model_fallback,
        gemini_config,
        prompt,
        stage,
        validator,
        request_budget,
        require_google_search=True,
        required_search_candidates=None,
        allow_partial_stock_results=False,
        max_attempts=None,
        budget_category="general"):
    """Call Gemini, require grounded research, parse JSON, and validate it."""
    models = [model_primary]
    if model_fallback and model_fallback != model_primary:
        models.append(model_fallback)

    attempt_limit = (
        max_transient_api_attempts
        if max_attempts is None else max(1, int(max_attempts))
    )
    request_budget.reserve(stage, category=budget_category)
    last_error = None
    for model_name in models:
        for attempt in range(attempt_limit):
            attempt_prompt = prompt
            metadata = None
            try:
                request_budget.record_api_attempt(
                    f"{stage} ({model_name}, attempt {attempt + 1})",
                    category=budget_category,
                )
                response = client.models.generate_content(
                    model=model_name,
                    config=gemini_config,
                    contents=attempt_prompt
                )
                response_text = getattr(response, "text", None)
                if not response_text or not response_text.strip():
                    raise ValueError("Empty response from Gemini.")

                metadata = extract_gemini_metadata(response)
                print_gemini_metadata(stage, metadata)

                used_search = bool(metadata["search_queries"]) or (
                    metadata["tool_tokens"] > 0
                )
                if require_google_search and not used_search:
                    raise ValueError(
                        "Gemini returned an ungrounded response without Google Search."
                    )

                # When query metadata is exposed, verify company-specific search
                # coverage. Some API responses expose tool tokens but omit query
                # strings, so validated sources remain the fallback evidence.
                if required_search_candidates and metadata["search_queries"]:
                    query_text = " ".join(
                        metadata["search_queries"]
                    ).upper()
                    normalized_query_text = re.sub(
                        r"[^A-Z0-9]+", " ", query_text
                    )
                    missing_symbols = []
                    company_suffixes = {
                        "INC", "INCORPORATED", "CORP", "CORPORATION", "LTD",
                        "LIMITED", "PLC", "LLC", "CO", "COMPANY"
                    }
                    for candidate in required_search_candidates:
                        symbol = str(candidate["Symbol"]).upper()
                        symbol_found = re.search(
                            rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])",
                            query_text
                        ) is not None
                        name_tokens = [
                            token
                            for token in re.sub(
                                r"[^A-Z0-9]+",
                                " ",
                                str(candidate.get("Name", "")).upper()
                            ).split()
                            if token not in company_suffixes
                        ]
                        company_name = " ".join(name_tokens)
                        name_found = bool(company_name) and (
                            company_name in normalized_query_text
                        )
                        if not symbol_found and not name_found:
                            missing_symbols.append(symbol)
                    if missing_symbols:
                        if allow_partial_stock_results:
                            print(
                                "Warning: Gemini did not expose query metadata "
                                "for these candidates; retaining their results "
                                "and enforcing source validation instead: "
                                + ", ".join(missing_symbols)
                            )
                        else:
                            raise ValueError(
                                "Gemini did not expose company-specific searches for: "
                                + ", ".join(missing_symbols)
                            )

                delimited_results = (
                    extract_delimited_stock_results(response_text)
                    if allow_partial_stock_results else []
                )
                if delimited_results:
                    delimited_results = deduplicate_stock_results(
                        delimited_results,
                        required_search_candidates or [],
                        stage=stage,
                    )
                    data = {"results": delimited_results}
                    print(
                        "Recovered independently delimited stock results: "
                        + ", ".join(
                            str(result.get("symbol", "")).upper()
                            for result in delimited_results
                        )
                    )
                else:
                    try:
                        data = parse_json_response(response_text)
                    except json.JSONDecodeError as parse_error:
                        if not allow_partial_stock_results:
                            raise

                        partial_results = extract_partial_stock_results(
                            response_text
                        )
                        recovered_symbols = [
                            str(result.get("symbol", "")).upper()
                            for result in partial_results
                            if str(result.get("symbol", "")).strip()
                        ]
                        print(
                            "Malformed stock-batch JSON; recovered "
                            f"{len(partial_results)} individual result(s): "
                            + (", ".join(recovered_symbols) or "none")
                        )
                        print(
                            "The recovered results will be validated and cached; "
                            "only missing or invalid stocks will be retried. "
                            f"Original JSON error: {parse_error}"
                        )
                        data = {"results": partial_results}

                validator(data)
                gemini_attempt_diagnostics.append({
                    "stage": stage,
                    "model": model_name,
                    "attempt": attempt + 1,
                    "category": budget_category,
                    "search_enabled": bool(require_google_search),
                    "status": "SUCCESS",
                    "prompt_tokens": metadata.get("prompt_tokens"),
                    "tool_tokens": metadata.get("tool_tokens"),
                    "cached_tokens": metadata.get("cached_tokens"),
                    "thinking_tokens": metadata.get("thinking_tokens"),
                    "output_tokens": metadata.get("output_tokens"),
                    "total_tokens": metadata.get("total_tokens"),
                    "search_queries": metadata.get("search_queries", []),
                    "grounding_chunks_exposed": len(metadata.get("grounding_chunks", [])),
                })
                return data, model_name, metadata
            except Exception as exc:
                last_error = exc
                error_text = str(exc).upper()
                error_type = type(exc).__name__.upper()
                error_code = getattr(exc, "code", None)

                daily_quota_exhausted = (
                    "GENERATE_CONTENT_FREE_TIER_REQUESTS" in error_text
                    or "GENERATEREQUESTSPERDAYPERPROJECTPERMODEL" in error_text
                    or "REQUESTS PER DAY" in error_text
                    or "DAILY QUOTA" in error_text
                )

                transient_error = (
                    not daily_quota_exhausted
                    and (
                        error_code in {408, 429, 500, 502, 503, 504}
                        or "EMPTY RESPONSE FROM GEMINI" in error_text
                        or "RESOURCE_EXHAUSTED" in error_text
                        or "TOO MANY REQUESTS" in error_text
                        or "UNAVAILABLE" in error_text
                        or "TIMEOUT" in error_text
                        or "TIMEOUT" in error_type
                    )
                )

                gemini_attempt_diagnostics.append({
                    "stage": stage,
                    "model": model_name,
                    "attempt": attempt + 1,
                    "category": budget_category,
                    "search_enabled": bool(require_google_search),
                    "status": (
                        "DAILY_QUOTA_EXHAUSTED" if daily_quota_exhausted
                        else "TRANSIENT_ERROR" if transient_error
                        else "ERROR"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                print(
                    f"Error on attempt {attempt + 1}/{attempt_limit} "
                    f"during {stage} with {model_name}: {exc}"
                )

                if daily_quota_exhausted:
                    if model_name != models[-1]:
                        print(
                            f"{model_name} daily quota is exhausted; not "
                            "retrying that model and switching to the "
                            f"fallback model for {stage}."
                        )
                        break
                    raise RuntimeError(
                        "Gemini daily request quota is exhausted for every "
                        f"configured model during {stage}; ending without "
                        "retrying the exhausted model."
                    ) from exc

                if transient_error and attempt < attempt_limit - 1:
                    delay = min(
                        max_transient_backoff_seconds,
                        initial_transient_backoff_seconds * (3 ** attempt),
                    ) + random.uniform(0, transient_backoff_jitter_seconds)
                    print(
                        f"Transient Gemini error; retrying the same model and "
                        f"unchanged request in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue

                # Content, validation, and parsing failures are handled by the
                # outer stock-validation rounds. Do not burn API attempts by
                # retrying them here as if they were transient service errors.
                if allow_partial_stock_results and metadata is not None:
                    print(
                        "Returning an empty partial batch so the outer "
                        "validation round can retry it: " + str(exc)
                    )
                    return {"results": []}, model_name, metadata
                break

        if model_name != models[-1]:
            print(f"Switching to fallback model after {stage} failure.")

    raise RuntimeError(f"Gemini {stage} failed: {last_error}")



CACHE_FILE = cache_file_path("yf_cache.json")
CACHE_EXPIRY_DAYS = 1



# ---------- yfinance CACHE HELPERS ----------
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)

        if not isinstance(cache, dict):
            print(
                f"Warning: {CACHE_FILE} does not contain a JSON object. "
                "Ignoring it."
            )
            return {}

    except (json.JSONDecodeError, OSError) as e:
        print(
            f"Warning: could not read {CACHE_FILE}: {e}. "
            "Ignoring the invalid cache and rebuilding it."
        )
        return {}

    fresh_cache = {}
    now = datetime.now(UTC)

    for ticker, entry in cache.items():
        try:
            ts = datetime.fromisoformat(entry["timestamp"])

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            if now - ts < timedelta(days=CACHE_EXPIRY_DAYS):
                fresh_cache[ticker] = entry
        except (KeyError, TypeError, ValueError):
            continue

    return fresh_cache


def save_cache(cache):
    temp_file = f"{CACHE_FILE}.tmp"

    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(cache, f)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_file, CACHE_FILE)

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)



# ---------- TOP QVM DATAFRAME CACHE ----------
# This cache stores the fully computed ranked QVM candidate pool so that
# Gemini prompt testing can skip the expensive stock-data/QVM pipeline.
# The cache file can be persisted by GitHub Actions in the same way as
# the existing JSON caches.
TOP_QVM_CACHE_FILE = cache_file_path("top_qvm_stocks_cache.pkl")
TOP_QVM_CACHE_EXPIRY_HOURS = 6
# Increment when QVM inputs or scoring semantics change so a prior cached
# ranking cannot bypass the updated calculation.
TOP_QVM_CACHE_VERSION = 6


def load_top_qvm_cache(benchmark_context=None, hurdle_tolerance_pct=0.25):
    """Load cached top-QVM DataFrame if it exists and is still fresh."""

    print(f"Checking top QVM cache: {TOP_QVM_CACHE_FILE}", flush=True)

    if not os.path.exists(TOP_QVM_CACHE_FILE):
        print("Top QVM cache does not exist.", flush=True)
        return None

    print("Top QVM cache exists.", flush=True)

    try:
        print("About to read top QVM pickle...", flush=True)

        cached = pd.read_pickle(TOP_QVM_CACHE_FILE)

        print("Top QVM pickle loaded successfully.", flush=True)

        if not isinstance(cached, dict):
            print("Invalid top QVM cache format. Rebuilding cache.")
            return None

        if cached.get("version") != TOP_QVM_CACHE_VERSION:
            print("Top QVM cache version mismatch. Rebuilding cache.")
            return None

        if benchmark_context:
            expected_symbols = benchmark_context.get("symbols", [])
            cached_symbols = cached.get("benchmark_symbols", [])
            if cached_symbols != expected_symbols:
                print("Top QVM benchmark list changed. Rebuilding cache.")
                return None
            current_hurdle = float(
                benchmark_context.get("hurdle_52_week_return", 0.0)
            )
            cached_hurdle = cached.get("benchmark_hurdle_52_week_return")
            if (
                    cached_hurdle is None
                    or abs(float(cached_hurdle) - current_hurdle)
                    > float(hurdle_tolerance_pct)
            ):
                print("Top QVM benchmark hurdle changed. Rebuilding cache.")
                return None

        created_at = cached.get("created_at")
        if not created_at:
            print("Top QVM cache has no creation timestamp. Rebuilding cache.")
            return None

        created_time = datetime.fromisoformat(created_at)

        if created_time.tzinfo is None:
            created_time = created_time.replace(tzinfo=UTC)

        age = datetime.now(UTC) - created_time

        if age >= timedelta(hours=TOP_QVM_CACHE_EXPIRY_HOURS):
            print(
                f"Top QVM cache is stale "
                f"({age.total_seconds() / 3600:.1f}h old)."
            )
            return None

        df = cached.get("data")

        if not isinstance(df, pd.DataFrame) or df.empty:
            print("Top QVM cache is empty or invalid. Rebuilding cache.")
            return None

        print(
            f"Using cached top {len(df)} QVM stocks "
            f"({age.total_seconds() / 3600:.1f}h old)."
        )

        return df

    except Exception as e:
        print(f"Could not load top QVM cache: {e}")
        return None

def save_top_qvm_cache(df, benchmark_context=None):
    """Save the fully computed top-QVM DataFrame for later Gemini testing."""
    try:
        cache = {
            "version": TOP_QVM_CACHE_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "benchmark_symbols": (
                benchmark_context.get("symbols", []) if benchmark_context else []
            ),
            "benchmark_hurdle_52_week_return": (
                benchmark_context.get("hurdle_52_week_return")
                if benchmark_context else None
            ),
            "data": df.copy()
        }

        pd.to_pickle(cache, TOP_QVM_CACHE_FILE)

        print(f"Saved top QVM cache to {TOP_QVM_CACHE_FILE}.")

    except Exception as e:
        print(f"Could not save top QVM cache: {e}")

def append_qvm_data_yfinance(
        df: pd.DataFrame,
        max_info_calls: int = 500,
        delay: float = 0.5,
        min_3_month_return: float = 0.0,
        benchmark_context=None,
):
    df = df.copy()
    tickers_list = df["Symbol"].tolist()

    # ---- Load cache ----
    cache = load_cache()

    # ---- Download price history (fast, bulk) ----
    print("Downloading price data...")
    price_data = yf.download(
        tickers_list,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True
    )

    data_map = {}
    momentum_scores = {}
    benchmark_returns = (
        benchmark_context.get("returns", {})
        if isinstance(benchmark_context, dict) else {}
    )
    best_benchmark_return = {}
    for metric in (
        "1M Return", "3M Return", "6M Return", "9M Return", "1Y Return"
    ):
        values = [
            returns.get(metric)
            for returns in benchmark_returns.values()
            if returns.get(metric) is not None
        ]
        best_benchmark_return[metric] = max(values) if values else None

    # ---- STEP 1: Compute momentum + extract Price ----
    print("Computing momentum and extracting latest price...")
    for symbol in tickers_list:
        try:
            df_prices = price_data if len(tickers_list) == 1 else price_data[symbol]
            close = df_prices["Close"].dropna()

            if len(close) < 10:
                momentum_scores[symbol] = -np.inf
                data_map[symbol] = {}
                continue

            # Do not label a recent listing's first-available-price return as
            # a one-year return. Approximately 230 sessions tolerates ordinary
            # exchange holidays and isolated missing observations while still
            # requiring close to a full trading year of history.
            ret_1y = (
                ((close.iloc[-1] / close.iloc[0]) - 1) * 100
                if len(close) >= 230 else None
            )
            ret_9m = ((close.iloc[-1] / close.iloc[-189]) - 1) * 100 if len(close) > 189 else None
            ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else None
            ret_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else None
            ret_1m = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 21 else None

            latest_price = float(close.iloc[-1])
            daily_returns = close.pct_change().dropna()
            annualized_volatility = (
                float(daily_returns.std(ddof=0) * np.sqrt(252) * 100)
                if len(daily_returns) >= 20 else None
            )
            downside_returns = daily_returns[daily_returns < 0]
            downside_volatility = (
                float(downside_returns.std(ddof=0) * np.sqrt(252) * 100)
                if len(downside_returns) >= 10 else None
            )
            running_peak = close.cummax()
            drawdowns = close / running_peak - 1
            max_drawdown = float(drawdowns.min() * 100)
            positive_day_pct = float((daily_returns > 0).mean() * 100)

            trend_window = close.tail(min(126, len(close)))
            log_prices = np.log(trend_window.to_numpy(dtype=float))
            x_values = np.arange(len(log_prices), dtype=float)
            if len(log_prices) >= 20 and np.isfinite(log_prices).all():
                slope, intercept = np.polyfit(x_values, log_prices, 1)
                fitted = slope * x_values + intercept
                residual_sum = float(np.square(log_prices - fitted).sum())
                total_sum = float(np.square(log_prices - log_prices.mean()).sum())
                trend_r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
                annualized_trend = float((np.exp(slope * 252) - 1) * 100)
            else:
                trend_r2 = None
                annualized_trend = None

            sma_50 = float(close.tail(min(50, len(close))).mean())
            sma_200 = float(close.tail(min(200, len(close))).mean())
            high_52w = float(close.max())
            distance_50dma = (latest_price / sma_50 - 1) * 100 if sma_50 else None
            distance_200dma = (latest_price / sma_200 - 1) * 100 if sma_200 else None
            distance_52w_high = (
                (latest_price / high_52w - 1) * 100 if high_52w else None
            )
            largest_1d_move = (
                float(daily_returns.abs().max() * 100)
                if not daily_returns.empty else None
            )
            rolling_5d = close.pct_change(5).dropna()
            largest_5d_move = (
                float(rolling_5d.abs().max() * 100)
                if not rolling_5d.empty else None
            )
            prior_2m_monthly = None
            if len(close) > 63:
                prior_2m_total = float(close.iloc[-21] / close.iloc[-63])
                prior_2m_monthly = (prior_2m_total ** 0.5 - 1) * 100
            momentum_acceleration = (
                float(ret_1m - prior_2m_monthly)
                if ret_1m is not None and prior_2m_monthly is not None else None
            )

            stock_returns = {
                "1M Return": ret_1m,
                "3M Return": ret_3m,
                "6M Return": ret_6m,
                "9M Return": ret_9m,
                "1Y Return": ret_1y,
            }
            benchmark_excess = {
                metric: (
                    value - best_benchmark_return[metric]
                    if value is not None
                    and best_benchmark_return.get(metric) is not None
                    else None
                )
                for metric, value in stock_returns.items()
            }

            score = np.nanmean([ret_3m, ret_6m, ret_9m])
            momentum_scores[symbol] = score

            data_map[symbol] = {
                "Price": latest_price,                  # ← Added
                "HistoryDays": len(close),
                "1M Return": ret_1m,
                "3M Return": ret_3m,
                "6M Return": ret_6m,
                "9M Return": ret_9m,
                "1Y Return": ret_1y,
                "AnnualizedVolatility": annualized_volatility,
                "DownsideVolatility": downside_volatility,
                "MaxDrawdown": max_drawdown,
                "PositiveDayPct": positive_day_pct,
                "TrendR2": trend_r2,
                "AnnualizedTrend": annualized_trend,
                "Distance50DMA": distance_50dma,
                "Distance200DMA": distance_200dma,
                "Distance52WHigh": distance_52w_high,
                "Largest1DayMove": largest_1d_move,
                "Largest5DayMove": largest_5d_move,
                "MomentumAcceleration": momentum_acceleration,
                "BenchmarkExcess1M": benchmark_excess["1M Return"],
                "BenchmarkExcess3M": benchmark_excess["3M Return"],
                "BenchmarkExcess6M": benchmark_excess["6M Return"],
                "BenchmarkExcess9M": benchmark_excess["9M Return"],
                "BenchmarkExcess1Y": benchmark_excess["1Y Return"],
            }

        except Exception as exc:
            print(f"Could not calculate price-history metrics for {symbol}: {exc}")
            momentum_scores[symbol] = -np.inf
            data_map[symbol] = {}

    # ---- STEP 2: Apply cheap momentum gate before expensive ticker.info calls ----
    # The final QVM pipeline already requires a known, non-negative 3M return.
    # Enforce that immediately after the single bulk history download so stocks
    # that can never survive the momentum gate do not trigger individual
    # fundamentals requests.
    momentum_eligible_symbols = [
        symbol
        for symbol in tickers_list
        if data_map.get(symbol, {}).get("3M Return") is not None
        and data_map[symbol]["3M Return"] >= min_3_month_return
    ]
    removed_by_3m_gate = len(tickers_list) - len(momentum_eligible_symbols)
    print(
        f"3M return gate >= {min_3_month_return:.2f}%: "
        f"{len(momentum_eligible_symbols)} survivors; "
        f"{removed_by_3m_gate} removed before fundamentals."
    )

    # ---- STEP 3: Select top N survivors for expensive calls ----
    sorted_symbols = sorted(
        momentum_eligible_symbols,
        key=lambda x: momentum_scores[x],
        reverse=True,
    )
    selected_for_info = (
        sorted_symbols if max_info_calls is None else sorted_symbols[:max_info_calls]
    )

    required_cached_fields = {
        "operatingMargins", "revenueGrowth", "earningsGrowth",
        "freeCashflow", "operatingCashflow", "totalDebt",
        "totalCash", "sharesOutstanding",
    }
    cache_hit_symbols = []
    live_fetch_symbols = []
    for symbol in selected_for_info:
        cached_info = (
            cache.get(symbol, {}).get("info", {})
            if isinstance(cache.get(symbol), dict) else {}
        )
        if cached_info and required_cached_fields.issubset(cached_info):
            cache_hit_symbols.append(symbol)
        else:
            live_fetch_symbols.append(symbol)

    print(
        "Fundamentals plan: "
        f"eligible={len(selected_for_info)}, "
        f"cache_hits={len(cache_hit_symbols)}, "
        f"live_fetches={len(live_fetch_symbols)}."
    )

    fundamentals_fetch_failures = []
    fundamentals_apply_failures = []
    fetched_info_by_symbol = {}
    for symbol in cache_hit_symbols:
        fetched_info_by_symbol[symbol] = cache[symbol]["info"]

    if live_fetch_symbols:
        print(
            f"Fetching Yahoo fundamentals for {len(live_fetch_symbols)} "
            "uncached tickers..."
        )
    for symbol in tqdm(
            live_fetch_symbols,
            desc="Yahoo fundamentals",
            disable=not live_fetch_symbols,
    ):
        try:
            ticker_obj = yf.Ticker(symbol)
            raw_info = ticker_obj.info or {}
            info = {
                "sector": raw_info.get("sector"),
                "returnOnEquity": raw_info.get("returnOnEquity"),
                "returnOnAssets": raw_info.get("returnOnAssets"),
                "profitMargins": raw_info.get("profitMargins"),
                "grossMargins": raw_info.get("grossMargins"),
                "operatingMargins": raw_info.get("operatingMargins"),
                "revenueGrowth": raw_info.get("revenueGrowth"),
                "earningsGrowth": raw_info.get("earningsGrowth"),
                "debtToEquity": raw_info.get("debtToEquity"),
                "currentRatio": raw_info.get("currentRatio"),
                "interestCoverage": raw_info.get("interestCoverage"),
                "trailingPE": raw_info.get("trailingPE"),
                "priceToBook": raw_info.get("priceToBook"),
                "pegRatio": raw_info.get("pegRatio"),
                "enterpriseValue": raw_info.get("enterpriseValue"),
                "ebitda": raw_info.get("ebitda"),
                "totalRevenue": raw_info.get("totalRevenue"),
                "freeCashflow": raw_info.get("freeCashflow"),
                "operatingCashflow": raw_info.get("operatingCashflow"),
                "totalDebt": raw_info.get("totalDebt"),
                "totalCash": raw_info.get("totalCash"),
                "sharesOutstanding": raw_info.get("sharesOutstanding"),
            }
            fetched_info_by_symbol[symbol] = info
            cache[symbol] = {
                "info": info,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            time.sleep(delay + random.uniform(0, 0.3))
        except Exception as exc:
            fundamentals_fetch_failures.append({
                "symbol": symbol,
                "error": str(exc),
            })
            print(f"Error retrieving fundamentals for {symbol}: {exc}")

    # ---- STEP 4: Apply cached/fetched info ----
    for symbol in selected_for_info:
        info = fetched_info_by_symbol.get(symbol)
        if not isinstance(info, dict):
            continue
        try:
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")
            revenue = info.get("totalRevenue")
            free_cash_flow = info.get("freeCashflow")
            operating_cash_flow = info.get("operatingCashflow")
            total_debt = info.get("totalDebt")
            total_cash = info.get("totalCash")

            data_map[symbol].update({
                "Sector": info.get("sector"),
                "ROE": info.get("returnOnEquity"),
                "ROA": info.get("returnOnAssets"),
                "ProfitMargin": info.get("profitMargins"),
                "GrossMargin": info.get("grossMargins"),
                "OperatingMargin": info.get("operatingMargins"),
                "RevenueGrowth": info.get("revenueGrowth"),
                "EarningsGrowth": info.get("earningsGrowth"),
                "DebtToEquity": info.get("debtToEquity"),
                "CurrentRatio": info.get("currentRatio"),
                "InterestCoverage": info.get("interestCoverage"),
                "PE": info.get("trailingPE"),
                "PriceToBook": info.get("priceToBook"),
                "PEG": info.get("pegRatio"),
                "EV_EBITDA": (ev / ebitda) if ev and ebitda and ebitda != 0 else None,
                "EV_Revenue": (ev / revenue) if ev and revenue and revenue != 0 else None,
                "FCFMargin": (
                    free_cash_flow / revenue
                    if free_cash_flow is not None and revenue else None
                ),
                "OperatingCashFlowMargin": (
                    operating_cash_flow / revenue
                    if operating_cash_flow is not None and revenue else None
                ),
                "NetDebt": (
                    (total_debt or 0) - (total_cash or 0)
                    if total_debt is not None or total_cash is not None else None
                ),
                "SharesOutstanding": info.get("sharesOutstanding"),
            })
        except Exception as exc:
            fundamentals_apply_failures.append({
                "symbol": symbol,
                "error": str(exc),
            })
            print(f"Error applying fundamentals for {symbol}: {exc}")

    global yfinance_fundamentals_diagnostics
    yfinance_fundamentals_diagnostics = {
        "eligible_symbols": len(selected_for_info),
        "cache_hits": len(cache_hit_symbols),
        "live_fetches": len(live_fetch_symbols),
        "fetch_successes": len(live_fetch_symbols) - len({
            item["symbol"] for item in fundamentals_fetch_failures
        }),
        "fetch_failures": fundamentals_fetch_failures,
        "apply_failures": fundamentals_apply_failures,
        "pipeline_used": True,
    }
    print(
        "Fundamentals summary: "
        f"eligible={yfinance_fundamentals_diagnostics['eligible_symbols']}, "
        f"cache_hits={yfinance_fundamentals_diagnostics['cache_hits']}, "
        f"yahoo_fetches={yfinance_fundamentals_diagnostics['live_fetches']}, "
        f"fetch_successes={yfinance_fundamentals_diagnostics['fetch_successes']}, "
        f"fetch_failures={len(yfinance_fundamentals_diagnostics['fetch_failures'])}, "
        f"apply_failures={len(yfinance_fundamentals_diagnostics['apply_failures'])}."
    )

    # ---- Save cache ----
    save_cache(cache)

    # ---- Map back to df ----
    all_columns = [
        "Sector", "Price", "ROE", "ROA", "ProfitMargin", "GrossMargin",
        "OperatingMargin", "RevenueGrowth", "EarningsGrowth", "FCFMargin",
        "OperatingCashFlowMargin", "NetDebt", "SharesOutstanding",
        "DebtToEquity", "CurrentRatio", "InterestCoverage",
        "PE", "PriceToBook", "PEG", "EV_EBITDA", "EV_Revenue",
        "HistoryDays",
        "1M Return", "3M Return", "6M Return", "9M Return", "1Y Return",
        "AnnualizedVolatility", "DownsideVolatility", "MaxDrawdown",
        "PositiveDayPct", "TrendR2", "AnnualizedTrend", "Distance50DMA",
        "Distance200DMA", "Distance52WHigh", "Largest1DayMove",
        "Largest5DayMove", "MomentumAcceleration",
        "BenchmarkExcess1M", "BenchmarkExcess3M", "BenchmarkExcess6M",
        "BenchmarkExcess9M", "BenchmarkExcess1Y"
    ]

    for col in all_columns:
        df[col] = df["Symbol"].map(lambda x: data_map.get(x, {}).get(col))

    print("Done.")
    return df



def score_qvm(df, top_n=100, weights=None, min_quality=40):
    """
    QVM scorer designed for durable, high-quality compounders.

    Philosophy:
      - Quality is the foundation.
      - Momentum rewards persistent performance, not one-off spikes.
      - Value is a secondary confirmation rather than the primary driver.
      - Penalize excessive leverage.
      - Penalize inconsistent / volatile return paths.
      - Prevent missing data and extreme outliers from distorting scores.

    Strategy:
      1. Strong underlying businesses.
      2. Sustained upward performance.
      3. Reasonable valuation relative to the universe.
      4. Manageable balance-sheet risk.

    Function defaults are approximately balanced. The production caller
    explicitly uses configurable continuation-oriented weights (currently
    Quality 45%, Value 10%, and Momentum 45%).
    """

    df = df.copy()

    if weights is None:
        weights = {
            'Quality': 0.34,
            'Value': 0.33,
            'Momentum': 0.33
        }

    # =========================================================
    # 0. BASIC DATA CLEANUP
    # =========================================================

    numeric_columns = [
        'ROE',
        'ROA',
        'ProfitMargin',
        'GrossMargin',
        'OperatingMargin',
        'RevenueGrowth',
        'EarningsGrowth',
        'FCFMargin',
        'OperatingCashFlowMargin',
        'CurrentRatio',
        'InterestCoverage',
        'DebtToEquity',
        'PE',
        'PEG',
        'PriceToBook',
        'EV_EBITDA',
        'EV_Revenue',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
        'AnnualizedVolatility',
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
        'BenchmarkExcess1M',
        'BenchmarkExcess3M',
        'BenchmarkExcess6M',
        'BenchmarkExcess9M',
        'BenchmarkExcess1Y'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace(['Infinity', '-Infinity'], np.nan)
                .infer_objects(copy=False)
            )
            df[col] = pd.to_numeric(
                df[col],
                errors='coerce'
            )

    if df.empty:
        return pd.DataFrame()

    # =========================================================
    # 1. QUALITY SCORE
    # =========================================================

    quality_metrics = [
        'ROE',
        'ROA',
        'ProfitMargin',
        'GrossMargin',
        'OperatingMargin',
        'RevenueGrowth',
        'EarningsGrowth',
        'FCFMargin',
        'OperatingCashFlowMargin',
        'CurrentRatio',
        'InterestCoverage'
    ]

    q_cols = [
        c for c in quality_metrics
        if c in df.columns
    ]

    if q_cols:

        q = df[q_cols].copy()

        # -----------------------------------------------------
        # Winsorize extreme observations.
        #
        # This prevents pathological values such as extremely
        # high ROE from dominating the percentile ranking.
        # -----------------------------------------------------

        for col in q_cols:
            valid = q[col].dropna()

            if len(valid) >= 20:
                lower = valid.quantile(0.02)
                upper = valid.quantile(0.98)

                q[col] = q[col].clip(
                    lower=lower,
                    upper=upper
                )

        # Higher quality metric = better.
        q_rank = q.rank(
            pct=True,
            ascending=True
        )

        valid_quality_count = q.notna().sum(axis=1)
        df['QualityMetricCount'] = valid_quality_count.astype(int)

        # Mean only across available metrics.
        quality = q_rank.mean(
            axis=1,
            skipna=True
        )

        # -----------------------------------------------------
        # Missing-data confidence adjustment
        #
        # 5+ valid metrics = full confidence
        # Sparse fundamentals are pulled progressively toward neutral.
        # 0 valid metrics = neutral score
        # -----------------------------------------------------

        quality_confidence = np.select(
            [
                valid_quality_count >= 5,
                valid_quality_count == 4,
                valid_quality_count == 3,
                valid_quality_count == 2,
                valid_quality_count == 1,
            ],
            [1.00, 0.90, 0.75, 0.60, 0.50],
            default=0.40,
        )

        # Move one-metric observations toward neutral (50).
        quality = (
                0.50 +
                (quality - 0.50) * quality_confidence
        )

        quality = quality.where(
            valid_quality_count > 0,
            0.50
        )

        # -----------------------------------------------------
        # Debt penalty
        # -----------------------------------------------------

        if 'DebtToEquity' in df.columns:

            debt_available = df['DebtToEquity'].notna()
            debt = (
                df['DebtToEquity']
                .fillna(50)
                .clip(
                    lower=0,
                    upper=300
                )
            )

            # Little/no penalty below 50 D/E.
            excess_debt = (
                    debt - 50
            ).clip(lower=0)

            # Nonlinear penalty.
            debt_penalty = (
                                   excess_debt / 250
                           ) ** 1.7

            quality = (
                    quality -
                    0.30 * debt_penalty
            )
            # Unknown leverage is uncertainty, not evidence that the company
            # is debt-free. Apply a small deterministic confidence penalty.
            quality = quality - np.where(debt_available, 0.0, 0.03)

        df['QualityScore'] = (
                quality * 100
        ).clip(0, 100)

    else:
        df['QualityScore'] = 50
        df['QualityMetricCount'] = 0

    # Debt is used as a penalty rather than a positively ranked quality
    # metric. Expose its availability so missing leverage data is visible in
    # diagnostics without changing the established scoring behavior.
    df['DebtDataAvailable'] = (
        df['DebtToEquity'].notna()
        if 'DebtToEquity' in df.columns else False
    )

    # Minimum quality filter.
    if min_quality > 0:

        df = df[
            df['QualityScore'] >= min_quality
            ].copy()

    if df.empty:
        return pd.DataFrame()

    # =========================================================
    # 2. VALUE SCORE
    # =========================================================

    value_metrics = [
        'PE',
        'PEG',
        'PriceToBook',
        'EV_EBITDA',
        'EV_Revenue'
    ]

    v_cols = [
        c for c in value_metrics
        if c in df.columns
    ]

    if v_cols:

        value_df = df[v_cols].copy()

        value_df = value_df.replace(
            ['Infinity', '-Infinity'],
            np.nan
        )

        value_df = value_df.apply(
            pd.to_numeric,
            errors='coerce'
        )

        # -----------------------------------------------------
        # Negative and zero valuation multiples are not
        # meaningful measures of "cheapness".
        #
        # Examples:
        #   negative P/E
        #   negative EV/EBITDA
        #
        # These become missing rather than receiving a high
        # value score.
        # -----------------------------------------------------

        value_df = value_df.mask(
            value_df <= 0
        )

        valid_value_count = (
            value_df.notna().sum(axis=1)
        )
        df['ValueMetricCount'] = valid_value_count.astype(int)

        # Lower multiple = better.
        value_rank = value_df.apply(
            lambda x: x.rank(
                pct=True,
                ascending=False
            )
        )

        value_score = value_rank.mean(
            axis=1,
            skipna=True
        )

        # -----------------------------------------------------
        # Missing-data confidence adjustment
        #
        # 2+ metrics = full confidence
        # 1 metric  = reduced confidence
        # 0 metrics = neutral
        # -----------------------------------------------------

        value_confidence = np.where(
            valid_value_count >= 2,
            1.00,
            np.where(
                valid_value_count == 1,
                0.65,
                0.50
            )
        )

        # Pull low-confidence observations toward neutral.
        value_score = (
                0.50 +
                (value_score - 0.50) *
                value_confidence
        )

        value_score = value_score.where(
            valid_value_count > 0,
            0.50
        )

        df['ValueScore'] = (
                value_score * 100
        ).clip(0, 100)

    else:
        df['ValueScore'] = 50
        df['ValueMetricCount'] = 0

    # =========================================================
    # 3. MOMENTUM SCORE
    # =========================================================

    mom_cols = [
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return'
    ]

    m_cols = [
        c for c in mom_cols
        if c in df.columns
    ]

    if m_cols:

        m = df[m_cols].copy()

        # -----------------------------------------------------
        # Prevent pathological return outliers from dominating
        # the ranking.
        # -----------------------------------------------------

        m = m.clip(
            lower=-100,
            upper=500
        )

        # -----------------------------------------------------
        # A. Overall momentum
        # -----------------------------------------------------

        mean_return = m.mean(
            axis=1,
            skipna=True
        )

        raw_momentum = (
                mean_return.rank(
                    pct=True
                ) * 100
        )

        # -----------------------------------------------------
        # B. Trend persistence
        #
        # Count positive periods.
        # -----------------------------------------------------

        positive_windows = (
                m > 0
        ).sum(axis=1)

        valid_momentum_count = (
            m.notna().sum(axis=1)
        )
        df['MomentumMetricCount'] = valid_momentum_count.astype(int)

        trend_alignment = (
                                  positive_windows /
                                  valid_momentum_count.replace(
                                      0,
                                      np.nan
                                  )
                          ) * 100

        trend_alignment = (
            trend_alignment
            .fillna(50)
        )

        # -----------------------------------------------------
        # C. Long-term consistency
        #
        # Compare 3M / 6M / 1Y where available.
        #
        # This rewards sustained strength while applying a
        # moderate penalty to highly erratic rank paths.
        # -----------------------------------------------------

        consistency_cols = [
            c for c in [
                '3M Return',
                '6M Return',
                '1Y Return'
            ]
            if c in df.columns
        ]

        if len(consistency_cols) >= 2:

            consistency = df[
                consistency_cols
            ].copy()

            consistency_rank = (
                consistency.rank(
                    pct=True,
                    axis=0
                )
            )

            consistency_score = (
                    consistency_rank.mean(
                        axis=1,
                        skipna=True
                    ) * 100
            )

            rank_dispersion = (
                consistency_rank.std(
                    axis=1,
                    skipna=True
                )
            )

            # Moderate rather than aggressive penalty.
            consistency_score = (
                    consistency_score -
                    rank_dispersion * 25
            ).clip(0, 100)

            consistency_score = (
                consistency_score
                .fillna(50)
            )

        else:
            consistency_score = raw_momentum

        # -----------------------------------------------------
        # D. Daily price-path quality
        # -----------------------------------------------------
        # Cumulative 1M/3M/6M returns are overlapping observations and their
        # dispersion is not volatility. Use actual daily-price statistics for
        # risk adjustment and trend quality.
        def percentile_score(column, ascending=True):
            if column not in df.columns:
                return pd.Series(50.0, index=df.index)
            values = pd.to_numeric(df[column], errors='coerce')
            return (values.rank(pct=True, ascending=ascending) * 100).fillna(50)

        low_daily_volatility = percentile_score(
            'AnnualizedVolatility', ascending=False
        )
        low_downside_volatility = percentile_score(
            'DownsideVolatility', ascending=False
        )
        shallow_drawdown = percentile_score('MaxDrawdown', ascending=True)
        positive_days = percentile_score('PositiveDayPct', ascending=True)
        trend_fit = percentile_score('TrendR2', ascending=True)
        price_path_quality = (
                0.20 * low_daily_volatility +
                0.20 * low_downside_volatility +
                0.25 * shallow_drawdown +
                0.15 * positive_days +
                0.20 * trend_fit
        )

        # -----------------------------------------------------
        # E. Recent trend confirmation
        # -----------------------------------------------------

        if '3M Return' in df.columns:

            recent_rank = (
                    df['3M Return']
                    .rank(pct=True) * 100
            )

            recent_rank = (
                recent_rank.fillna(50)
            )

        else:
            recent_rank = raw_momentum

        benchmark_relative_components = {
            'BenchmarkExcess1M': 0.05,
            'BenchmarkExcess3M': 0.15,
            'BenchmarkExcess6M': 0.25,
            'BenchmarkExcess9M': 0.25,
            'BenchmarkExcess1Y': 0.30,
        }
        weighted_excess = pd.Series(0.0, index=df.index)
        available_weight = pd.Series(0.0, index=df.index)
        for column, weight in benchmark_relative_components.items():
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors='coerce')
            valid = values.notna()
            weighted_excess = weighted_excess.add(
                values.fillna(0) * weight, fill_value=0
            )
            available_weight = available_weight.add(
                valid.astype(float) * weight, fill_value=0
            )
        weighted_excess = weighted_excess / available_weight.replace(0, np.nan)
        benchmark_relative_score = (
            weighted_excess.rank(pct=True) * 100
        ).fillna(50)

        # -----------------------------------------------------
        # F. Final momentum blend
        #
        # 20% overall return
        # 15% persistence
        # 15% multi-period consistency
        # 25% daily price-path quality
        #  5% recent confirmation
        # 20% performance relative to the configured benchmarks
        # -----------------------------------------------------

        momentum = (
                0.20 * raw_momentum +
                0.15 * trend_alignment +
                0.15 * consistency_score +
                0.25 * price_path_quality +
                0.05 * recent_rank +
                0.20 * benchmark_relative_score
        )

        # Established listings normally provide all five windows. Pull sparse
        # histories gently toward neutral so a recent listing cannot receive a
        # top persistence score from only one or two cumulative returns.
        momentum_confidence = np.select(
            [
                valid_momentum_count >= 4,
                valid_momentum_count == 3,
                valid_momentum_count == 2,
                valid_momentum_count == 1,
            ],
            [1.00, 0.90, 0.75, 0.60],
            default=0.50,
        )
        momentum = 50 + (momentum - 50) * momentum_confidence

        df['MomentumScore'] = (
            momentum
            .fillna(50)
            .clip(0, 100)
        )

    else:
        df['MomentumScore'] = 50
        df['MomentumMetricCount'] = 0

    # =========================================================
    # 4. ADDITIONAL MOMENTUM PENALTIES
    # =========================================================

    if '3M Return' in df.columns:

        recent_return = (
            df['3M Return']
            .fillna(0)
        )

        # Negative recent performance receives a meaningful
        # penalty, but does not automatically eliminate the
        # stock.
        recent_penalty = np.where(
            recent_return < 0,
            np.minimum(
                20,
                -recent_return * 0.5
            ),
            0
        )

        df['MomentumScore'] = (
                df['MomentumScore'] -
                recent_penalty
        ).clip(0, 100)

    # Penalize only unusually fragile entries, not strong momentum by itself.
    # Percentile construction adapts to the current candidate universe and
    # avoids arbitrary universal cutoffs across sectors and volatility regimes.
    entry_risk_components = []
    for column in (
        'Distance50DMA', 'MomentumAcceleration', 'Largest5DayMove',
        'AnnualizedVolatility',
    ):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors='coerce')
            entry_risk_components.append(values.rank(pct=True) * 100)
    if entry_risk_components:
        quantitative_entry_risk = pd.concat(
            entry_risk_components, axis=1
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

    # =========================================================
    # 5. FINAL QVM SCORE
    # =========================================================

    df['QVMScore'] = (
            df['QualityScore'] *
            weights['Quality'] +

            df['ValueScore'] *
            weights['Value'] +

            df['MomentumScore'] *
            weights['Momentum']
    )

    # =========================================================
    # 6. FINAL RISK OVERRIDE
    # =========================================================
    #
    # Extremely leveraged companies should have difficulty
    # reaching the very top regardless of cheap valuation.
    # =========================================================

    if 'DebtToEquity' in df.columns:

        extreme_debt = (
                df['DebtToEquity'] > 200
        )

        df.loc[
            extreme_debt,
            'QVMScore'
        ] *= 0.90

    # =========================================================
    # 7. OUTPUT
    # =========================================================

    return (
        df.sort_values(
            'QVMScore',
            ascending=False
        )
        .head(top_n)
        .reset_index(drop=True)
    )



def update_html_page(
        recommendations_table,
        recommendations_summary,
        df_html_table,
        template_name,
        display_page,
        model_used):
    # --- Read HTML template ---
    with open(template_name, "r", encoding="utf-8") as f:
        template = f.read()

    # --- Insert content ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_output = template.replace("<!--LAST_UPDATED_HERE-->", timestamp)
    html_output = html_output.replace(
        "<!--RECOMMENDATIONS_TABLE_HERE-->", recommendations_table
    )
    html_output = html_output.replace(
        "<!--RECOMMENDATIONS_SUMMARY_HERE-->", recommendations_summary
    )
    html_output = html_output.replace("<!--FULL_DF_TABLE_HERE-->", df_html_table)
    html_output = html_output.replace("<!--MODEL_USED_HERE-->", model_used)

    # --- Write final index.html ---
    with open(display_page, "w", encoding="utf-8") as f:
        f.write(html_output)


def json_safe_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def dataframe_records(df):
    return [
        {column: json_safe_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def build_context_review(decision_ledger):
    headers = [
        "QVM Rank", "Symbol", "Sector Group", "Reversal Risk / Status",
        "Final Score", "Continuation", "Benchmark Outlook",
        "Sector Selected After", "Total Selected After", "Explanation"
    ]
    rows = []
    for decision in decision_ledger:
        rows.append([
            decision["qvm_rank"],
            decision["symbol"],
            decision["sector_group"],
            decision["status"],
            decision.get("final_selection_score"),
            decision.get("continuation_strength"),
            decision.get("benchmark_outperformance_outlook"),
            decision["sector_selected_after"],
            decision["total_selected_after"],
            decision["explanation"],
        ])
    return "CONTEXT REVIEW\n\n" + pd.DataFrame(rows, columns=headers).to_markdown(index=False)


def build_classification_snapshots(candidate_records, research_by_symbol):
    candidates = {
        str(candidate["Symbol"]).upper(): candidate
        for candidate in candidate_records
    }
    snapshots = {}
    for symbol, research in research_by_symbol.items():
        candidate = candidates.get(symbol, {})
        snapshots[symbol] = {
            "qvm_rank": candidate.get("QVM Rank"),
            "qvm_score": candidate.get("QVMScore"),
            **{
                field: research.get(field)
                for field in DECISION_DIAGNOSTIC_FIELDS
            },
            "source_count": len(research.get("sources") or []),
        }
    return snapshots


def build_stock_analysis_snapshots(candidate_records, research_by_symbol, cache_entries, decisions):
    """Build one upload-friendly per-stock record with quantitative and model provenance."""
    candidates = {
        str(candidate["Symbol"]).upper(): candidate
        for candidate in candidate_records
    }
    snapshots = {}
    for symbol, research in research_by_symbol.items():
        candidate = candidates.get(symbol, {})
        cache_key = global_cache_keys_by_symbol.get(symbol)
        cache_entry = cache_entries.get(cache_key, {}) if cache_key else {}
        decision = decisions.get(symbol, {})
        snapshots[symbol] = {
            "qvm_rank": candidate.get("QVM Rank"),
            "qvm_score": candidate.get("QVMScore"),
            "quality_score": candidate.get("QualityScore"),
            "value_score": candidate.get("ValueScore"),
            "momentum_score": candidate.get("MomentumScore"),
            "quantitative_entry_risk": candidate.get("QuantitativeEntryRisk"),
            "overextension_penalty": candidate.get("OverextensionPenalty"),
            "sector": candidate.get("Sector"),
            "market_cap": candidate.get("MarketCap"),
            "price": candidate.get("Price"),
            "returns": {
                "1m": candidate.get("1M Return"),
                "3m": candidate.get("3M Return"),
                "6m": candidate.get("6M Return"),
                "9m": candidate.get("9M Return"),
                "1y": candidate.get("1Y Return", candidate.get("52 WkChange %")),
            },
            "price_path": {
                "annualized_volatility": candidate.get("AnnualizedVolatility"),
                "downside_volatility": candidate.get("DownsideVolatility"),
                "max_drawdown": candidate.get("MaxDrawdown"),
                "positive_day_pct": candidate.get("PositiveDayPct"),
                "trend_r2": candidate.get("TrendR2"),
                "distance_50dma": candidate.get("Distance50DMA"),
                "distance_200dma": candidate.get("Distance200DMA"),
                "distance_52w_high": candidate.get("Distance52WHigh"),
                "largest_5d_move": candidate.get("Largest5DayMove"),
                "momentum_acceleration": candidate.get("MomentumAcceleration"),
            },
            "benchmark_excess_returns": {
                "1m": candidate.get("BenchmarkExcess1M"),
                "3m": candidate.get("BenchmarkExcess3M"),
                "6m": candidate.get("BenchmarkExcess6M"),
                "9m": candidate.get("BenchmarkExcess9M"),
                "1y": candidate.get("BenchmarkExcess1Y"),
            },
            "research_provenance": {
                "model": cache_entry.get("research_model") or cache_entry.get("model"),
                "created_at": cache_entry.get("created_at"),
                "age_hours_at_report": cache_age_hours(cache_entry.get("created_at")),
                "fresh_this_run": symbol in researched_symbols_this_run,
                "search_queries": (cache_entry.get("research_metadata") or {}).get("search_queries", []),
                "source_count": len(research.get("sources") or []),
                "source_publication_dates": [
                    {"url": source.get("url"), "published_at": next(
                        (source[key] for key in (
                            "published_at", "publication_date", "published_date", "date"
                        ) if source.get(key)), None
                    )}
                    for source in research.get("sources") or []
                    if isinstance(source, dict)
                ],
            },
            "material_company_event": research.get("material_company_event"),
            "material_company_event_review": (
                "REPORTED" if research.get("material_company_event") else
                "NONE_FOUND_IN_RESEARCH" if "material_company_event" in research else
                "NOT_RECORDED"
            ),
            "judgment_provenance": {
                "model": cache_entry.get("judgment_model"),
                "judged_at": cache_entry.get("judged_at"),
                "fallback_used": cache_entry.get("judgment_fallback_used"),
                "fresh_this_run": any(
                    symbol in (call.get("symbols") or [])
                    and call.get("success")
                    for call in classification_call_diagnostics
                ),
            },
            "classification": {
                field: research.get(field)
                for field in DECISION_DIAGNOSTIC_FIELDS
            },
            "benchmark_outperformance_basis": research.get(
                "benchmark_outperformance_basis"
            ),
            "decision": decision.get("status"),
            "final_score_components": decision.get("final_score_components"),
        }
    return snapshots


def compare_classification_snapshots(previous, current):
    drift = []
    comparable_fields = ("qvm_rank", "qvm_score") + DECISION_DIAGNOSTIC_FIELDS
    for symbol in sorted(set(previous).intersection(current)):
        changes = {
            field: {
                "previous": previous[symbol].get(field),
                "current": current[symbol].get(field),
            }
            for field in comparable_fields
            if previous[symbol].get(field) != current[symbol].get(field)
        }
        if changes:
            drift.append({
                "symbol": symbol,
                "changed_fields": sorted(changes),
                "changes": changes,
            })
    return drift


def build_decision_snapshots(decision_ledger):
    return {
        decision["symbol"]: {
            "qvm_rank": decision["qvm_rank"],
            "sector_group": decision["sector_group"],
            "status": decision["status"],
            "final_selection_score": decision.get("final_selection_score"),
            "final_score_components": decision.get("final_score_components"),
            "benchmark_outperformance_basis": decision.get(
                "benchmark_outperformance_basis"
            ),
            "material_company_event": decision.get("material_company_event"),
            "continuation_strength": decision.get("continuation_strength"),
            "benchmark_outperformance_outlook": decision.get(
                "benchmark_outperformance_outlook"
            ),
        }
        for decision in decision_ledger
    }


def build_portfolio_changes(
        previous_selected, current_selected, previous_decisions,
        current_decisions, classification_drift):
    drift_by_symbol = {
        item["symbol"]: item["changed_fields"]
        for item in classification_drift
    }
    added = []
    removed = []
    previous_set = set(previous_selected)
    current_set = set(current_selected)
    for symbol in current_selected:
        if symbol not in previous_set:
            added.append({
                "symbol": symbol,
                "previous_status": previous_decisions.get(symbol, {}).get(
                    "status", "not present"
                ),
                "current_status": current_decisions.get(symbol, {}).get(
                    "status", "not present"
                ),
                "classification_fields_changed": drift_by_symbol.get(
                    symbol, []
                ),
            })
    for symbol in previous_selected:
        if symbol not in current_set:
            removed.append({
                "symbol": symbol,
                "previous_status": previous_decisions.get(symbol, {}).get(
                    "status", "not present"
                ),
                "current_status": current_decisions.get(symbol, {}).get(
                    "status", "not present"
                ),
                "classification_fields_changed": drift_by_symbol.get(
                    symbol, []
                ),
            })
    return {"added": added, "removed": removed}


def format_number(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}"


def yahoo_link(symbol, label):
    safe_symbol = escape(str(symbol))
    safe_label = escape(str(label))
    return (
        f'<a href="https://finance.yahoo.com/quote/{safe_symbol}/" '
        f'target="_blank"><strong>{safe_label}</strong></a>'
    )


def build_recommendations_table(selected):
    lines = [
        '<table class="recommendations-table">',
        "<thead>",
        "<tr>",
        "<th>Symbol</th>",
        "<th>Stock Name</th>",
        "<th>Sector Group</th>",
        "<th>52 Wk Change (%)</th>",
        "<th>3 Mo Return (%)</th>",
        "<th>QVMScore</th>",
        "<th>Reversal Risk</th>",
        "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for item in selected:
        candidate = item["candidate"]
        research = item["research"]
        symbol = candidate["Symbol"]
        lines.extend([
            "<tr>",
            f"<td>{yahoo_link(symbol, symbol)}</td>",
            f"<td>{yahoo_link(symbol, candidate['Name'])}</td>",
            f"<td>{escape(str(candidate['Sector']))}</td>",
            f"<td>{format_number(candidate.get('52 WkChange %'))}</td>",
            f"<td>{format_number(candidate.get('3M Return'))}</td>",
            f"<td>{format_number(candidate.get('QVMScore'))}</td>",
            f"<td><strong>{escape(research['reversal_risk'])}</strong></td>",
            "</tr>",
        ])
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)


def sector_context_sentence(market_context, sector):
    context = market_context.get("sector_context", {})
    if sector in context and str(context[sector]).strip():
        return str(context[sector]).strip()

    sector_lower = str(sector).lower()
    for key, value in context.items():
        if str(key).lower() == sector_lower and str(value).strip():
            return str(value).strip()

    return f"{sector} remains relevant under the current market environment."


def build_recommendations_summary(market_context, selected):
    grouped = {}
    for item in selected:
        sector = item["candidate"]["Sector"]
        grouped.setdefault(sector, []).append(item)

    lines = [
        '<div class="summary">',
        "<h2>Market Chat:</h2>",
        f"<p>{escape(str(market_context['market_intro']))}</p>",
    ]

    for sector, items in grouped.items():
        links = ", ".join(
            yahoo_link(item["candidate"]["Symbol"], item["candidate"]["Symbol"])
            for item in items
        )
        industry_contexts = list(dict.fromkeys(
            str(item["research"].get("industry_context", "")).strip()
            for item in items
            if str(item["research"].get("industry_context", "")).strip()
        ))
        sector_context = " ".join(industry_contexts) or sector_context_sentence(
            market_context, sector
        )
        sentences = [escape(sector_context)]
        for item in items:
            candidate = item["candidate"]
            research = item["research"]
            symbol_link = yahoo_link(candidate["Symbol"], candidate["Symbol"])
            name_link = yahoo_link(candidate["Symbol"], candidate["Name"])
            sentences.append(
                f"{symbol_link} ({name_link}), "
                f"{escape(research['business_description'])}, has "
                f"<strong>{escape(research['reversal_risk'])}</strong> reversal risk: "
                f"{escape(research['explanation'])}"
            )

        lines.append(
            f"<p><strong>{escape(str(sector))}</strong> ({links}): "
            + " ".join(sentences)
            + "</p>"
        )

    lines.append("</div>")
    return "\n".join(lines)


def validate_summary_presence(data):
    summary_html = data.get("summary_html")
    if not isinstance(summary_html, str) or not summary_html.strip():
        raise ValueError("Final summary response has no summary_html.")


def validate_summary_response(data, selected):
    summary_html = data.get("summary_html")
    if not isinstance(summary_html, str) or not summary_html.strip():
        raise ValueError("Final summary response has no summary_html.")

    soup = BeautifulSoup(summary_html, "html.parser")
    summary_divs = soup.find_all("div", class_="summary")
    if len(summary_divs) != 1:
        raise ValueError("Final summary must contain exactly one summary div.")
    if not summary_divs[0].find("h2"):
        raise ValueError("Final summary is missing its Market Chat heading.")

    expected_symbols = {
        str(item["candidate"]["Symbol"]).upper()
        for item in selected
    }
    linked_symbol_sequence = [
        match.upper()
        for match in re.findall(
            r"https://finance\.yahoo\.com/quote/([^/]+)/",
            summary_html,
            flags=re.IGNORECASE,
        )
    ]
    linked_symbols = set(linked_symbol_sequence)
    if linked_symbols != expected_symbols:
        raise ValueError(
            "Final summary symbol mismatch: expected "
            f"{sorted(expected_symbols)}, received {sorted(linked_symbols)}"
        )

    summary_text = soup.get_text(" ", strip=True)
    paragraphs = summary_divs[0].find_all("p")
    if not paragraphs or not re.search(
            r"\d+(?:\.\d+)?\s*%", paragraphs[0].get_text(" ", strip=True)):
        raise ValueError("Final summary market paragraph has no percentage.")
    prohibited_filler = re.compile(
        r"\b(?:most other sectors are currently experiencing general weakness|"
        r"conditions are mixed|the sector remains relevant)\b",
        flags=re.IGNORECASE,
    )
    if prohibited_filler.search(summary_text):
        raise ValueError("Final summary contains generic sector filler.")
    matched_sector_paragraph_ids = set()
    for item in selected:
        candidate = item["candidate"]
        research = item["research"]
        symbol = str(candidate["Symbol"]).upper()
        stock_paragraphs = [
            paragraph
            for paragraph in paragraphs
            if f"/quote/{symbol}/" in str(paragraph)
        ]
        if len(stock_paragraphs) != 1:
            raise ValueError(
                f"Final summary must discuss {symbol} in exactly one paragraph."
            )
        paragraph = stock_paragraphs[0]
        paragraph_text = paragraph.get_text(" ", strip=True)
        matched_sector_paragraph_ids.add(id(paragraph))
        if str(candidate["Name"]) not in paragraph_text:
            raise ValueError(
                f"Final summary omitted stock name {candidate['Name']}."
            )
        if str(candidate["Sector"]) not in paragraph_text:
            raise ValueError(
                f"Final summary placed {symbol} outside {candidate['Sector']}."
            )
        if str(research["reversal_risk"]) not in paragraph_text:
            raise ValueError(
                "Final summary omitted reversal risk "
                f"{research['reversal_risk']}."
            )

    expected_sector_count = len({
        item["candidate"]["Sector"] for item in selected
    })
    if len(matched_sector_paragraph_ids) != expected_sector_count:
        raise ValueError(
            "Final summary does not contain exactly one paragraph per sector."
        )
    if len(paragraphs) != expected_sector_count + 1:
        raise ValueError(
            "Final summary must contain one market paragraph plus exactly one "
            "paragraph per represented sector."
        )

    if re.search(r"\b(?:QVM|CONTEXT REVIEW|FINAL_SELECTED_STOCKS)\b", summary_text):
        raise ValueError("Final summary exposed internal selection terminology.")



def final_candidate_components(research):
    """Named parts of the existing deterministic post-research adjustment."""
    risk = combined_reversal_risk(research)
    components = {}
    components["reversal_risk"] = {
        "MINIMAL": 5.0,
        "LOW": 3.0,
        "MODERATE": -3.0,
        "ELEVATED": -1000.0,
        "SEVERE": -1000.0,
    }.get(risk, -1000.0)

    benchmark_outlook = str(
        research.get("benchmark_outperformance_outlook") or "UNCERTAIN"
    ).upper()
    components["benchmark_outlook"] = {
        "LIKELY": benchmark_likely_bonus,
        "UNCERTAIN": benchmark_uncertain_penalty,
        "UNLIKELY": -1000.0,
    }.get(benchmark_outlook, -2.0)

    continuation_strength = str(
        research.get("continuation_strength") or "WEAK"
    ).upper()
    components["continuation_strength"] = continuation_strength_adjustments.get(
        continuation_strength, continuation_strength_adjustments["WEAK"]
    )

    catalyst_dependence = str(
        research.get("catalyst_dependence") or "LOW"
    ).upper()
    components["catalyst_dependence"] = {"LOW": 0.0, "MODERATE": -1.0, "HIGH": -2.0}.get(
        catalyst_dependence, -1.0
    )

    for field in ("business_concentration", "binary_event_risk"):
        level = str(research.get(field) or "LOW").upper()
        components[field] = {"LOW": 0.0, "MODERATE": -1.0, "HIGH": -2.0}.get(
            level, -1.0
        )

    mechanism = str(research.get("mechanism_status") or "").upper()
    components["mechanism_status"] = (
        1.0 if mechanism == "NONE" else
        active_mechanism_penalty if mechanism == "ACTIVE" else 0.0
    )
    return components


def final_candidate_adjustment(research):
    """Small deterministic post-research adjustment; QVM remains dominant."""
    return sum(final_candidate_components(research).values())


def final_candidate_score(candidate, research):
    return float(candidate.get("QVMScore") or 0.0) + final_candidate_adjustment(research)


# Main execution
# Record the start time
start_time = time.perf_counter()
signal.signal(signal.SIGALRM, handle_total_runtime_timeout)
signal.alarm(TOTAL_RUNTIME_TIMEOUT_SECONDS)

with open("stock_config.yml") as f:
    config = yaml.safe_load(f)
run_report_file = config.get(
    "run_report_file",
    "run_reports/stock_run_report.json",
)
previous_run_diagnostics = load_json_object(run_report_file)

print(f"Run report: {Path(run_report_file).resolve()}")
print(f"Run log: {STOCK_RUN_LOG_FILE.resolve()}")
benchmark_symbols = [
    str(symbol).strip().upper()
    for symbol in config.get("benchmark_symbols", ["SPMO", "VGT"])
    if str(symbol).strip()
]
benchmark_min_52_week_change_fallback = float(
    config.get("benchmark_min_52_week_change_fallback", 15.0)
)
benchmark_hurdle_cache_tolerance_pct = max(
    0.0, float(config.get("benchmark_hurdle_cache_tolerance_pct", 0.25))
)
min_market_cap = int(config.get("min_market_cap", 300_000_000))
min_price = float(config.get("min_price", 5.0))
min_average_volume = int(config.get("min_average_volume", 100_000))
min_3_month_return = float(config.get("min_3_month_return", 0.0))
stock_screener_page_size = int(config.get("stock_screener_page_size", 250))
configured_max_info_calls = config.get("max_info_calls")
max_info_calls = None if configured_max_info_calls is None else int(configured_max_info_calls)
max_retries = config["max_retries"]
initial_delay = config["initial_delay"]
max_validation_rounds = max(
    1,
    int(config.get("max_validation_rounds", 4)),
)
max_transient_api_attempts = int(
    config.get("max_transient_api_attempts", 4)
)
initial_transient_backoff_seconds = float(
    config.get("initial_transient_backoff_seconds", 15)
)
max_transient_backoff_seconds = float(
    config.get("max_transient_backoff_seconds", 90)
)
transient_backoff_jitter_seconds = float(
    config.get("transient_backoff_jitter_seconds", 10)
)
model_primary = config["model_primary"]
model_fallback = config["model_fallback"]
classification_model = str(config.get("classification_model", "gemini-3.5-flash"))
classification_fallback_model = str(
    config.get("classification_fallback_model", model_primary)
)
classification_thinking_budget = int(
    config.get("classification_thinking_budget", 8192)
)
classification_max_output_tokens = int(
    config.get("classification_max_output_tokens", 32768)
)
classification_attempts = max(1, int(config.get("classification_attempts", 2)))
max_classification_calls_per_run = max(
    1, int(config.get("max_classification_calls_per_run", 8))
)
classification_batch_target = max(1, int(config.get("classification_batch_target", 25)))
classification_batch_soft_max = max(
    classification_batch_target, int(config.get("classification_batch_soft_max", 30))
)
classification_min_intermediate_batch = max(
    1, int(config.get("classification_min_intermediate_batch", 15))
)
gemini_batch_size = int(config.get("gemini_batch_size", 5))
max_research_candidates_per_open_sector_slot = max(
    1,
    int(config.get("max_research_candidates_per_open_sector_slot", 3)),
)
max_candidates = int(config.get("max_candidates", 50))
normal_candidate_limit = int(
    config.get("normal_candidate_limit", min(50, max_candidates))
)
if not 1 <= normal_candidate_limit <= max_candidates:
    raise ValueError(
        "normal_candidate_limit must be between 1 and max_candidates."
    )
target_selected_stocks = int(config.get("target_selected_stocks", 10))
max_stocks_per_sector = int(config.get("max_stocks_per_sector", 2))
max_stocks_per_risk_event = int(
    config.get(
        "max_stocks_per_risk_event",
        config.get("max_moderate_per_risk_event", 2),
    )
)
if max_stocks_per_risk_event < 1:
    raise ValueError("max_stocks_per_risk_event must be at least 1.")
max_stocks_per_return_driver = int(config.get("max_stocks_per_return_driver", 2))
if max_stocks_per_return_driver < 1:
    raise ValueError("max_stocks_per_return_driver must be at least 1.")
second_return_driver_minimum_lead = float(
    config.get("second_return_driver_minimum_lead", 2.0)
)
if not np.isfinite(second_return_driver_minimum_lead) or second_return_driver_minimum_lead < 0:
    raise ValueError("second_return_driver_minimum_lead must be finite and nonnegative.")
risk_adjusted_selection_enabled = bool(
    config.get("risk_adjusted_selection_enabled", True)
)
moderate_risk_qvm_penalty = float(
    config.get("moderate_risk_qvm_penalty", 6.0)
)
repeated_risk_event_qvm_penalty = float(
    config.get("repeated_risk_event_qvm_penalty", 4.0)
)
max_moderate_risk_selections = max(
    0, int(config.get("max_moderate_risk_selections", 3))
)
exclude_elevated_entry_risk = bool(
    config.get("exclude_elevated_entry_risk", True)
)
benchmark_uncertain_qvm_penalty = float(
    config.get("benchmark_uncertain_qvm_penalty", 3.0)
)
exclude_unlikely_benchmark_outperformance = bool(
    config.get("exclude_unlikely_benchmark_outperformance", True)
)
qvm_weights = {
    "Quality": float(config.get("qvm_quality_weight", 0.45)),
    "Value": float(config.get("qvm_value_weight", 0.10)),
    "Momentum": float(config.get("qvm_momentum_weight", 0.45)),
}
if not np.isclose(sum(qvm_weights.values()), 1.0):
    raise ValueError("Configured QVM weights must sum to 1.0.")
minimum_final_selection_score = float(config.get("minimum_final_selection_score", 72.0))
minimum_uncertain_moderate_score = float(config.get("minimum_uncertain_moderate_score", 75.0))
minimum_uncertain_low_score = float(config.get("minimum_uncertain_low_score", 72.0))
second_event_minimum_score = float(config.get("second_event_minimum_score", 78.0))
benchmark_likely_bonus = float(config.get("benchmark_likely_bonus", 5.0))
benchmark_uncertain_penalty = float(config.get("benchmark_uncertain_penalty", -4.0))
active_mechanism_penalty = float(config.get("active_mechanism_penalty", -5.0))
second_event_score_penalty = float(config.get("second_event_score_penalty", -3.0))
for name, value in (
    ("minimum_final_selection_score", minimum_final_selection_score),
    ("minimum_uncertain_moderate_score", minimum_uncertain_moderate_score),
    ("minimum_uncertain_low_score", minimum_uncertain_low_score),
    ("second_event_minimum_score", second_event_minimum_score),
    ("benchmark_likely_bonus", benchmark_likely_bonus),
    ("benchmark_uncertain_penalty", benchmark_uncertain_penalty),
    ("active_mechanism_penalty", active_mechanism_penalty),
    ("second_event_score_penalty", second_event_score_penalty),
):
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
continuation_strength_adjustments = {
    "STRONG": float(config.get("continuation_strong_bonus", 3.0)),
    "ADEQUATE": float(config.get("continuation_adequate_bonus", 0.0)),
    "WEAK": float(config.get("continuation_weak_penalty", -4.0)),
}
cautious_exposure_floor_enabled = bool(
    config.get("cautious_exposure_floor_enabled", True)
)
cautious_exposure_floor_min_dependence = str(
    config.get("cautious_exposure_floor_min_dependence", "MODERATE")
).strip().upper()
if cautious_exposure_floor_min_dependence not in DEPENDENCE_LEVELS:
    raise ValueError(
        "cautious_exposure_floor_min_dependence must be LOW, MODERATE, or HIGH."
    )
excluded_crypto_dependence = {
    str(level).strip().upper()
    for level in config.get("excluded_crypto_dependence", ["MATERIAL", "PRIMARY"])
}
if not excluded_crypto_dependence.issubset(CRYPTO_DEPENDENCE_LEVELS):
    raise ValueError(
        "excluded_crypto_dependence contains an unsupported level."
    )
thinking_budget = int(config.get("thinking_budget", 12288))
require_google_search = bool(config.get("require_google_search", True))
max_gemini_calls_per_run = int(config.get("max_gemini_calls_per_run", 12))
max_stock_research_calls_per_run = int(
    config.get("max_stock_research_calls_per_run", max_gemini_calls_per_run)
)
reserved_summary_calls = int(config.get("reserved_summary_calls", 1))
max_research_attempts_per_stock = int(
    config.get("max_research_attempts_per_stock", 2)
)
max_deferred_research_attempts_per_run = max(
    1,
    int(config.get("max_deferred_research_attempts_per_run", 1)),
)
max_structural_repairs_per_stock = int(
    config.get("max_structural_repairs_per_stock", 2)
)
max_stock_sources = int(config.get("max_stock_sources", 8))
market_context_cache_file = cache_file_path(config.get(
    "market_context_cache_file", "gemini_market_context_cache.json"
))
stock_research_cache_file = cache_file_path(config.get(
    "stock_research_cache_file", "gemini_stock_research_cache.json"
))
gemini_research_cache_hours = float(
    config.get("gemini_research_cache_hours", 12)
)
cache_version = int(config.get("cache_version", 1))
final_summary_enabled = bool(config.get("final_summary_enabled", True))
summary_thinking_budget = int(config.get("summary_thinking_budget", 4096))

print(f"Stock config loaded from: {Path('stock_config.yml').resolve()}")
print(
    "Effective Gemini settings: "
    f"batch_size={gemini_batch_size}, "
    f"research_candidates_per_open_sector_slot="
    f"{max_research_candidates_per_open_sector_slot}, "
    f"max_calls={max_gemini_calls_per_run}, "
    f"max_stock_calls={max_stock_research_calls_per_run}, "
    f"research_attempts_per_stock={max_research_attempts_per_stock}, "
    f"deferred_research_attempts_per_run="
    f"{max_deferred_research_attempts_per_run}, "
    f"structural_repairs_per_stock={max_structural_repairs_per_stock}, "
    f"thinking_budget={thinking_budget}, cache_version={cache_version}, "
    f"cautious_floor={cautious_exposure_floor_enabled}, "
    f"normal_candidates={normal_candidate_limit}, "
    f"max_candidates={max_candidates}, "
    f"stocks_per_risk_event={max_stocks_per_risk_event}, "
    f"max_moderate={max_moderate_risk_selections}, "
    f"qvm_weights={qvm_weights}, "
    f"benchmarks={benchmark_symbols}, "
    f"excluded_crypto={sorted(excluded_crypto_dependence)}"
)

benchmark_context = fetch_benchmark_performance(
    benchmark_symbols,
    benchmark_min_52_week_change_fallback,
)
min_52_week_change = float(
    benchmark_context["hurdle_52_week_return"]
)

# ---------------------------------------------------------
# Load the cached ranked QVM candidate pool when fresh.
#
# This check happens BEFORE any stock-page, yfinance, or
# QVM-scoring work, so Gemini prompt testing can reuse the
# exact same quantitative input without rerunning the
# expensive pipeline.
# ---------------------------------------------------------
top_stocks = load_top_qvm_cache(
    benchmark_context=benchmark_context,
    hurdle_tolerance_pct=benchmark_hurdle_cache_tolerance_pct,
)

if top_stocks is None:

    df = fetch_stock_universe(
        min_52_week_change=min_52_week_change,
        min_market_cap=min_market_cap,
        min_price=min_price,
        min_average_volume=min_average_volume,
        max_retries=max_retries,
        page_size=stock_screener_page_size,
    )

    print("\nStocks passing screener-level filters:")
    print(df[['Symbol', 'Name', '52 WkChange %']].reset_index(drop=True))

    minimal_cols = [
        'Symbol',
        'Name',
        'Market Cap',
        'Price',
        'P/E Ratio(TTM)',
        '52 WkChange %',
        'Avg Vol (3M)'
    ]
    df_minimal = df[minimal_cols].copy()

    df_yf = append_qvm_data_yfinance(
        df_minimal,
        max_info_calls=max_info_calls,
        min_3_month_return=min_3_month_return,
        benchmark_context=benchmark_context,
    )
    if "3M Return" in df_yf.columns:
        df_yf = df_yf[
            df_yf["3M Return"].notna()
            & (df_yf["3M Return"] >= min_3_month_return)
        ].copy()
    df_scored = score_qvm(df_yf, weights=qvm_weights)

    # Keep the configured top QVM candidate stream for Gemini evaluation.
    top_stocks = df_scored.head(max_candidates).copy()

    save_top_qvm_cache(top_stocks, benchmark_context=benchmark_context)

print("\nTop QVM Stocks:")
top_stocks = top_stocks.head(max_candidates).copy().reset_index(drop=True)
print(top_stocks[['Symbol', 'QVMScore', '3M Return','1Y Return']])

cols_for_eval = [
    'Symbol',
    'QVMScore',
    'QualityScore',
    'QualityMetricCount',
    'ValueScore',
    'ValueMetricCount',
    'MomentumScore',
    'MomentumMetricCount',
    'QuantitativeEntryRisk',
    'OverextensionPenalty',
    'HistoryDays',
    'ROE',
    'DebtToEquity',
    'DebtDataAvailable',
    'EV_EBITDA',
    'PEG',
    'RevenueGrowth',
    'EarningsGrowth',
    'OperatingMargin',
    'FCFMargin',
    '3M Return',
    '6M Return',
    '1Y Return',
    'BenchmarkExcess3M',
    'BenchmarkExcess6M',
    'BenchmarkExcess1Y'
]
#print to file for inspection and evaluation
TOP_QVM_STOCKS_MD_FILE.write_text(
    top_stocks[cols_for_eval].to_markdown(index=False),
    encoding="utf-8",
)

essential_columns_for_gemini = [
    # Identity
    "Symbol", "Name", "Sector",

    # Current trading identity / size context
    "Price", "Avg Vol (3M)", "Market Cap", "HistoryDays",

    # Core QVM output (most important)
    "QVMScore",

    # Decomposed signals (compressed)
    "QualityScore",
    "ValueScore",
    "MomentumScore",
    "QuantitativeEntryRisk",
    "OverextensionPenalty",

    # Key fundamentals
    "ROE",
    "ProfitMargin",
    "OperatingMargin",
    "RevenueGrowth",
    "EarningsGrowth",
    "FCFMargin",
    "OperatingCashFlowMargin",
    "DebtToEquity",
    "EV_EBITDA",
    "PEG",

    # Valuation anchor
    "PE",

    # Momentum anchors
    "1M Return",
    "3M Return",
    "6M Return",
    "9M Return",
    "52 WkChange %",

    # Daily price-path and entry-risk evidence
    "AnnualizedVolatility",
    "DownsideVolatility",
    "MaxDrawdown",
    "PositiveDayPct",
    "TrendR2",
    "AnnualizedTrend",
    "Distance50DMA",
    "Distance200DMA",
    "Distance52WHigh",
    "Largest1DayMove",
    "Largest5DayMove",
    "MomentumAcceleration",
    "BenchmarkExcess1M", "BenchmarkExcess3M", "BenchmarkExcess6M",
    "BenchmarkExcess9M", "BenchmarkExcess1Y"
]

# Build the strict, ranked candidate stream sent to Gemini in batches.
df_gemini = top_stocks[essential_columns_for_gemini].copy().reset_index(drop=True)
df_gemini.insert(0, "QVM Rank", range(1, len(df_gemini) + 1))
candidate_records = dataframe_records(df_gemini)
required_sector_groups = list(dict.fromkeys(
    str(candidate["Sector"]) for candidate in candidate_records
))


def validate_current_market_context(data):
    validate_market_context(data)
    context = data.get("sector_context", {})
    normalized_context = {
        str(key).strip().casefold(): str(value).strip()
        for key, value in context.items()
    }
    missing_sectors = [
        sector for sector in required_sector_groups
        if not normalized_context.get(sector.strip().casefold())
    ]
    if missing_sectors:
        raise ValueError(
            "Market context is missing required sector context: "
            + ", ".join(missing_sectors)
        )

client = initialize_gemini_client()
gemini_config = build_gemini_config(thinking_budget)
request_budget = GeminiRequestBudget(
    max_gemini_calls_per_run,
    stock_maximum=max_stock_research_calls_per_run,
    reserved_summary_calls=(reserved_summary_calls if final_summary_enabled else 0),
)

market_prompt = (
    config["prompt_market_context"]
    + f"\n\nCURRENT_DATE_UTC: {datetime.now(UTC).date().isoformat()}\n"
    + "REQUIRED_SECTOR_GROUPS:\n"
    + json.dumps(required_sector_groups, ensure_ascii=False)
    + "\n"
)
market_prompt_hash = stable_json_hash({
    "cache_version": cache_version,
    "model": model_primary,
    "prompt": market_prompt,
})
market_cache = load_json_object(market_context_cache_file)
market_context = None
market_model = None

if (
        market_cache.get("version") == cache_version
        and market_cache.get("prompt_hash") == market_prompt_hash
        and market_cache.get("model") == model_primary
        and cache_entry_is_fresh(
            market_cache, gemini_research_cache_hours
        )
):
    try:
        validate_current_market_context(market_cache["market_context"])
        market_context = market_cache["market_context"]
        market_model = market_cache["model"]
        print(f"Using validated market context cache: {market_context_cache_file}")
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Ignoring invalid market context cache: {exc}")

if market_context is None:
    print("\n...calling Gemini for current market context...\n")
    market_context, market_model, market_metadata = call_gemini_json(
        client=client,
        model_primary=model_primary,
        model_fallback=model_fallback,
        gemini_config=gemini_config,
        prompt=market_prompt,
        stage="market context",
        validator=validate_current_market_context,
        request_budget=request_budget,
        require_google_search=require_google_search,
        budget_category="market",
    )
    save_json_object_atomic(market_context_cache_file, {
        "version": cache_version,
        "created_at": datetime.now(UTC).isoformat(),
        "prompt_hash": market_prompt_hash,
        "model": market_model,
        "market_context": market_context,
        "research_metadata": {
            "search_queries": market_metadata["search_queries"],
            "tool_tokens": market_metadata["tool_tokens"],
        },
    })

market_context_hash = stable_json_hash(market_context)
allowed_risk_event_ids = {
    event["event_id"] for event in market_context["active_risk_events"]
}
print(
    "Dynamic market risk-event catalog: "
    f"{len(allowed_risk_event_ids)} canonical event(s)."
)
# Output-transport wording does not change researched facts or classifications,
# so it should not invalidate otherwise valid stock research. This preserves
# the prior prompt hash while still applying the no-duplicate instruction to
# every new Gemini request.
stock_prompt_cache_text = config["prompt_stock_batch"].replace(
    "After emitting END_STOCK_RESULT for a symbol, never emit that symbol again.\n"
    "Do not repeat, revise, or self-correct an earlier completed result block.\n",
    "",
)
stock_prompt_hash = stable_json_hash({
    "cache_version": cache_version,
    "model": model_primary,
    "prompt": stock_prompt_cache_text,
    # The cache stores the post-judgment merged result, so a judgment-schema
    # change must invalidate stock research even when grounded research text is unchanged.
    "judgment_prompt": config["prompt_stock_judgment"],
    "classification_model": classification_model,
})
# The prior v6 cache key included the former judgment instructions. Reuse its
# still-fresh grounded evidence once, then require a new 3.5 judgment for the
# dynamically assigned return-driver group. This saves 2.5 Search quota.
previous_judgment_stock_prompt_hash = (
    "9523141ba91b41acb469b70c9af4ec13a10c1c95f08c25c03eeefebfaddc7d77"
    if cache_version == 6
    and model_primary == "gemini-2.5-flash"
    and classification_model == "gemini-3.5-flash"
    else None
)
stock_research_cache = load_json_object(stock_research_cache_file)
if stock_research_cache.get("version") != cache_version:
    stock_research_cache = {
        "version": cache_version, "entries": {}, "deferred_entries": {}
    }
stock_research_cache.setdefault("entries", {})
stock_research_cache.setdefault("deferred_entries", {})
stock_research_cache["entries"] = {
    key: entry
    for key, entry in stock_research_cache["entries"].items()
    if isinstance(entry, dict)
    and cache_entry_is_fresh(entry, gemini_research_cache_hours)
}
stock_research_cache["deferred_entries"] = {
    key: entry
    for key, entry in stock_research_cache["deferred_entries"].items()
    if isinstance(entry, dict)
    and cache_entry_is_fresh(entry, gemini_research_cache_hours)
}
save_json_object_atomic(stock_research_cache_file, stock_research_cache)

# Resolve every fresh stock cache entry before making any new request. This
# allows Python to determine whether the complete portfolio can already be
# built from prior successful calls, including lower-ranked cached candidates.
validated_cached_research = {}
pending_classification = {}
global_cache_keys_by_symbol = {}
for candidate in candidate_records:
    symbol = str(candidate["Symbol"]).upper()
    cache_key = stable_json_hash({
        "market_context_hash": market_context_hash,
        "stock_prompt_hash": stock_prompt_hash,
        "model": model_primary,
        "candidate": candidate,
    })
    global_cache_keys_by_symbol[symbol] = cache_key
    entry = stock_research_cache["entries"].get(cache_key)
    if not entry and previous_judgment_stock_prompt_hash:
        previous_cache_key = stable_json_hash({
            "market_context_hash": market_context_hash,
            "stock_prompt_hash": previous_judgment_stock_prompt_hash,
            "model": model_primary,
            "candidate": candidate,
        })
        previous_entry = stock_research_cache["entries"].get(previous_cache_key)
        if (
            isinstance(previous_entry, dict)
            and previous_entry.get("market_context_hash") == market_context_hash
            and previous_entry.get("candidate_hash") == stable_json_hash(candidate)
            and previous_entry.get("stock_prompt_hash")
            == previous_judgment_stock_prompt_hash
            and isinstance(previous_entry.get("research"), dict)
        ):
            # Keep all original grounded facts and sources. The old judgment
            # fields are ignored until the new classifier overwrites them.
            entry = {
                **previous_entry,
                "research": dict(previous_entry["research"]),
                "judgment_model": None,
                "judged_at": None,
                "stock_prompt_hash": stock_prompt_hash,
            }
            stock_research_cache["entries"][cache_key] = entry
            print(f"Reusing grounded research for {symbol}; refreshing return-driver judgment.")
    if not entry:
        deferred = stock_research_cache["deferred_entries"].get(cache_key)
        if (isinstance(deferred, dict)
                and isinstance(deferred.get("research"), dict)
                and "invalid continuation_strength" in str(
                    deferred.get("validation_error") or ""
                )):
            # Older runs incorrectly demanded a 3.5-owned field from 2.5.
            # Recover the grounded draft when it passes research validation.
            draft = deferred.get("research")
            metadata = deferred.get("research_metadata") or {}
            try:
                validate_stock_research_evidence(
                    {"results": [draft]}, [candidate],
                    minimum_sources=minimum_sources_for_candidate(
                        metadata.get("search_queries", []), candidate
                    ),
                )
            except (KeyError, TypeError, ValueError):
                pass
            else:
                entry = {
                    "created_at": datetime.now(UTC).isoformat(),
                    "model": deferred.get("model") or model_primary,
                    "research_model": deferred.get("model") or model_primary,
                    "judgment_model": None,
                    "market_context_hash": market_context_hash,
                    "stock_prompt_hash": stock_prompt_hash,
                    "candidate_hash": stable_json_hash(candidate),
                    "research": draft,
                    "research_metadata": metadata,
                }
                stock_research_cache["entries"][cache_key] = entry
                stock_research_cache["deferred_entries"].pop(cache_key, None)
                print(f"Recovered grounded research for {symbol} for 3.5 classification.")
    if not entry:
        continue
    try:
        cached_result = entry["research"]
        cached_search_queries = entry.get(
            "research_metadata", {}
        ).get("search_queries", [])
        cache_minimum_sources = minimum_sources_for_candidate(
            cached_search_queries, candidate
        )
        if entry.get("judgment_model"):
            validate_stock_batch(
                {"results": [cached_result]}, [candidate],
                minimum_sources=cache_minimum_sources,
                allowed_risk_event_ids=allowed_risk_event_ids,
            )
        else:
            validate_stock_research_evidence(
                {"results": [cached_result]}, [candidate],
                minimum_sources=cache_minimum_sources,
            )
        if entry.get("judgment_model"):
            validated_cached_research[symbol] = cached_result
        else:
            pending_classification[symbol] = cached_result
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Ignoring invalid stock cache entry for {symbol}: {exc}")

save_json_object_atomic(stock_research_cache_file, stock_research_cache)

# Individual cache entries can each be structurally valid while disagreeing
# about a shared market mechanism. Audit them together before deciding the
# cache can fill the portfolio. An inconsistent LOW result becomes a deferred
# draft so it receives a focused, evidence-backed correction.
cache_consistency_changed = False
for symbol, cached_result in list(validated_cached_research.items()):
    comparison_results = [
        other
        for other_symbol, other in validated_cached_research.items()
        if other_symbol != symbol
    ]
    try:
        validate_shared_event_consistency(cached_result, comparison_results)
    except ValueError as exc:
        cache_key = global_cache_keys_by_symbol[symbol]
        prior_entry = stock_research_cache["entries"].pop(cache_key, {})
        validated_cached_research.pop(symbol, None)
        stock_research_cache["deferred_entries"][cache_key] = {
            "created_at": datetime.now(UTC).isoformat(),
            "model": prior_entry.get("model") or model_primary,
            "research": cached_result,
            "validation_error": str(exc),
            "research_attempts": 0,
            "repair_attempts": 0,
            "research_metadata": prior_entry.get("research_metadata", {}),
        }
        cache_consistency_changed = True
        print(f"Deferred inconsistent cached research for {symbol}: {exc}")

# Cross-check individually valid cached records against one another so stale
# issuer-fact contamination cannot survive simply because each object validates
# in isolation. Flagged entries are deferred for one fresh issuer-specific retry.
cached_contamination = detect_cross_company_evidence_contamination(
    list(validated_cached_research.values())
)
if cached_contamination:
    for symbol, reason in cached_contamination.items():
        cached_result = validated_cached_research.pop(symbol, None)
        if cached_result is None:
            continue
        cache_key = global_cache_keys_by_symbol[symbol]
        prior_entry = stock_research_cache["entries"].pop(cache_key, {})
        stock_research_cache["deferred_entries"][cache_key] = {
            "created_at": datetime.now(UTC).isoformat(),
            "model": prior_entry.get("model") or model_primary,
            "research": cached_result,
            "validation_error": reason,
            "research_attempts": 0,
            "repair_attempts": 0,
            "research_metadata": prior_entry.get("research_metadata", {}),
        }
        cache_consistency_changed = True
        print(f"Deferred contaminated cached research for {symbol}: {reason}")

if cache_consistency_changed:
    save_json_object_atomic(stock_research_cache_file, stock_research_cache)

initial_validated_cache_symbols = set(validated_cached_research)

def final_selection_rejection(score, research, event_count):
    """Use one quality bar for provisional stopping and final portfolio selection."""
    continuation = str(research.get("continuation_strength") or "").upper()
    benchmark = str(research.get("benchmark_outperformance_outlook") or "").upper()
    risk = combined_reversal_risk(research)
    mechanism = str(research.get("mechanism_status") or "").upper()
    if continuation not in {"STRONG", "ADEQUATE"}:
        return "NOT SELECTED — WEAK CONTINUATION"
    if score < minimum_final_selection_score:
        return "NOT SELECTED — BELOW BENCHMARK FALLBACK BAR"
    if benchmark == "UNCERTAIN":
        if risk in {"MINIMAL", "LOW"}:
            if continuation != "STRONG" or score < minimum_uncertain_low_score:
                return "NOT SELECTED — UNCERTAIN BENCHMARK CASE"
        elif risk == "MODERATE":
            if continuation != "STRONG" or score < minimum_uncertain_moderate_score:
                return "NOT SELECTED — UNCERTAIN MODERATE-RISK CASE"
            if mechanism == "ACTIVE":
                return "NOT SELECTED — ACTIVE ADVERSE MECHANISM"
        else:
            return "NOT SELECTED — UNCERTAIN BENCHMARK CASE"
    if event_count and (
        score + second_event_score_penalty < second_event_minimum_score
        or (benchmark != "LIKELY" and
            (continuation != "STRONG" or risk not in {"MINIMAL", "LOW"}))
    ):
        return "NOT SELECTED — SECOND RISK-EVENT QUALITY BAR"
    return None


def best_independent_alternative(
        remaining, current_driver, sector_counts, event_counts,
        driver_counts, moderate_count):
    """Find a still-selectable candidate with a different return driver."""
    for alt_score, _, alt_candidate, alt_research in remaining:
        symbol = str(alt_candidate["Symbol"]).upper()
        driver = normalized_return_driver_key(alt_research)
        # Compare against a candidate that can enter without itself consuming
        # a second driver slot; otherwise a later rejection can make the
        # supposed alternative disappear.
        if driver == current_driver or (driver and driver_counts.get(driver, 0)):
            continue
        risk = combined_reversal_risk(alt_research)
        event = normalized_risk_event_key(alt_research)
        benchmark = str(
            alt_research.get("benchmark_outperformance_outlook") or "UNCERTAIN"
        ).upper()
        entry_risk = str(alt_research.get("entry_reversal_risk") or risk).upper()
        if (
            not bool(alt_research.get("eligible", True))
            or excluded_by_crypto_policy(alt_research, excluded_crypto_dependence)
            or risk in {"ELEVATED", "SEVERE"}
            or (exclude_elevated_entry_risk and entry_risk in {"ELEVATED", "SEVERE"})
            or (exclude_unlikely_benchmark_outperformance and benchmark == "UNLIKELY")
            or sector_counts.get(alt_candidate["Sector"], 0) >= max_stocks_per_sector
            or (risk == "MODERATE" and moderate_count >= max_moderate_risk_selections)
            or (event and event_counts.get(event, 0) >= max_stocks_per_risk_event)
            or (driver and driver_counts.get(driver, 0) >= max_stocks_per_return_driver)
            or final_selection_rejection(
                alt_score, alt_research, event_counts.get(event, 0) if event else 0
            )
        ):
            continue
        return symbol, alt_score
    return None


def build_stock_portfolio(candidate_records, validated_cached_research, verbose=False):
    selected = []
    decision_ledger = []
    sector_counts = {}
    risk_event_counts = {}
    return_driver_counts = {}
    moderate_selected_count = 0
    
    scored_validated_candidates = []
    for candidate in candidate_records:
        symbol = str(candidate["Symbol"]).upper()
        research = validated_cached_research.get(symbol)
        if research is None:
            continue
        scored_validated_candidates.append((
            final_candidate_score(candidate, research),
            -int(candidate["QVM Rank"]),
            candidate,
            research,
        ))
    scored_validated_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    
    if verbose:
        print("FINAL VALIDATED CANDIDATE COMPARISON")
    for score, _, candidate, research in scored_validated_candidates:
        if verbose:
            print(
            f"  {candidate['Symbol']}: final_score={score:.2f}, "
            f"qvm={float(candidate.get('QVMScore') or 0):.2f}, "
            f"adjustments={final_candidate_components(research)}, "
            f"risk={combined_reversal_risk(research)}, "
            f"continuation={research.get('continuation_strength')}, "
            f"benchmark={research.get('benchmark_outperformance_outlook')}, "
            f"mechanism={research.get('mechanism_status')}, "
            f"risk_event={normalized_risk_event_key(research)}, "
            f"return_driver={normalized_return_driver_key(research)}"
        )
    
    for position, (score, _, candidate, research) in enumerate(scored_validated_candidates):
        symbol = str(candidate["Symbol"]).upper()
        sector = candidate["Sector"]
        sector_count = sector_counts.get(sector, 0)
        reversal_risk = combined_reversal_risk(research)
        entry_reversal_risk = str(
            research.get("entry_reversal_risk") or reversal_risk
        ).upper()
        benchmark_outlook = str(
            research.get("benchmark_outperformance_outlook") or "UNCERTAIN"
        ).upper()
        continuation_strength = str(
            research.get("continuation_strength") or "WEAK"
        ).upper()
        explanation = str(research.get("explanation") or "").strip()
        event_key = normalized_risk_event_key(research)
        driver_key = normalized_return_driver_key(research)
        independent_alternative = None
    
        status = None
        if not bool(research.get("eligible", True)):
            status = "EXCLUDED — ELIGIBILITY"
        elif excluded_by_crypto_policy(research, excluded_crypto_dependence):
            status = "EXCLUDED — CRYPTO DEPENDENCE"
        elif reversal_risk in {"ELEVATED", "SEVERE"}:
            status = f"NOT SELECTED — {reversal_risk}"
        elif (
            exclude_elevated_entry_risk
            and entry_reversal_risk in {"ELEVATED", "SEVERE"}
        ):
            status = f"NOT SELECTED — {entry_reversal_risk} ENTRY RISK"
        elif (
            exclude_unlikely_benchmark_outperformance
            and benchmark_outlook == "UNLIKELY"
        ):
            status = "NOT SELECTED — BENCHMARK OUTPERFORMANCE UNLIKELY"
        elif (quality_rejection := final_selection_rejection(
            score, research, risk_event_counts.get(event_key, 0) if event_key else 0
        )):
            status = quality_rejection
        elif sector_count >= max_stocks_per_sector:
            status = "SKIPPED — SECTOR CAPACITY"
        elif (
            reversal_risk == "MODERATE"
            and moderate_selected_count >= max_moderate_risk_selections
        ):
            status = "NOT SELECTED — MODERATE-RISK CAPACITY"
        elif (
            event_key
            and risk_event_counts.get(event_key, 0) >= max_stocks_per_risk_event
        ):
            status = "NOT SELECTED — RISK-EVENT CAPACITY"
        elif (
            driver_key
            and return_driver_counts.get(driver_key, 0) >= max_stocks_per_return_driver
        ):
            status = "NOT SELECTED — RETURN-DRIVER CAPACITY"
        elif (
            driver_key
            and return_driver_counts.get(driver_key, 0) == 1
            and (
                independent_alternative := best_independent_alternative(
                    scored_validated_candidates[position + 1:], driver_key,
                    sector_counts, risk_event_counts, return_driver_counts,
                    moderate_selected_count,
                )
            )
            and score - independent_alternative[1] < second_return_driver_minimum_lead
        ):
            status = "NOT SELECTED — SECOND RETURN-DRIVER QUALITY BAR"
        elif len(selected) >= target_selected_stocks:
            status = "NOT SELECTED — LOWER FINAL SCORE"
        else:
            selected.append({"candidate": candidate, "research": research})
            sector_counts[sector] = sector_count + 1
            if reversal_risk == "MODERATE":
                moderate_selected_count += 1
            if event_key:
                risk_event_counts[event_key] = risk_event_counts.get(event_key, 0) + 1
            if driver_key:
                return_driver_counts[driver_key] = return_driver_counts.get(driver_key, 0) + 1
            status = f"SELECTED — {reversal_risk}"

        if verbose and independent_alternative:
            print(
                f"  Return-driver comparison [{symbol}, {driver_key}]: "
                f"{score:.2f} vs independent {independent_alternative[0]} "
                f"{independent_alternative[1]:.2f}; minimum lead "
                f"{second_return_driver_minimum_lead:.2f}; {status}"
            )
    
        decision_ledger.append({
            "qvm_rank": candidate["QVM Rank"],
            "symbol": symbol,
            "sector_group": sector,
            "status": status,
            "sector_selected_after": sector_counts.get(sector, 0),
            "total_selected_after": len(selected),
            "final_selection_score": score,
            "final_score_components": {
                "qvm": float(candidate.get("QVMScore") or 0.0),
                **final_candidate_components(research),
            },
            "benchmark_outperformance_basis": research.get(
                "benchmark_outperformance_basis"
            ),
            "material_company_event": research.get("material_company_event"),
            "return_driver_group": driver_key,
            "independent_alternative": (
                independent_alternative[0] if independent_alternative else None
            ),
            "independent_alternative_score": (
                independent_alternative[1] if independent_alternative else None
            ),
            "continuation_strength": continuation_strength,
            "benchmark_outperformance_outlook": benchmark_outlook,
            "explanation": explanation,
        })
    return selected, decision_ledger, sector_counts, scored_validated_candidates


selected = []
decision_ledger = []
sector_counts = {}
# Research coverage is independent from provisional portfolio fills. Final
# sector/risk caps are applied only after the validated pool is frozen.
research_sector_counts = {}
risk_event_counts = {}
moderate_selected_count = 0
prior_research_decisions = []
research_failures_by_symbol = {}
classification_failures_by_symbol = {}
classification_failure_details_by_symbol = {}
classification_unavailable_this_run = False
models_used = [market_model]
researched_symbols_this_run = set()
stock_search_attempts_this_run = 0
prior_invalid_results_by_symbol = {}
research_attempts_by_symbol = {}
repair_attempts_by_symbol = {}
deferred_symbols_this_run = set()
preexisting_deferred_cache_keys = set(
    stock_research_cache["deferred_entries"]
)
unsearched_missing_refunds_this_run = set()
batch_research_diagnostics = []
normalization_diagnostics_by_symbol = {}


def per_run_attempt_limit(symbol, needs_research):
    """Bound retries within this execution without permanently locking drafts."""
    if needs_research:
        if symbol in deferred_symbols_this_run:
            return max_deferred_research_attempts_per_run
        return max_research_attempts_per_stock
    return max_structural_repairs_per_stock

candidate_by_symbol = {str(row["Symbol"]).upper(): row for row in candidate_records}


def optimistic_portfolio_capacity(
        provisional, pending, candidates_by_symbol, sector_limit,
        excluded_crypto_levels):
    """Upper bound on slots pending evidence could fill, before 3.5 judges it.

    Only deterministic eligibility and sector limits reduce this bound. Risk,
    benchmark outlook, and shared-event limits are decided after classification.
    """
    counts = {}
    for item in provisional:
        sector = item["candidate"]["Sector"]
        counts[sector] = counts.get(sector, 0) + 1
    capacity = len(provisional)
    for symbol, research in pending.items():
        if not bool(research.get("eligible", True)):
            continue
        if excluded_by_crypto_policy(research, excluded_crypto_levels):
            continue
        sector = candidates_by_symbol[symbol]["Sector"]
        if counts.get(sector, 0) < sector_limit:
            counts[sector] = counts.get(sector, 0) + 1
            capacity += 1
    return capacity


def classify_pending(force=False):
    """Judge validated 2.5 drafts in large batches; preserve drafts on failure."""
    global classification_unavailable_this_run
    if classification_unavailable_this_run:
        return
    while pending_classification and (force or len(pending_classification) >= classification_batch_target):
        symbols = list(pending_classification)[:classification_batch_soft_max]
        tail = len(pending_classification) - len(symbols)
        if 0 < tail < classification_min_intermediate_batch:
            symbols = symbols[:max(
                classification_min_intermediate_batch,
                len(symbols) - (classification_min_intermediate_batch - tail),
            )]
        drafts = [pending_classification[symbol] for symbol in symbols]
        candidates = [candidate_by_symbol[symbol] for symbol in symbols]
        judged, judge_model = judge_stock_research_batch(
            client, drafts, candidates, market_context,
            previous_run_diagnostics.get("classifications", {}),
            prior_research_decisions,
        )
        if not judge_model:
            print("Judgment unavailable; leaving validated research queued for a later run.")
            classification_unavailable_this_run = True
            break
        models_used.append(judge_model)
        valid_count = 0
        for result in judged:
            symbol = str(result.get("symbol") or "").upper()
            if symbol not in symbols or symbol not in pending_classification:
                continue
            candidate = candidate_by_symbol[symbol]
            cache_key = global_cache_keys_by_symbol[symbol]
            entry = stock_research_cache["entries"].get(cache_key, {})
            # Capture the model's values before validation can normalize or
            # mutate them. This record has no effect on eligibility or scores.
            judgment_fields = {
                field: result.get(field)
                for field in DECISION_DIAGNOSTIC_FIELDS
            }
            evidence_fields = {
                field: compact_diagnostic_value(result.get(field), maximum=300)
                for field in (
                    "temporary_drivers", "reversal_mechanism", "current_fact",
                    "probability_evidence", "material_effect",
                )
            }
            try:
                if result.pop("_missing_return_driver_group", False):
                    raise ValueError(
                        f"{symbol} judgment omitted risk_exposure_group; "
                        "grounded research remains cached for reclassification."
                    )
                validate_stock_batch(
                    {"results": [result]}, [candidate],
                    minimum_sources=minimum_sources_for_candidate(
                        (entry.get("research_metadata") or {}).get("search_queries", []),
                        candidate,
                    ),
                    allowed_risk_event_ids=allowed_risk_event_ids,
                )
                validate_shared_event_consistency(
                    result, list(validated_cached_research.values())
                )
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Classification invalid for {symbol}: {exc}; preserving grounded draft for a future judgment.")
                classification_failures_by_symbol[symbol] = str(exc)
                classification_failure_details_by_symbol[symbol] = {
                    "error": str(exc),
                    "judgment_model": judge_model,
                    "fields_before_validation": judgment_fields,
                    "evidence_excerpt": evidence_fields,
                }
                print(
                    f"  Invalid judgment fields for {symbol}: "
                    f"{json.dumps(classification_failure_details_by_symbol[symbol], default=str)}"
                )
                # Keep the previously validated 2.5 evidence in entries with
                # judgment_model=None. It must not consume another Search call.
            else:
                valid_count += 1
                validated_cached_research[symbol] = result
                entry.update({
                    "research": result,
                    "judgment_model": judge_model,
                    "judgment_fallback_used": bool(
                        judge_model and judge_model != classification_model
                    ),
                    "judged_at": datetime.now(UTC).isoformat() if judge_model else None,
                })
                stock_research_cache["entries"][cache_key] = entry
                research_failures_by_symbol.pop(symbol, None)
                classification_failures_by_symbol.pop(symbol, None)
                classification_failure_details_by_symbol.pop(symbol, None)
            del pending_classification[symbol]
        save_json_object_atomic(stock_research_cache_file, stock_research_cache)
        if valid_count == 0:
            print(
                "No judgments passed final validation; stopping further 2.5 "
                "research. Grounded drafts remain cached for later classification."
            )
            classification_unavailable_this_run = True
            break
        if not force and len(pending_classification) < classification_batch_target:
            break


batch_start = 0
carried_ranked_candidates = []
backfill_announced = False
while batch_start < len(candidate_records) or carried_ranked_candidates:
    if classification_unavailable_this_run:
        print("Stopping stock research because no classification model is available.")
        break
    validated_for_pool = []
    for pool_candidate in candidate_records:
        pool_symbol = str(pool_candidate["Symbol"]).upper()
        pool_research = validated_cached_research.get(pool_symbol)
        if not pool_research:
            continue
        pool_risk = combined_reversal_risk(pool_research)
        pool_benchmark = str(
            pool_research.get("benchmark_outperformance_outlook") or "UNCERTAIN"
        ).upper()
        if (
            pool_risk in {"ELEVATED", "SEVERE"}
            or excluded_by_crypto_policy(pool_research, excluded_crypto_dependence)
            or (exclude_unlikely_benchmark_outperformance and pool_benchmark == "UNLIKELY")
        ):
            continue
        validated_for_pool.append(pool_symbol)

    if (
        request_budget.stock_maximum is not None
        and request_budget.stock_used >= request_budget.stock_maximum
    ):
        print(
            "Stock research-call budget exhausted; freezing the validated "
            f"comparison pool at {len(validated_for_pool)} eligible candidates."
        )
        break

    provisional, _, _, _ = build_stock_portfolio(
        candidate_records, validated_cached_research
    )
    if len(provisional) >= target_selected_stocks:
        print(f"Portfolio meeting final quality bar: {len(provisional)}/{target_selected_stocks}; freezing research.")
        break
    optimistic_capacity = optimistic_portfolio_capacity(
        provisional, pending_classification, candidate_by_symbol,
        max_stocks_per_sector, excluded_crypto_dependence,
    )
    if pending_classification and optimistic_capacity >= target_selected_stocks:
        print(
            "Classifying pending research before another 2.5 call: "
            f"{len(provisional)} selected, {len(pending_classification)} pending, "
            f"optimistic capacity {optimistic_capacity}/{target_selected_stocks}."
        )
        classify_pending(force=True)
        provisional, _, _, _ = build_stock_portfolio(
            candidate_records, validated_cached_research
        )
        if len(provisional) >= target_selected_stocks:
            print(f"Portfolio meeting final quality bar: {len(provisional)}/{target_selected_stocks}; freezing research.")
            break

    if batch_start >= normal_candidate_limit and not backfill_announced:
        print(
            f"Validated pool has {len(validated_for_pool)} eligible stocks "
            f"after the normal top-{normal_candidate_limit} search depth; "
            f"backfilling from QVM ranks {normal_candidate_limit + 1}-"
            f"{max_candidates}."
        )
        backfill_announced = True
    batch_end = min(batch_start + gemini_batch_size, len(candidate_records))
    if batch_start < normal_candidate_limit < batch_end:
        batch_end = normal_candidate_limit
    new_ranked_candidates = candidate_records[batch_start:batch_end]
    batch_start = batch_end
    ranked_pool = sorted(
        carried_ranked_candidates + new_ranked_candidates,
        key=lambda candidate: int(candidate["QVM Rank"]),
    )
    carried_ranked_candidates = []

    # A batch needs alternatives because research can exclude candidates, but
    # sending six same-sector names for one remaining slot wastes searches.
    # Keep the highest-QVM alternatives within the configured per-slot limit
    # and carry excess candidates forward. They are reconsidered after the
    # current results update sector capacity, so they are never mislabeled as
    # invalid or permanently skipped.
    (
        ranked_batch,
        research_candidates,
        carried_ranked_candidates,
        queued_by_sector,
    ) = partition_ranked_research_candidates(
        ranked_pool,
        research_sector_counts,
        max_stocks_per_sector,
        gemini_batch_size,
        max_research_candidates_per_open_sector_slot,
    )

    if carried_ranked_candidates:
        print(
            "Deferred excess same-sector research candidates: "
            + ", ".join(
                str(candidate["Symbol"]).upper()
                for candidate in carried_ranked_candidates
            )
        )

    research_by_symbol = {}
    uncached_candidates = []
    cache_keys_by_symbol = {}

    for candidate in research_candidates:
        symbol = str(candidate["Symbol"]).upper()
        cache_key = global_cache_keys_by_symbol[symbol]
        cache_keys_by_symbol[symbol] = cache_key
        if symbol in research_failures_by_symbol or symbol in pending_classification:
            continue
        cached_result = validated_cached_research.get(symbol)
        if cached_result is not None:
            research_by_symbol[symbol] = cached_result
            entry = stock_research_cache["entries"].get(cache_key, {})
            models_used.append(entry.get("model") or model_primary)
            print(f"Using validated stock research cache for {symbol}.")
            continue
        uncached_candidates.append(candidate)

    # Keep ordinary research calls full while the portfolio remains unfillable.
    ordinary_uncached_count = len(uncached_candidates)
    desired_research_count = gemini_batch_size
    if (
        uncached_candidates
        and len(uncached_candidates) < desired_research_count
    ):
        queued_symbols = {
            str(candidate["Symbol"]).upper()
            for candidate in uncached_candidates
        }
        future_start = batch_start
        future_limit = (
            len(candidate_records)
            if backfill_announced
            else normal_candidate_limit
        )
        for future_candidate in candidate_records[future_start:future_limit]:
            if len(uncached_candidates) >= desired_research_count:
                break
            future_symbol = str(future_candidate["Symbol"]).upper()
            if (
                    future_symbol in queued_symbols
                    or future_symbol in research_failures_by_symbol
            ):
                continue
            if research_sector_counts.get(future_candidate["Sector"], 0) >= (
                max_stocks_per_sector
            ):
                continue
            future_sector = future_candidate["Sector"]
            future_open_slots = (
                max_stocks_per_sector - research_sector_counts.get(future_sector, 0)
            )
            future_sector_limit = (
                future_open_slots
                * max_research_candidates_per_open_sector_slot
            )
            if queued_by_sector.get(future_sector, 0) >= future_sector_limit:
                continue

            future_cache_key = stable_json_hash({
                "market_context_hash": market_context_hash,
                "stock_prompt_hash": stock_prompt_hash,
                "model": model_primary,
                "candidate": future_candidate,
            })
            future_entry = stock_research_cache["entries"].get(
                future_cache_key
            )
            if future_entry and cache_entry_is_fresh(
                future_entry, gemini_research_cache_hours
            ):
                continue

            cache_keys_by_symbol[future_symbol] = future_cache_key
            uncached_candidates.append(future_candidate)
            queued_symbols.add(future_symbol)
            queued_by_sector[future_sector] = (
                queued_by_sector.get(future_sector, 0) + 1
            )

        if len(uncached_candidates) > ordinary_uncached_count:
            print(
                "Filled ordinary research batch with future uncached QVM "
                "candidates: "
                + ", ".join(
                    str(candidate["Symbol"]).upper()
                    for candidate in uncached_candidates
                )
            )

    if uncached_candidates:
        pending_candidates = list(uncached_candidates)
        batched_symbols = {
            str(candidate["Symbol"]).upper()
            for candidate in pending_candidates
        }
        exhausted_symbols = set()
        validation_errors = {}
        for candidate in pending_candidates:
            symbol = str(candidate["Symbol"]).upper()
            deferred = stock_research_cache["deferred_entries"].get(
                cache_keys_by_symbol[symbol]
            )
            if (
                    cache_keys_by_symbol[symbol]
                    not in preexisting_deferred_cache_keys
                    or not isinstance(deferred, dict)
            ):
                continue
            draft = deferred.get("research")
            error = str(deferred.get("validation_error") or "").strip()
            if isinstance(draft, dict) and error:
                prior_invalid_results_by_symbol[symbol] = draft
                validation_errors[symbol] = error
                deferred_symbols_this_run.add(symbol)
                # Attempt counts are per execution. Preserve the draft and its
                # sources, but do not let an earlier run permanently lock out
                # a stock that Gemini may complete successfully today.
                research_attempts_by_symbol.setdefault(symbol, 0)
                repair_attempts_by_symbol.setdefault(symbol, 0)
                prior_research_attempts = int(
                    deferred.get("research_attempts", 0)
                )
                prior_repair_attempts = int(deferred.get("repair_attempts", 0))
                print(
                    f"Using deferred research draft for {symbol}; "
                    "allowing a fresh per-run attempt "
                    f"(previous run: research={prior_research_attempts}, "
                    f"repair={prior_repair_attempts})."
                )
        eligible_pending = []
        for candidate in pending_candidates:
            symbol = str(candidate["Symbol"]).upper()
            prior_error = validation_errors.get(symbol)
            if not prior_error:
                eligible_pending.append(candidate)
                continue
            needs_research = validation_error_requires_fresh_research(
                prior_error
            )
            attempts_used = (
                research_attempts_by_symbol.get(symbol, 0)
                if needs_research
                else repair_attempts_by_symbol.get(symbol, 0)
            )
            attempt_limit = per_run_attempt_limit(symbol, needs_research)
            if attempts_used >= attempt_limit:
                exhausted_symbols.add(symbol)
                research_failures_by_symbol[symbol] = prior_error
                print(
                    f"Skipping deferred {symbol}; its permitted "
                    f"{'research' if needs_research else 'repair'} attempts "
                    f"are exhausted ({attempts_used}/{attempt_limit} this run)."
                )
                continue
            eligible_pending.append(candidate)
        pending_candidates = eligible_pending
        research_round = 0

        # This loop is bounded per stock, rather than by a global number of
        # batch calls. Targeted calls contain only candidates that failed the
        # preceding validation attempt.
        while pending_candidates:
            if (request_budget.stock_maximum is not None
                    and request_budget.stock_used >= request_budget.stock_maximum):
                print(
                    "Stock research-call budget exhausted with pending retries; "
                    "keeping validated evidence and proceeding to classification."
                )
                break
            research_round += 1
            research_required_symbols = {
                str(candidate["Symbol"]).upper()
                for candidate in pending_candidates
                if str(candidate["Symbol"]).upper() not in validation_errors
                or validation_error_requires_fresh_research(
                    validation_errors[str(candidate["Symbol"]).upper()]
                )
            }
            research_required_candidates = [
                candidate for candidate in pending_candidates
                if str(candidate["Symbol"]).upper()
                in research_required_symbols
            ]
            for candidate in pending_candidates:
                symbol = str(candidate["Symbol"]).upper()
                if symbol in research_required_symbols:
                    research_attempts_by_symbol[symbol] = (
                        research_attempts_by_symbol.get(symbol, 0) + 1
                    )
                else:
                    repair_attempts_by_symbol[symbol] = (
                        repair_attempts_by_symbol.get(symbol, 0) + 1
                    )
            print(
                "\n...calling Gemini for stock batch: "
                + ", ".join(row["Symbol"] for row in pending_candidates)
                + "...\n"
            )

            retry_correction = ""
            if validation_errors:
                retry_details = "\n".join(
                    f"  - {symbol}: {message}"
                    for symbol, message in validation_errors.items()
                )
                retry_correction = f"""

DEFERRED VALIDATION CORRECTIONS
Valid research for other candidates has already been saved. Candidates listed
below previously failed validation; correct each exact error. Their prior JSON
objects are supplied as drafts. Preserve every valid field and source from each
draft. Change only fields needed to correct the listed error, unless fresh
research contradicts the draft. Do not discard a valid researched answer and
start over merely to repair formatting or a derived synthesis. This targeted
call contains no new filler candidates. Return every supplied candidate once.

Run one new focused Google search only for the following candidates whose
errors require missing evidence to be researched:
{json.dumps(sorted(research_required_symbols), ensure_ascii=False)}
For every other candidate, perform no new search; repair the supplied draft and
retain its valid sources.

VALIDATION_ERRORS:
{retry_details}

PRIOR_INVALID_RESULTS_TO_REPAIR:
{json.dumps({
    symbol: prior_invalid_results_by_symbol[symbol]
    for symbol in validation_errors
    if symbol in prior_invalid_results_by_symbol
}, ensure_ascii=False)}
"""

            stock_prompt = (
                config["prompt_stock_batch"]
                + retry_correction
                + f"\n\nCURRENT_DATE_UTC: {datetime.now(UTC).date().isoformat()}\n"
                + "\n\nMARKET_CONTEXT:\n"
                + json.dumps(market_context, ensure_ascii=False)
                + "\n\nPRIOR_RESEARCH_DECISIONS:\n"
                + json.dumps(prior_research_decisions, ensure_ascii=False)
                + "\n\nCANDIDATES:\n"
                + json.dumps(pending_candidates, ensure_ascii=False)
            )
            stage = (
                f"stock batch containing {len(pending_candidates)} stocks, "
                f"spanning QVM ranks {pending_candidates[0]['QVM Rank']}-"
                f"{pending_candidates[-1]['QVM Rank']}"
            )
            if research_round > 1:
                stage += f" targeted retry {research_round - 1}"

            batch_data, batch_model, batch_metadata = call_gemini_json(
                client=client,
                model_primary=model_primary,
                # Flash-Lite proved unreliable for this large, search-grounded,
                # strict-schema workload. Retry transient errors on Flash and
                # preserve caches for a later rerun instead of degrading.
                model_fallback=model_primary,
                gemini_config=gemini_config,
                prompt=stock_prompt,
                stage=stage,
                validator=lambda data, expected=pending_candidates: (
                    validate_stock_batch_structure(data, expected)
                ),
                request_budget=request_budget,
                require_google_search=(
                    require_google_search
                    and bool(research_required_candidates)
                ),
                required_search_candidates=research_required_candidates,
                allow_partial_stock_results=True,
                budget_category="stock",
            )
            models_used.append(batch_model)

            # Classify only after per-stock research validation below.
            judgment_model_used = None

            requested_research_symbols = {
                str(candidate["Symbol"]).upper()
                for candidate in research_required_candidates
            }
            repeated_searches = len(
                requested_research_symbols.intersection(
                    researched_symbols_this_run
                )
            )
            stock_search_attempts_this_run += len(requested_research_symbols)
            researched_symbols_this_run.update(requested_research_symbols)
            results_by_symbol = {
                str(result.get("symbol", "")).upper(): result
                for result in batch_data["results"]
            }
            missing_response_symbols = [
                str(candidate["Symbol"]).upper()
                for candidate in pending_candidates
                if str(candidate["Symbol"]).upper() not in results_by_symbol
            ]
            response_coverage = len(results_by_symbol) / len(pending_candidates)
            coverage_status = (
                "SEVERE_UNDERCOVERAGE" if response_coverage < 0.50 else
                "DEGRADED" if response_coverage < 0.80 else "NORMAL"
            )
            print(
                f"Stock response coverage: {len(results_by_symbol)}/"
                f"{len(pending_candidates)} returned; "
                f"{len(missing_response_symbols)} missing; "
                f"status={coverage_status}."
            )
            print(
                f"Research searches requested this call: "
                f"{len(requested_research_symbols)} total; "
                f"{len(requested_research_symbols) - repeated_searches} first, "
                f"{repeated_searches} repeated."
            )
            contamination_conflicts = detect_cross_company_evidence_contamination(
                batch_data["results"]
            )
            if contamination_conflicts:
                print(
                    "Cross-company evidence contamination check flagged: "
                    + ", ".join(sorted(contamination_conflicts))
                )

            next_pending = []
            next_errors = {}
            exhausted_errors = {}
            batch_stats = {
                "stage": stage,
                "coverage_status": coverage_status,
                "coverage_fraction": round(response_coverage, 3),
                "candidates_sent": len(pending_candidates),
                "fresh_research_requested": len(requested_research_symbols),
                "company_searches_exposed": len(
                    batch_metadata["search_queries"]
                ),
                "results_returned": len(results_by_symbol),
                "validated": 0,
                "missing": len(missing_response_symbols),
                "incomplete": 0,
                "structural_repairs_needed": 0,
                "genuine_repeated_searches": repeated_searches,
                "missing_unsearched_attempts_refunded": 0,
            }

            for candidate in pending_candidates:
                symbol = str(candidate["Symbol"]).upper()
                result = results_by_symbol.get(symbol)
                if result is None:
                    error_message = "result was missing from the response"
                    research_failures_by_symbol[symbol] = error_message
                    if (
                        symbol not in unsearched_missing_refunds_this_run
                        and candidate_search_count(
                            batch_metadata["search_queries"], candidate
                        ) == 0
                    ):
                        research_attempts_by_symbol[symbol] = max(
                            0,
                            research_attempts_by_symbol.get(symbol, 0) - 1,
                        )
                        unsearched_missing_refunds_this_run.add(symbol)
                        researched_symbols_this_run.discard(symbol)
                        stock_search_attempts_this_run = max(
                            0, stock_search_attempts_this_run - 1
                        )
                        batch_stats[
                            "missing_unsearched_attempts_refunded"
                        ] += 1
                        print(
                            f"Did not charge {symbol} a research attempt: "
                            "no result or company-specific search was exposed."
                        )
                    attempt_limit = per_run_attempt_limit(symbol, True)
                    can_retry = (
                        research_round < max_validation_rounds
                        and research_attempts_by_symbol.get(symbol, 0)
                        < attempt_limit
                    )
                    if can_retry:
                        next_pending.append(candidate)
                        next_errors[symbol] = error_message
                    else:
                        exhausted_errors[symbol] = error_message
                    continue

                raw_result = json.loads(json.dumps(result, default=str))
                contamination_error = contamination_conflicts.get(symbol)
                if contamination_error:
                    prior_invalid_results_by_symbol[symbol] = result
                    research_failures_by_symbol[symbol] = contamination_error
                    batch_stats["incomplete"] += 1
                    attempt_limit = per_run_attempt_limit(symbol, True)
                    can_retry = (
                        research_round < max_validation_rounds
                        and research_attempts_by_symbol.get(symbol, 0) < attempt_limit
                    )
                    stock_research_cache["deferred_entries"][
                        cache_keys_by_symbol[symbol]
                    ] = {
                        "created_at": datetime.now(UTC).isoformat(),
                        "model": batch_model,
                        "research": result,
                        "validation_error": contamination_error,
                        "research_attempts": research_attempts_by_symbol.get(symbol, 0),
                        "repair_attempts": repair_attempts_by_symbol.get(symbol, 0),
                        "research_metadata": {
                            "search_queries": batch_metadata["search_queries"],
                            "tool_tokens": batch_metadata["tool_tokens"],
                        },
                    }
                    if can_retry:
                        next_pending.append(candidate)
                        next_errors[symbol] = contamination_error
                    else:
                        exhausted_errors[symbol] = contamination_error
                    print(
                        f"{symbol} requires fresh issuer-specific research: "
                        f"{contamination_error}"
                    )
                    continue

                minimum_sources = minimum_sources_for_candidate(
                    batch_metadata["search_queries"], candidate
                )
                try:
                    validate_stock_research_evidence(
                        {"results": [result]},
                        [candidate],
                        minimum_sources=minimum_sources,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    # Keep the normalized, otherwise usable object so the next
                    # call can repair it instead of recreating its research.
                    prior_invalid_results_by_symbol[symbol] = result
                    research_failures_by_symbol[symbol] = str(exc)
                    needs_research = validation_error_requires_fresh_research(exc)
                    if "research incomplete" in str(exc).lower():
                        batch_stats["incomplete"] += 1
                    if not needs_research:
                        batch_stats["structural_repairs_needed"] += 1
                    attempts_used = (
                        research_attempts_by_symbol.get(symbol, 0)
                        if needs_research
                        else repair_attempts_by_symbol.get(symbol, 0)
                    )
                    attempt_limit = per_run_attempt_limit(symbol, needs_research)
                    can_retry = (
                        research_round < max_validation_rounds
                        and attempts_used < attempt_limit
                    )
                    stock_research_cache["deferred_entries"][
                        cache_keys_by_symbol[symbol]
                    ] = {
                        "created_at": datetime.now(UTC).isoformat(),
                        "model": batch_model,
                        "research": result,
                        "validation_error": str(exc),
                        "research_attempts": research_attempts_by_symbol.get(
                            symbol, 0
                        ),
                        "repair_attempts": repair_attempts_by_symbol.get(
                            symbol, 0
                        ),
                        "research_metadata": {
                            "search_queries": batch_metadata["search_queries"],
                            "tool_tokens": batch_metadata["tool_tokens"],
                        },
                    }
                    if can_retry:
                        next_pending.append(candidate)
                        next_errors[symbol] = str(exc)
                        print(
                            f"Saving other valid results; {symbol} requires "
                            f"a targeted {'research' if needs_research else 'repair'} "
                            f"retry: {exc}"
                        )
                    else:
                        exhausted_errors[symbol] = str(exc)
                        print(
                            f"{symbol} remains invalid after "
                            f"{attempts_used}/{attempt_limit} permitted "
                            f"{'research' if needs_research else 'repair'} "
                            "attempts this run: "
                            f"{exc}"
                        )
                    continue

                batch_stats["validated"] += 1
                normalized_fields = {
                    field: {
                        "raw": raw_result.get(field),
                        "final": result.get(field),
                    }
                    for field in DECISION_DIAGNOSTIC_FIELDS
                    if raw_result.get(field) != result.get(field)
                }
                if normalized_fields:
                    normalization_diagnostics_by_symbol[symbol] = {
                        "symbol": symbol,
                        "changed_fields": sorted(normalized_fields),
                        "changes": normalized_fields,
                    }

                prior_invalid_results_by_symbol.pop(symbol, None)
                research_failures_by_symbol.pop(symbol, None)
                stock_research_cache["deferred_entries"].pop(
                    cache_keys_by_symbol[symbol], None
                )
                research_by_symbol[symbol] = result
                pending_classification[symbol] = result
                stock_research_cache["entries"][
                    cache_keys_by_symbol[symbol]
                ] = {
                    "created_at": datetime.now(UTC).isoformat(),
                    "model": batch_model,
                    "research_model": batch_model,
                    "judgment_model": judgment_model_used,
                    "judgment_fallback_used": bool(
                        judgment_model_used
                        and judgment_model_used != classification_model
                    ),
                    "judged_at": (
                        datetime.now(UTC).isoformat()
                        if judgment_model_used else None
                    ),
                    "market_context_hash": market_context_hash,
                    "stock_prompt_hash": stock_prompt_hash,
                    "candidate_hash": stable_json_hash(candidate),
                    "research": result,
                    "research_metadata": {
                        "search_queries": batch_metadata["search_queries"],
                        "tool_tokens": batch_metadata["tool_tokens"],
                    },
                }

            batch_research_diagnostics.append(batch_stats)
            print("Batch research summary:")
            for field, label in (
                ("candidates_sent", "candidates sent"),
                ("company_searches_exposed", "company searches exposed"),
                ("results_returned", "results returned"),
                ("validated", "results validated"),
                ("incomplete", "incomplete research"),
                ("structural_repairs_needed", "structural repairs needed"),
                ("genuine_repeated_searches", "genuine repeated searches"),
                (
                    "missing_unsearched_attempts_refunded",
                    "missing/unsearched attempts refunded",
                ),
            ):
                print(f"  {label}: {batch_stats[field]}")

            save_json_object_atomic(
                stock_research_cache_file, stock_research_cache
            )
            classify_pending()

            # Valid results from this response have already been cached. Stop
            # only after a stock has received its own complete allowance.
            if exhausted_errors:
                error_summary = "; ".join(
                    f"{symbol}: {message}"
                    for symbol, message in exhausted_errors.items()
                )
                research_failures_by_symbol.update(exhausted_errors)
                exhausted_symbols.update(exhausted_errors)
                print(
                    "Skipping exhausted invalid research and continuing to "
                    "lower-ranked candidates: " + error_summary
                )

            # Check the pending evidence before spending a 2.5 call on retries
            # or fresh candidates. A smaller 3.5 batch is worthwhile if it
            # might already complete the actual portfolio.
            provisional, _, _, _ = build_stock_portfolio(
                candidate_records, validated_cached_research
            )
            optimistic_capacity = optimistic_portfolio_capacity(
                provisional, pending_classification, candidate_by_symbol,
                max_stocks_per_sector, excluded_crypto_dependence,
            )
            if (len(provisional) < target_selected_stocks
                    and pending_classification
                    and optimistic_capacity >= target_selected_stocks):
                print(
                    "Classifying pending research before another 2.5 call: "
                    f"{len(provisional)} selected, "
                    f"{len(pending_classification)} pending, "
                    f"optimistic capacity {optimistic_capacity}/"
                    f"{target_selected_stocks}."
                )
                classify_pending(force=True)
                provisional, _, _, _ = build_stock_portfolio(
                    candidate_records, validated_cached_research
                )
            if len(provisional) >= target_selected_stocks:
                print(
                    f"Portfolio meeting final quality bar: {len(provisional)}/"
                    f"{target_selected_stocks}; stopping stock research."
                )
                break
            if classification_unavailable_this_run:
                print("Stopping retries because no classification model is available.")
                break

            if not next_pending:
                break

            # Carry retryable drafts into the next research call, then fill its
            # remaining capacity with fresh lower-ranked candidates. This keeps
            # a small no-search repair set from consuming an otherwise mostly
            # empty Gemini call. Only the fresh candidates (and retries whose
            # errors genuinely require evidence) are included in the next
            # call's required-search list.
            retry_symbols = {
                str(candidate["Symbol"]).upper()
                for candidate in next_pending
            }
            retry_queued_by_sector = {}
            for retry_candidate in next_pending:
                retry_sector = retry_candidate["Sector"]
                retry_queued_by_sector[retry_sector] = (
                    retry_queued_by_sector.get(retry_sector, 0) + 1
                )
            future_limit = (
                len(candidate_records)
                if backfill_announced
                else normal_candidate_limit
            )
            for future_candidate in candidate_records[:future_limit]:
                if len(next_pending) >= gemini_batch_size:
                    break
                future_symbol = str(future_candidate["Symbol"]).upper()
                if (
                        future_symbol in batched_symbols
                        or future_symbol in retry_symbols
                        or future_symbol in exhausted_symbols
                        or future_symbol in validated_cached_research
                    or future_symbol in pending_classification
                        or future_symbol in research_failures_by_symbol
                ):
                    continue
                if research_sector_counts.get(future_candidate["Sector"], 0) >= (
                        max_stocks_per_sector
                ):
                    continue
                future_sector = future_candidate["Sector"]
                future_open_slots = (
                    max_stocks_per_sector
                    - research_sector_counts.get(future_sector, 0)
                )
                future_sector_limit = (
                    future_open_slots
                    * max_research_candidates_per_open_sector_slot
                )
                if (
                    retry_queued_by_sector.get(future_sector, 0)
                    >= future_sector_limit
                ):
                    continue

                future_cache_key = global_cache_keys_by_symbol[future_symbol]
                valid_entry = stock_research_cache["entries"].get(
                    future_cache_key
                )
                if valid_entry and cache_entry_is_fresh(
                        valid_entry, gemini_research_cache_hours):
                    continue

                deferred = stock_research_cache["deferred_entries"].get(
                    future_cache_key
                )
                if (
                        future_cache_key in preexisting_deferred_cache_keys
                        and isinstance(deferred, dict)
                ):
                    draft = deferred.get("research")
                    error = str(
                        deferred.get("validation_error") or ""
                    ).strip()
                    if isinstance(draft, dict) and error:
                        prior_invalid_results_by_symbol[future_symbol] = draft
                        next_errors[future_symbol] = error
                        deferred_symbols_this_run.add(future_symbol)
                        research_attempts_by_symbol.setdefault(future_symbol, 0)
                        repair_attempts_by_symbol.setdefault(future_symbol, 0)
                        needs_research = validation_error_requires_fresh_research(
                            error
                        )
                        attempts_used = (
                            research_attempts_by_symbol.get(future_symbol, 0)
                            if needs_research
                            else repair_attempts_by_symbol.get(future_symbol, 0)
                        )
                        attempt_limit = per_run_attempt_limit(
                            future_symbol, needs_research
                        )
                        if attempts_used >= attempt_limit:
                            exhausted_symbols.add(future_symbol)
                            research_failures_by_symbol[future_symbol] = error
                            continue

                next_pending.append(future_candidate)
                cache_keys_by_symbol[future_symbol] = future_cache_key
                batched_symbols.add(future_symbol)
                retry_queued_by_sector[future_sector] = (
                    retry_queued_by_sector.get(future_sector, 0) + 1
                )

            added_symbols = [
                str(candidate["Symbol"]).upper()
                for candidate in next_pending
                if str(candidate["Symbol"]).upper() not in retry_symbols
            ]
            if added_symbols:
                print(
                    "Carrying retryable stocks into the next batch and filling "
                    "remaining capacity with: " + ", ".join(added_symbols)
                )
            pending_candidates = next_pending
            validation_errors = next_errors

    # Preserve validated prior-batch judgments as peer context for later Gemini
    # calls, but do not make portfolio selections yet. Final selection happens
    # only after the validated research pool is frozen.
    known_peer_symbols = {
        str(item.get("symbol") or "").upper()
        for item in prior_research_decisions
    }
    for candidate in ranked_batch:
        symbol = str(candidate["Symbol"]).upper()
        research = validated_cached_research.get(symbol)
        if research is None or symbol in known_peer_symbols:
            continue
        prior_research_decisions.append({
            "symbol": symbol,
            "industry_group": research.get("industry_group"),
            "reversal_risk": combined_reversal_risk(research),
            "business_reversal_risk": research.get("business_reversal_risk"),
            "entry_reversal_risk": research.get("entry_reversal_risk"),
            "business_concentration": research.get("business_concentration"),
            "binary_event_risk": research.get("binary_event_risk"),
            "benchmark_outperformance_outlook": research.get(
                "benchmark_outperformance_outlook"
            ),
            "continuation_strength": research.get("continuation_strength"),
            "risk_basis": research.get("risk_basis"),
            "catalyst_dependence": research.get("catalyst_dependence"),
            "crypto_dependence": research.get("crypto_dependence"),
            "mechanism_status": research.get("mechanism_status"),
            "normalization_probability": research.get(
                "normalization_probability"
            ),
            "probability_indicator_type": research.get(
                "probability_indicator_type"
            ),
            "continuation_outlook": research.get("continuation_outlook"),
            "primary_risk_event_id": research.get("primary_risk_event_id"),
            "risk_exposure_group": research.get("risk_exposure_group"),
            "company_difference": research.get("company_difference"),
        })
        known_peer_symbols.add(symbol)

classify_pending(force=True)

# Rebuild the portfolio from the entire frozen validated pool. Earlier batch
# selections were provisional only and must not prevent a later, stronger LOW/
# MODERATE candidate from displacing them.
selected, decision_ledger, sector_counts, scored_validated_candidates = (
    build_stock_portfolio(candidate_records, validated_cached_research, verbose=True)
)

# Keep failed research visible in diagnostics even though it cannot enter the
# frozen comparison pool.
scored_symbols = {str(item[2]["Symbol"]).upper() for item in scored_validated_candidates}
for candidate in candidate_records:
    symbol = str(candidate["Symbol"]).upper()
    if symbol in scored_symbols:
        continue
    failure_reason = research_failures_by_symbol.get(symbol)
    if failure_reason:
        decision_ledger.append({
            "qvm_rank": candidate["QVM Rank"],
            "symbol": symbol,
            "sector_group": candidate["Sector"],
            "status": "SKIPPED — RESEARCH INVALID",
            "sector_selected_after": sector_counts.get(candidate["Sector"], 0),
            "total_selected_after": len(selected),
            "final_selection_score": None,
            "continuation_strength": None,
            "benchmark_outperformance_outlook": None,
            "explanation": "Research did not pass validation: " + failure_reason,
        })
    elif symbol in classification_failures_by_symbol:
        decision_ledger.append({
            "qvm_rank": candidate["QVM Rank"],
            "symbol": symbol,
            "sector_group": candidate["Sector"],
            "status": "SKIPPED — CLASSIFICATION INVALID",
            "sector_selected_after": sector_counts.get(candidate["Sector"], 0),
            "total_selected_after": len(selected),
            "final_selection_score": None,
            "continuation_strength": None,
            "benchmark_outperformance_outlook": None,
            "explanation": "Judgment did not pass final validation: "
                           + classification_failures_by_symbol[symbol],
        })

validate_final_research_state(validated_cached_research)

print(
    f"Research efficiency: {len(researched_symbols_this_run)} unique stocks, "
    f"{stock_search_attempts_this_run} requested searches, "
    f"{stock_search_attempts_this_run - len(researched_symbols_this_run)} "
    "repeated searches."
)

context_review = build_context_review(decision_ledger)
print("\n" + context_review + "\n")

current_classifications = build_classification_snapshots(
    candidate_records, validated_cached_research
)
previous_classifications = previous_run_diagnostics.get(
    "classifications", {}
)
classification_drift = compare_classification_snapshots(
    previous_classifications, current_classifications
)
print("CLASSIFICATION DRIFT VERSUS PREVIOUS SUCCESSFUL RUN")
if classification_drift:
    for item in classification_drift:
        symbol = item["symbol"]
        print(f"  {symbol}: {', '.join(item['changed_fields'])}")
        for field, values in item["changes"].items():
            if field in DECISION_DIAGNOSTIC_FIELDS:
                print(
                    f"    {field}: "
                    f"{compact_diagnostic_value(values['previous'])} -> "
                    f"{compact_diagnostic_value(values['current'])}"
                )
else:
    print("  No comparable classification fields changed.")

current_selected_symbols = [
    str(item["candidate"]["Symbol"]).upper() for item in selected
]
previous_selected_symbols = previous_run_diagnostics.get(
    "selected_symbols", []
)
current_decisions = build_decision_snapshots(decision_ledger)
previous_decisions = previous_run_diagnostics.get("decisions", {})
portfolio_changes = build_portfolio_changes(
    previous_selected_symbols,
    current_selected_symbols,
    previous_decisions,
    current_decisions,
    classification_drift,
)
print("PORTFOLIO CHANGES VERSUS PREVIOUS SUCCESSFUL RUN")
if not previous_run_diagnostics:
    print("  No prior diagnostics snapshot is available; baseline created.")
elif portfolio_changes["added"] or portfolio_changes["removed"]:
    for item in portfolio_changes["added"]:
        changed = item["classification_fields_changed"]
        detail = f"; changed fields: {', '.join(changed)}" if changed else ""
        print(
            f"  Added {item['symbol']}: {item['previous_status']} -> "
            f"{item['current_status']}{detail}"
        )
    for item in portfolio_changes["removed"]:
        changed = item["classification_fields_changed"]
        detail = f"; changed fields: {', '.join(changed)}" if changed else ""
        print(
            f"  Removed {item['symbol']}: {item['previous_status']} -> "
            f"{item['current_status']}{detail}"
        )
else:
    print("  No selected symbols changed.")

reconciliation_counts = {}
for item in runtime_reconciliation_diagnostics:
    item_type = item["type"]
    reconciliation_counts[item_type] = (
        reconciliation_counts.get(item_type, 0) + 1
    )
print("NO-CALL NORMALIZATION SUMMARY")
if reconciliation_counts:
    for item_type, count in sorted(reconciliation_counts.items()):
        print(f"  {item_type}: {count}")
else:
    print("  No Python label reconciliations were needed.")

if not selected:
    raise RuntimeError("No stocks passed the full research and classification rules.")
if len(selected) < target_selected_stocks:
    print(
        f"PORTFOLIO SHORTFALL: {len(selected)}/{target_selected_stocks}; "
        f"QVM pool={len(candidate_records)}, researched unique="
        f"{len(researched_symbols_this_run)}, classified="
        f"{len(validated_cached_research)}, invalid="
        f"{len(research_failures_by_symbol)}, judgment invalid="
        f"{len(classification_failures_by_symbol)}, unresearched="
        f"{len(candidate_records) - len(set(validated_cached_research) | set(research_failures_by_symbol) | researched_symbols_this_run)}. "
        "Publishing qualified stocks without relaxing selection rules."
    )

recommendations_table = build_recommendations_table(selected)
recommendations_summary = build_recommendations_summary(
    market_context, selected
)

if final_summary_enabled:
    selected_summary_input = [
        {
            "Symbol": item["candidate"]["Symbol"],
            "Name": item["candidate"]["Name"],
            "Sector Group": item["candidate"]["Sector"],
            "Reversal Risk": item["research"]["reversal_risk"],
            "Explanation": item["research"]["explanation"],
            "Business Description": item["research"]["business_description"],
            "Industry Group": item["research"].get("industry_group"),
            "Industry Context": item["research"]["industry_context"],
            "Current Operating Evidence": item["research"].get(
                "current_operating_evidence"
            ),
            "Probability Evidence": item["research"].get(
                "probability_evidence"
            ),
            "Material Effect": item["research"].get("material_effect"),
            "Durable Drivers": item["research"].get("durable_drivers", []),
            "Reversal Mechanism": item["research"].get(
                "reversal_mechanism"
            ),
        }
        for item in selected
    ]
    summary_prompt = (
        config["prompt_html_summary"]
        + "\n\nMARKET_CONTEXT:\n"
        + json.dumps(market_context, ensure_ascii=False)
        + "\n\nSELECTED_STOCKS:\n"
        + json.dumps(selected_summary_input, ensure_ascii=False)
    )
    try:
        summary_data, summary_model, _ = call_gemini_json(
            client=client,
            model_primary=model_primary,
            # Summary generation is lower-risk than stock classification, so
            # Flash-Lite is an acceptable fallback when Flash has exhausted its
            # separate per-model daily quota.
            model_fallback=model_fallback,
            gemini_config=build_gemini_config(
                summary_thinking_budget, enable_search=False
            ),
            prompt=summary_prompt,
            stage="final HTML summary",
            validator=lambda data: validate_summary_response(data, selected),
            request_budget=request_budget,
            require_google_search=False,
            max_attempts=1,
            budget_category="summary",
        )
        recommendations_summary = summary_data["summary_html"].strip()
        models_used.append(summary_model)
        print("Using Gemini-written HTML summary.")
    except Exception as exc:
        print(
            "Warning: final Gemini summary was unavailable; using the "
            f"deterministic Python summary instead: {exc}"
        )

# The full QVM table remains the complete ranked candidate stream.
output_columns = [
    'Symbol', 'Name', 'Sector', '52 WkChange %', '3M Return', 'QVMScore'
]
df_html = df_gemini[output_columns].copy()
df_html["_RawSymbol"] = df_html["Symbol"].astype(str)
df_html["Symbol"] = df_html["Symbol"].apply(
    lambda symbol: yahoo_link(symbol, symbol)
)
df_html["Name"] = df_html.apply(
    lambda row: yahoo_link(row["_RawSymbol"], row["Name"]),
    axis=1
)
df_html = df_html.drop(columns=["_RawSymbol"])
df_html_table = df_html.to_html(
    escape=False,
    index=False,
    classes="recommendations-table",
    border=0
)

model_used = ", ".join(dict.fromkeys(models_used))
update_html_page(
    recommendations_table,
    recommendations_summary,
    df_html_table,
    "stock_page_template.html",
    "index.html",
    model_used,
)

end_time = time.perf_counter()
stock_call_diagnostics = [
    item for item in gemini_call_diagnostics
    if str(item.get("stage", "")).startswith("stock batch")
]

def summed_call_metric(field):
    return sum(
        int(item[field])
        for item in gemini_call_diagnostics
        if isinstance(item.get(field), (int, float))
    )


total_duplicate_blocks = sum(
    int(item.get("extra_blocks", 0))
    for item in duplicate_result_diagnostics
)
newly_validated_symbols = sorted(
    researched_symbols_this_run.intersection(current_classifications)
)
source_counts = {
    symbol: snapshot["source_count"]
    for symbol, snapshot in current_classifications.items()
}
stock_analysis = build_stock_analysis_snapshots(
    candidate_records,
    validated_cached_research,
    stock_research_cache.get("entries", {}),
    current_decisions,
)

# Keep the evidence behind a selected label visible in the plain run log.
# Missing fields on older cached research are marked unknown, not "no event".
selected_evidence_audit = []
for symbol in current_selected_symbols:
    snapshot = stock_analysis[symbol]
    provenance = snapshot["research_provenance"]
    source_dates = provenance["source_publication_dates"]
    audit = {
        "symbol": symbol,
        "benchmark_outlook": snapshot["classification"].get(
            "benchmark_outperformance_outlook"
        ),
        "benchmark_excess_returns": snapshot["benchmark_excess_returns"],
        "benchmark_outperformance_basis": snapshot[
            "benchmark_outperformance_basis"
        ],
        "material_company_event_review": snapshot[
            "material_company_event_review"
        ],
        "material_company_event": snapshot["material_company_event"],
        "research_age_hours": provenance["age_hours_at_report"],
        "sources_with_publication_date": sum(
            bool(source.get("published_at")) for source in source_dates
        ),
        "source_count": provenance["source_count"],
    }
    selected_evidence_audit.append(audit)
print("SELECTED EVIDENCE AUDIT (missing event/date fields are unknown)")
for audit in selected_evidence_audit:
    print("  " + json.dumps(audit, ensure_ascii=False, default=str))

run_report = {
    "schema_version": 2,
    "created_at": datetime.now(UTC).isoformat(),
    "elapsed_seconds": round(end_time - start_time),
    "models": {
        "research_primary": model_primary,
        "research_fallback": model_fallback,
        "classification_primary": classification_model,
        "classification_fallback": classification_fallback_model,
        "models_used_this_run": list(dict.fromkeys(models_used)),
    },
    "qvm_candidate_hash": stable_json_hash(candidate_records),
    "benchmark_context": benchmark_context,
    "yfinance_fundamentals": yfinance_fundamentals_diagnostics,
    "request_budget": {
        "used": request_budget.used,
        "maximum": request_budget.maximum,
        "stock_used": request_budget.stock_used,
        "stock_maximum": request_budget.stock_maximum,
        "api_attempts": request_budget.api_attempts,
        "stock_api_attempts": request_budget.stock_api_attempts,
        "classification_api_attempts_used": classification_calls_used,
        "classification_api_attempts_maximum": max_classification_calls_per_run,
    },
    "gemini_call_ledger": gemini_attempt_diagnostics,
    "successful_gemini_metadata": gemini_call_diagnostics,
    "classification_calls": classification_call_diagnostics,
    "classification_validation_failures": classification_failures_by_symbol,
    "classification_failure_details": classification_failure_details_by_symbol,
    "token_totals": {
        field: summed_call_metric(field)
        for field in (
            "prompt_tokens", "tool_tokens", "cached_tokens",
            "thinking_tokens", "output_tokens", "total_tokens",
        )
    },
    "research": {
        "unique_symbols_requested": len(researched_symbols_this_run),
        "charged_search_attempts": stock_search_attempts_this_run,
        "attempted_stock_requests": request_budget.stock_used,
        "completed_stock_responses": len(stock_call_diagnostics),
        "newly_validated_symbols": newly_validated_symbols,
        "initial_validated_cache_symbols": sorted(initial_validated_cache_symbols),
        "cached_research_records_used": sum(
            1 for item in stock_analysis.values()
            if not item["research_provenance"]["fresh_this_run"]
        ),
        "cached_judgment_records_used": sum(
            1 for item in stock_analysis.values()
            if item["judgment_provenance"]["model"]
            and not item["judgment_provenance"]["fresh_this_run"]
        ),
        "source_counts": source_counts,
        "batches": batch_research_diagnostics,
    },
    "stocks": stock_analysis,
    "selected_evidence_audit": selected_evidence_audit,
    "classifications": current_classifications,
    "classification_drift": classification_drift,
    "raw_to_final_normalizations": sorted(
        normalization_diagnostics_by_symbol.values(),
        key=lambda item: item["symbol"],
    ),
    "runtime_reconciliations": runtime_reconciliation_diagnostics,
    "reconciliation_counts": reconciliation_counts,
    "duplicate_results": duplicate_result_diagnostics,
    "duplicate_extra_block_count": total_duplicate_blocks,
    "decisions": current_decisions,
    "selected_symbols": current_selected_symbols,
    "portfolio_changes": portfolio_changes,
    "market_context": {
        "as_of_date": market_context.get("as_of_date"),
        "market_status": market_context.get("market_status"),
        "market_summary": market_context.get("market_summary") or market_context.get("market_intro"),
        "active_risk_events": market_context.get("active_risk_events", []),
        "sources": market_context.get("sources", []),
    },
}
save_json_object_atomic(run_report_file, run_report)

print("GEMINI EFFICIENCY SUMMARY")
print(f"  requests used: {request_budget.used}/{request_budget.maximum}")
print(
    f"  stock research calls: {request_budget.stock_used}/"
    f"{request_budget.stock_maximum}"
)
print(f"  newly validated stocks: {len(newly_validated_symbols)}")
if stock_call_diagnostics:
    print(
        "  newly validated stocks per completed stock response: "
        f"{len(newly_validated_symbols) / len(stock_call_diagnostics):.2f}"
    )
if request_budget.stock_used:
    print(
        "  newly validated stocks per attempted stock request: "
        f"{len(newly_validated_symbols) / request_budget.stock_used:.2f}"
    )
print(
    f"  exposed searches per newly validated stock: "
    f"{(sum(item['searches_exposed'] for item in stock_call_diagnostics) / len(newly_validated_symbols)) if newly_validated_symbols else 0:.2f}"
)
print(f"  duplicate extra result blocks: {total_duplicate_blocks}")
print(
    f"  tokens: prompt={run_report['token_totals']['prompt_tokens']}, "
    f"tool={run_report['token_totals']['tool_tokens']}, "
    f"thinking={run_report['token_totals']['thinking_tokens']}, "
    f"output={run_report['token_totals']['output_tokens']}, "
    f"total={run_report['token_totals']['total_tokens']}"
)
print(f"Saved comprehensive run report: {run_report_file}")
print(f"Generated research with model(s): {model_used}")
print("Selected symbols: " + ", ".join(
    item["candidate"]["Symbol"] for item in selected
))

# print elapsed time
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")
signal.alarm(0)
