import re
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from google import genai
from pathlib import Path
from bs4 import BeautifulSoup
from google.genai import types
from datetime import datetime, timedelta, UTC
import requests, time, random, json, yaml, os, hashlib
from html import escape

CACHE_DIR = Path("caches")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOP_QVM_STOCKS_MD_FILE = CACHE_DIR / "top_qvm_stocks.md"

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

    return genai.Client(api_key=api_key)


def build_gemini_config(thinking_budget, enable_search=True):
    """Create a low-variance Gemini configuration."""
    tools = None
    if enable_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(
        tools=tools,
        temperature=0,
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget
        )
    )


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

    def consume(self, stage, category="general"):
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
        print(
            f"Gemini request {self.used}/{self.maximum}: {stage}"
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
    """Return the event-cap key for one economically comparable exposure."""
    event_id = re.sub(
        r"[^A-Z0-9]+", "_",
        str(research.get("primary_risk_event_id") or "").upper(),
    ).strip("_")
    if not event_id:
        return None
    exposure_group = re.sub(
        r"[^A-Z0-9]+", "_",
        str(research.get("risk_exposure_group") or "").upper(),
    ).strip("_")
    return event_id, exposure_group


def preview_selectable_symbols(
        candidates,
        research_by_symbol,
        target_count,
        sector_limit,
        moderate_event_limit,
        excluded_crypto_levels,
        exclude_high_catalyst):
    """Replay portfolio rules without mutating the real selection ledger."""
    preview_selected = []
    preview_sector_counts = {}
    preview_event_counts = {}

    for candidate in candidates:
        if len(preview_selected) >= target_count:
            break

        symbol = str(candidate["Symbol"]).upper()
        sector = candidate["Sector"]
        if preview_sector_counts.get(sector, 0) >= sector_limit:
            continue

        research = research_by_symbol.get(symbol)
        if research is None or not bool(research.get("eligible", True)):
            continue
        if excluded_by_crypto_policy(research, excluded_crypto_levels):
            continue
        if excluded_by_high_catalyst_policy(research, exclude_high_catalyst):
            continue

        reversal_risk = str(research.get("reversal_risk", "")).upper()
        if reversal_risk in {"ELEVATED", "SEVERE"}:
            continue
        if reversal_risk not in {"MINIMAL", "LOW", "MODERATE"}:
            continue

        event_key = normalized_risk_event_key(research)

        if (
            reversal_risk == "MODERATE"
            and event_key
            and preview_event_counts.get(event_key, 0)
            >= moderate_event_limit
        ):
            continue

        preview_selected.append(symbol)
        preview_sector_counts[sector] = (
            preview_sector_counts.get(sector, 0) + 1
        )
        if reversal_risk == "MODERATE" and event_key:
            preview_event_counts[event_key] = (
                preview_event_counts.get(event_key, 0) + 1
            )

    return preview_selected


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
    )
    return any(marker in error for marker in research_markers)


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


def deduplicate_stock_results(results, expected_candidates):
    """Keep one recovered object per symbol in authoritative input order."""
    expected_order = [
        str(candidate["Symbol"]).strip().upper()
        for candidate in expected_candidates
    ]
    first_by_symbol = {}
    unsymbolized = []
    duplicate_symbols = []
    conflicting_symbols = []

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
        if result != first_by_symbol[symbol]:
            conflicting_symbols.append(symbol)

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

    ordered = [
        first_by_symbol.pop(symbol)
        for symbol in expected_order
        if symbol in first_by_symbol
    ]
    # Retain unexpected symbols so the existing validator reports them.
    ordered.extend(first_by_symbol.values())
    ordered.extend(unsymbolized)
    return ordered


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
    validate_sources(data["sources"], "Market context")


ALLOWED_REVERSAL_RISKS = {
    "MINIMAL",
    "LOW",
    "MODERATE",
    "ELEVATED",
    "SEVERE",
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
    r"\b(?:refund|settlement|recovery|asset[ -]sale gain|tax benefit|"
    r"tax credit|accounting gain|one[ -](?:time|off) gain)\b",
    re.IGNORECASE,
)


def has_nonrecurring_temporary_item(temporary_drivers):
    """Detect comparison adjustments incorrectly modeled as operating drivers."""
    return any(
        NONRECURRING_TEMPORARY_ITEM_PATTERN.search(str(driver))
        for driver in temporary_drivers or []
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


def excluded_by_high_catalyst_policy(research, enabled):
    return bool(
        enabled
        and str(research.get("reversal_risk") or "").upper() == "MODERATE"
        and str(research.get("risk_basis") or "").upper()
        == "TEMPORARY_DRIVER_NORMALIZATION"
        and str(research.get("catalyst_dependence") or "").upper() == "HIGH"
    )


def downgrade_unproven_material_risk(result, symbol, reason):
    """Apply the prompt's LOW mapping when probability was not established."""
    prior_risk = str(result.get("reversal_risk", "")).upper()
    result["reversal_risk"] = "LOW"
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


def validate_stock_batch(data, expected_candidates, minimum_sources=2):
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
            raise ValueError(f"{symbol} research incomplete: {reason}")

        reversal_risk = str(result.get("reversal_risk", "")).upper()
        if reversal_risk not in ALLOWED_REVERSAL_RISKS:
            raise ValueError(f"{symbol} has invalid reversal_risk {reversal_risk!r}.")
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

        risk_exposure_group = result.get("risk_exposure_group")
        if risk_exposure_group is not None:
            risk_exposure_group = re.sub(
                r"[^A-Z0-9]+", "_", str(risk_exposure_group).upper()
            ).strip("_") or None
        if result.get("primary_risk_event_id") and not risk_exposure_group:
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

        reconciliation_reason = None
        if reversal_risk in evidence_required_risks and not exposure_floor:
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
                if exposure_floor
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

        if risk_basis == "NONRECURRING_COMPARISON_ONLY" and reversal_risk not in {
            "MINIMAL", "LOW"
        }:
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
            raise ValueError(
                f"{result.get('symbol')} differs from {other.get('symbol')} on "
                f"shared event {event_id} ({', '.join(differences)}) without "
                "a sourced company_difference."
            )


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
    last_error = None
    for model_name in models:
        for attempt in range(attempt_limit):
            attempt_prompt = prompt
            metadata = None
            try:
                request_budget.consume(
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
                return data, model_name, metadata
            except Exception as exc:
                last_error = exc
                error_text = str(exc).upper()
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
                    )
                )

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
# This cache stores the fully computed top-50 QVM DataFrame so that
# Gemini prompt testing can skip the expensive stock-data/QVM pipeline.
# The cache file can be persisted by GitHub Actions in the same way as
# the existing JSON caches.
TOP_QVM_CACHE_FILE = cache_file_path("top_qvm_stocks_cache.pkl")
TOP_QVM_CACHE_EXPIRY_HOURS = 6
TOP_QVM_CACHE_VERSION = 1


def load_top_qvm_cache():
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

def save_top_qvm_cache(df):
    """Save the fully computed top-QVM DataFrame for later Gemini testing."""
    try:
        cache = {
            "version": TOP_QVM_CACHE_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "data": df.copy()
        }

        pd.to_pickle(cache, TOP_QVM_CACHE_FILE)

        print(f"Saved top QVM cache to {TOP_QVM_CACHE_FILE}.")

    except Exception as e:
        print(f"Could not save top QVM cache: {e}")

def append_qvm_data_yfinance(
        df: pd.DataFrame,
        max_info_calls: int = 500,
        delay: float = 0.5
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

            ret_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
            ret_9m = ((close.iloc[-1] / close.iloc[-189]) - 1) * 100 if len(close) > 189 else None
            ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else None
            ret_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else None
            ret_1m = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 21 else None

            latest_price = float(close.iloc[-1])   # ← This is the fix

            score = np.nanmean([ret_3m, ret_6m, ret_9m])
            momentum_scores[symbol] = score

            data_map[symbol] = {
                "Price": latest_price,                  # ← Added
                "1M Return": ret_1m,
                "3M Return": ret_3m,
                "6M Return": ret_6m,
                "9M Return": ret_9m,
                "1Y Return": ret_1y
            }

        except:
            momentum_scores[symbol] = -np.inf
            data_map[symbol] = {}

    # ---- STEP 2: Select top N for expensive calls ----
    sorted_symbols = sorted(momentum_scores, key=lambda x: momentum_scores[x], reverse=True)
    selected_for_info = set(sorted_symbols[:max_info_calls])

    print(f"Fetching fundamentals for top {len(selected_for_info)} tickers...")

    # ---- STEP 3: Fetch info (unchanged) ----
    for symbol in tqdm(selected_for_info):
        try:
            if symbol in cache:
                info = cache[symbol]["info"]
            else:
                ticker_obj = yf.Ticker(symbol)
                info = ticker_obj.info

                cache[symbol] = {
                    "info": {
                        "sector": info.get("sector"),
                        "returnOnEquity": info.get("returnOnEquity"),
                        "returnOnAssets": info.get("returnOnAssets"),
                        "profitMargins": info.get("profitMargins"),
                        "grossMargins": info.get("grossMargins"),
                        "debtToEquity": info.get("debtToEquity"),
                        "currentRatio": info.get("currentRatio"),
                        "interestCoverage": info.get("interestCoverage"),
                        "trailingPE": info.get("trailingPE"),
                        "priceToBook": info.get("priceToBook"),
                        "pegRatio": info.get("pegRatio"),
                        "enterpriseValue": info.get("enterpriseValue"),
                        "ebitda": info.get("ebitda"),
                        "totalRevenue": info.get("totalRevenue"),
                    },
                    "timestamp": datetime.now(UTC).isoformat()
                }
                time.sleep(delay + random.uniform(0, 0.3))

            # compute derived metrics
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")
            revenue = info.get("totalRevenue")

            data_map[symbol].update({
                "Sector": info.get("sector"),
                "ROE": info.get("returnOnEquity"),
                "ROA": info.get("returnOnAssets"),
                "ProfitMargin": info.get("profitMargins"),
                "GrossMargin": info.get("grossMargins"),
                "DebtToEquity": info.get("debtToEquity"),
                "CurrentRatio": info.get("currentRatio"),
                "InterestCoverage": info.get("interestCoverage"),
                "PE": info.get("trailingPE"),
                "PriceToBook": info.get("priceToBook"),
                "PEG": info.get("pegRatio"),
                "EV_EBITDA": (ev / ebitda) if ev and ebitda and ebitda != 0 else None,
                "EV_Revenue": (ev / revenue) if ev and revenue and revenue != 0 else None,
            })

        except Exception as e:
            print(f"Error on {symbol}: {e}")

    # ---- Save cache ----
    save_cache(cache)

    # ---- Map back to df ----
    all_columns = [
        "Sector", "Price", "ROE", "ROA", "ProfitMargin", "GrossMargin",
        "DebtToEquity", "CurrentRatio", "InterestCoverage",
        "PE", "PriceToBook", "PEG", "EV_EBITDA", "EV_Revenue",
        "1M Return", "3M Return", "6M Return", "9M Return", "1Y Return"
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

    Default weights:
      Quality  = 45%
      Value    = 15%
      Momentum = 40%
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
        '1Y Return'
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

        # Mean only across available metrics.
        quality = q_rank.mean(
            axis=1,
            skipna=True
        )

        # -----------------------------------------------------
        # Missing-data confidence adjustment
        #
        # 2+ valid metrics = full confidence
        # 1 valid metric  = reduced confidence
        # 0 valid metrics = neutral score
        # -----------------------------------------------------

        quality_confidence = np.where(
            valid_quality_count >= 2,
            1.00,
            np.where(
                valid_quality_count == 1,
                0.75,
                0.50
            )
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

            debt = (
                df['DebtToEquity']
                .fillna(0)
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

        df['QualityScore'] = (
                quality * 100
        ).clip(0, 100)

    else:
        df['QualityScore'] = 50

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
        # D. Risk-adjusted momentum
        # -----------------------------------------------------

        volatility = m.std(
            axis=1,
            skipna=True
        )

        risk_adjusted_return = (
                mean_return /
                (volatility + 1e-5)
        )

        risk_adj = (
                risk_adjusted_return.rank(
                    pct=True
                ) * 100
        )

        risk_adj = (
            risk_adj.fillna(50)
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

        # -----------------------------------------------------
        # F. Final momentum blend
        #
        # 30% overall return
        # 30% persistence
        # 20% multi-period consistency
        # 10% risk adjustment
        # 10% recent confirmation
        # -----------------------------------------------------

        momentum = (
                0.30 * raw_momentum +
                0.30 * trend_alignment +
                0.20 * consistency_score +
                0.10 * risk_adj +
                0.10 * recent_rank
        )

        df['MomentumScore'] = (
            momentum
            .fillna(50)
            .clip(0, 100)
        )

    else:
        df['MomentumScore'] = 50

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
        "Sector Selected After", "Total Selected After", "Explanation"
    ]
    rows = []
    for decision in decision_ledger:
        rows.append([
            decision["qvm_rank"],
            decision["symbol"],
            decision["sector_group"],
            decision["status"],
            decision["sector_selected_after"],
            decision["total_selected_after"],
            decision["explanation"],
        ])
    return "CONTEXT REVIEW\n\n" + pd.DataFrame(rows, columns=headers).to_markdown(index=False)


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



# Main execution
# Record the start time
start_time = time.perf_counter()

with open("stock_config.yml") as f:
    config = yaml.safe_load(f)
url = config["url"]
min_52_week_change = config["min_52_week_change"]
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
gemini_batch_size = int(config.get("gemini_batch_size", 5))
max_research_candidates_per_open_sector_slot = max(
    1,
    int(config.get("max_research_candidates_per_open_sector_slot", 3)),
)
max_candidates = int(config.get("max_candidates", 50))
target_selected_stocks = int(config.get("target_selected_stocks", 10))
max_stocks_per_sector = int(config.get("max_stocks_per_sector", 2))
max_moderate_per_risk_event = int(
    config.get("max_moderate_per_risk_event", 2)
)
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
exclude_high_catalyst_dependence = bool(
    config.get("exclude_high_catalyst_dependence", True)
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
    f"exclude_high_catalyst={exclude_high_catalyst_dependence}, "
    f"excluded_crypto={sorted(excluded_crypto_dependence)}"
)

# ---------------------------------------------------------
# Load cached top-50 QVM DataFrame when fresh.
#
# This check happens BEFORE any stock-page, yfinance, or
# QVM-scoring work, so Gemini prompt testing can reuse the
# exact same quantitative input without rerunning the
# expensive pipeline.
# ---------------------------------------------------------
top_stocks = load_top_qvm_cache()

if top_stocks is None:

    df = fetch_all_stock_pages_from_url(url, min_52_week_change)
    df = df.drop_duplicates()

    # Filter rules
    df = df[
        (df['Market Cap'] >= 300_000_000) &
        (df['Price'] >= 5.0) &
        (df['Avg Vol (3M)'] >= 75_000) &
        ((df['P/E Ratio(TTM)'].isna()) | (df['P/E Ratio(TTM)'] > 0))
    ].copy()

    print("\nTrash Filtered Stocks:")
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

    df_yf = append_qvm_data_yfinance(df_minimal)
    df_scored = score_qvm(df_yf, weights = {
        'Quality': 0.45,
        'Value': 0.10,
        'Momentum': 0.45
    })

    # Keep the configured top QVM candidate stream for Gemini evaluation.
    top_stocks = df_scored.head(max_candidates).copy()

    save_top_qvm_cache(top_stocks)

print("\nTop QVM Stocks:")
top_stocks = top_stocks.head(max_candidates).copy().reset_index(drop=True)
print(top_stocks[['Symbol', 'QVMScore', '3M Return','1Y Return']])

cols_for_eval = [
    'Symbol',
    'QVMScore',
    'QualityScore',
    'ValueScore',
    'MomentumScore',
    'ROE',
    'DebtToEquity',
    'EV_EBITDA',
    'PEG',
    '3M Return',
    '6M Return',
    '1Y Return'
]
#print to file for inspection and evaluation
TOP_QVM_STOCKS_MD_FILE.write_text(
    top_stocks[cols_for_eval].to_markdown(index=False),
    encoding="utf-8",
)

essential_columns_for_gemini = [
    # Identity
    "Symbol", "Name", "Sector",

    # Size / context
    "Market Cap",

    # Core QVM output (most important)
    "QVMScore",

    # Decomposed signals (compressed)
    "QualityScore",
    "ValueScore",
    "MomentumScore",

    # Key fundamentals
    "ROE",
    "ProfitMargin",
    "DebtToEquity",
    "EV_EBITDA",
    "PEG",

    # Valuation anchor
    "PE",

    # Momentum anchors
    "3M Return",
    "6M Return",
    "9M Return",
    "52 WkChange %"
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

print("\nValidated market context:")
print(json.dumps(market_context, indent=2, ensure_ascii=False))

market_context_hash = stable_json_hash(market_context)
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
})
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
    if not entry:
        continue
    try:
        cached_result = entry["research"]
        cached_search_queries = entry.get(
            "research_metadata", {}
        ).get("search_queries", [])
        validate_stock_batch(
            {"results": [cached_result]},
            [candidate],
            minimum_sources=minimum_sources_for_candidate(
                cached_search_queries, candidate
            ),
        )
        validated_cached_research[symbol] = cached_result
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Ignoring invalid stock cache entry for {symbol}: {exc}")

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

if cache_consistency_changed:
    save_json_object_atomic(stock_research_cache_file, stock_research_cache)

cached_preview_symbols = preview_selectable_symbols(
    candidate_records,
    validated_cached_research,
    target_selected_stocks,
    max_stocks_per_sector,
    max_moderate_per_risk_event,
    excluded_crypto_dependence,
    exclude_high_catalyst_dependence,
)
cache_already_fills_portfolio = (
    len(cached_preview_symbols) >= target_selected_stocks
)
if cache_already_fills_portfolio:
    print(
        "Validated stock cache already supports the full portfolio; "
        "no new Gemini stock research is required. Preview: "
        + ", ".join(cached_preview_symbols)
    )

selected = []
decision_ledger = []
sector_counts = {}
moderate_event_counts = {}
prior_research_decisions = []
research_failures_by_symbol = {}
models_used = [market_model]
researched_symbols_this_run = set()
stock_search_attempts_this_run = 0
prior_invalid_results_by_symbol = {}
research_attempts_by_symbol = {}
repair_attempts_by_symbol = {}
deferred_symbols_this_run = set()


def per_run_attempt_limit(symbol, needs_research):
    """Bound retries within this execution without permanently locking drafts."""
    if needs_research:
        if symbol in deferred_symbols_this_run:
            return max_deferred_research_attempts_per_run
        return max_research_attempts_per_stock
    return max_structural_repairs_per_stock

batch_start = 0
carried_ranked_candidates = []
while batch_start < len(candidate_records) or carried_ranked_candidates:
    if len(selected) >= target_selected_stocks:
        break

    new_ranked_candidates = candidate_records[
        batch_start:batch_start + gemini_batch_size
    ]
    batch_start += gemini_batch_size
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
        sector_counts,
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
        cached_result = validated_cached_research.get(symbol)
        if cached_result is not None:
            research_by_symbol[symbol] = cached_result
            entry = stock_research_cache["entries"].get(cache_key, {})
            models_used.append(entry.get("model") or model_primary)
            print(f"Using validated stock research cache for {symbol}.")
            continue
        uncached_candidates.append(candidate)

    cached_candidate_count = (
        len(research_candidates) - len(uncached_candidates)
    )

    if cache_already_fills_portfolio:
        uncached_candidates = []

    # Keep the first ordinary call full, but do not pad later calls to 18 after
    # the portfolio is almost complete. Three alternatives per remaining slot,
    # with a floor of five, balances sector/risk exclusions against search cost.
    ordinary_uncached_count = len(uncached_candidates)
    remaining_selection_slots = max(
        1, target_selected_stocks - len(selected)
    )
    desired_research_count = min(
        gemini_batch_size,
        max(5, remaining_selection_slots * 3),
    )
    if (
        uncached_candidates
        and cached_candidate_count == 0
        and len(uncached_candidates) < desired_research_count
    ):
        queued_symbols = {
            str(candidate["Symbol"]).upper()
            for candidate in uncached_candidates
        }
        future_start = batch_start
        for future_candidate in candidate_records[future_start:]:
            if len(uncached_candidates) >= desired_research_count:
                break
            future_symbol = str(future_candidate["Symbol"]).upper()
            if future_symbol in queued_symbols:
                continue
            if sector_counts.get(future_candidate["Sector"], 0) >= (
                max_stocks_per_sector
            ):
                continue
            future_sector = future_candidate["Sector"]
            future_open_slots = (
                max_stocks_per_sector - sector_counts.get(future_sector, 0)
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
            if not isinstance(deferred, dict):
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
                + "\n\nMARKET_CONTEXT:\n"
                + json.dumps(market_context, ensure_ascii=False)
                + "\n\nPRIOR_RESEARCH_DECISIONS:\n"
                + json.dumps(prior_research_decisions, ensure_ascii=False)
                + "\n\nCANDIDATES:\n"
                + json.dumps(pending_candidates, ensure_ascii=False)
            )
            stage = (
                f"stock batch ranks {pending_candidates[0]['QVM Rank']}-"
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
            print(
                f"Stock response coverage: {len(results_by_symbol)}/"
                f"{len(pending_candidates)} returned; "
                f"{len(missing_response_symbols)} missing."
            )
            print(
                f"Research searches requested this call: "
                f"{len(requested_research_symbols)} total; "
                f"{len(requested_research_symbols) - repeated_searches} first, "
                f"{repeated_searches} repeated."
            )
            next_pending = []
            next_errors = {}
            exhausted_errors = {}

            for candidate in pending_candidates:
                symbol = str(candidate["Symbol"]).upper()
                result = results_by_symbol.get(symbol)
                if result is None:
                    error_message = "result was missing from the response"
                    research_failures_by_symbol[symbol] = error_message
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

                minimum_sources = minimum_sources_for_candidate(
                    batch_metadata["search_queries"], candidate
                )
                try:
                    validate_stock_batch(
                        {"results": [result]},
                        [candidate],
                        minimum_sources=minimum_sources,
                    )
                    validate_shared_event_consistency(
                        result,
                        list(research_by_symbol.values())
                        + prior_research_decisions,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    # Keep the normalized, otherwise usable object so the next
                    # call can repair it instead of recreating its research.
                    prior_invalid_results_by_symbol[symbol] = result
                    research_failures_by_symbol[symbol] = str(exc)
                    needs_research = validation_error_requires_fresh_research(exc)
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

                prior_invalid_results_by_symbol.pop(symbol, None)
                stock_research_cache["deferred_entries"].pop(
                    cache_keys_by_symbol[symbol], None
                )
                research_by_symbol[symbol] = result
                validated_cached_research[symbol] = result
                stock_research_cache["entries"][
                    cache_keys_by_symbol[symbol]
                ] = {
                    "created_at": datetime.now(UTC).isoformat(),
                    "model": batch_model,
                    "market_context_hash": market_context_hash,
                    "stock_prompt_hash": stock_prompt_hash,
                    "candidate_hash": stable_json_hash(candidate),
                    "research": result,
                    "research_metadata": {
                        "search_queries": batch_metadata["search_queries"],
                        "tool_tokens": batch_metadata["tool_tokens"],
                    },
                }

            save_json_object_atomic(
                stock_research_cache_file, stock_research_cache
            )

            preview_symbols = preview_selectable_symbols(
                candidate_records,
                validated_cached_research,
                target_selected_stocks,
                max_stocks_per_sector,
                max_moderate_per_risk_event,
                excluded_crypto_dependence,
                exclude_high_catalyst_dependence,
            )
            if len(preview_symbols) >= target_selected_stocks:
                cache_already_fills_portfolio = True
                print(
                    "Validated research now supports the full portfolio; "
                    "skipping remaining targeted retries. Preview: "
                    + ", ".join(preview_symbols)
                )
                break

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
            for future_candidate in candidate_records:
                if len(next_pending) >= gemini_batch_size:
                    break
                future_symbol = str(future_candidate["Symbol"]).upper()
                if (
                        future_symbol in batched_symbols
                        or future_symbol in retry_symbols
                        or future_symbol in exhausted_symbols
                        or future_symbol in validated_cached_research
                        or future_symbol in research_failures_by_symbol
                ):
                    continue
                if sector_counts.get(future_candidate["Sector"], 0) >= (
                        max_stocks_per_sector
                ):
                    continue
                future_sector = future_candidate["Sector"]
                future_open_slots = (
                    max_stocks_per_sector
                    - sector_counts.get(future_sector, 0)
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
                if isinstance(deferred, dict):
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

    # Python alone owns portfolio selection and the authoritative ledger.
    for candidate in ranked_batch:
        if len(selected) >= target_selected_stocks:
            break

        symbol = str(candidate["Symbol"]).upper()
        sector = candidate["Sector"]
        sector_count = sector_counts.get(sector, 0)

        if sector_count >= max_stocks_per_sector:
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": "SKIPPED — SECTOR CAPACITY",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": (
                    f"{max_stocks_per_sector} stocks from {sector} were already "
                    "selected."
                ),
            })
            continue

        research = research_by_symbol.get(symbol)
        if research is None:
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": "SKIPPED — RESEARCH INVALID",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": (
                    "Research did not pass validation: "
                    + research_failures_by_symbol.get(
                        symbol, "no validated result was available"
                    )
                ),
            })
            continue

        reversal_risk = str(research["reversal_risk"]).upper()
        explanation = str(research["explanation"]).strip()

        prior_research_decisions.append({
            "symbol": symbol,
            "industry_group": research.get("industry_group"),
            "reversal_risk": reversal_risk,
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

        if not bool(research.get("eligible", True)):
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": "EXCLUDED — ELIGIBILITY",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": str(
                    research.get("eligibility_reason") or explanation
                ),
            })
            continue

        if excluded_by_crypto_policy(research, excluded_crypto_dependence):
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": "EXCLUDED — CRYPTO DEPENDENCE",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": (
                    f"{research.get('crypto_dependence')} crypto dependence is "
                    "outside the configured portfolio policy. " + explanation
                ),
            })
            continue

        if excluded_by_high_catalyst_policy(
                research, exclude_high_catalyst_dependence
        ):
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": "NOT SELECTED — HIGH CATALYST DEPENDENCE",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": (
                    "A HIGH-dependence temporary driver creates too much "
                    "single-catalyst exposure for selection. " + explanation
                ),
            })
            continue

        if reversal_risk in {"ELEVATED", "SEVERE"}:
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": f"NOT SELECTED — {reversal_risk}",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": explanation,
            })
            continue

        primary_event = research.get("primary_risk_event_id")
        if primary_event:
            primary_event = re.sub(
                r"[^A-Z0-9]+", "_", str(primary_event).upper()
            ).strip("_")
        event_key = normalized_risk_event_key(research)
        exposure_group = event_key[1] if event_key else None

        if (
                reversal_risk == "MODERATE"
                and event_key
                and moderate_event_counts.get(event_key, 0)
                >= max_moderate_per_risk_event
        ):
            decision_ledger.append({
                "qvm_rank": candidate["QVM Rank"],
                "symbol": symbol,
                "sector_group": sector,
                "status": "NOT SELECTED — RISK-EVENT CAPACITY",
                "sector_selected_after": sector_count,
                "total_selected_after": len(selected),
                "explanation": (
                    f"The {primary_event} / {exposure_group} risk group already "
                    f"contained "
                    f"{max_moderate_per_risk_event} selected moderate-risk stocks."
                ),
            })
            continue

        selected.append({"candidate": candidate, "research": research})
        sector_counts[sector] = sector_count + 1

        if reversal_risk == "MODERATE" and event_key:
            moderate_event_counts[event_key] = (
                    moderate_event_counts.get(event_key, 0) + 1
            )

        decision_ledger.append({
            "qvm_rank": candidate["QVM Rank"],
            "symbol": symbol,
            "sector_group": sector,
            "status": f"SELECTED — {reversal_risk}",
            "sector_selected_after": sector_counts[sector],
            "total_selected_after": len(selected),
            "explanation": explanation,
        })

print(
    f"Research efficiency: {len(researched_symbols_this_run)} unique stocks, "
    f"{stock_search_attempts_this_run} requested searches, "
    f"{stock_search_attempts_this_run - len(researched_symbols_this_run)} "
    "repeated searches."
)

context_review = build_context_review(decision_ledger)
print("\n" + context_review + "\n")

if len(selected) != target_selected_stocks:
    raise RuntimeError(
        f"Only {len(selected)} stocks satisfied all rules after evaluating "
        f"{len(candidate_records)} candidates; index.html was not replaced."
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
print(f"Generated research with model(s): {model_used}")
print("Selected symbols: " + ", ".join(
    item["candidate"]["Symbol"] for item in selected
))

# print elapsed time
end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")
