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
from collections import Counter
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
    def __init__(self, maximum):
        self.maximum = int(maximum)
        self.used = 0

    def consume(self, stage):
        if self.used >= self.maximum:
            raise RuntimeError(
                f"Gemini request budget of {self.maximum} was exhausted "
                f"before {stage}."
            )
        self.used += 1
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
    # One source is accepted only when metadata proves that both required
    # company-specific searches ran for this exact candidate.
    if candidate_search_count(search_queries, candidate) >= 2:
        return 1
    return 2


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

    if len(unique_urls) < minimum:
        raise ValueError(
            f"{label} requires at least {minimum} distinct source URLs."
        )
    if maximum is not None and len(unique_urls) > maximum:
        raise ValueError(
            f"{label} returned {len(unique_urls)} distinct source URLs; "
            f"the maximum is {maximum}."
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

ALLOWED_CATALYST_DEPENDENCE = {"LOW", "MODERATE", "HIGH"}
ALLOWED_MECHANISM_STATUSES = {
    "NONE", "HYPOTHETICAL", "ACTIVE", "UNUSUALLY_PROBABLE"
}
ALLOWED_NORMALIZATION_PROBABILITIES = {
    "NOT_APPLICABLE", "NOT_ESTABLISHED", "REASONABLY_PROBABLE",
    "AT_LEAST_AS_LIKELY",
}
ALLOWED_CONTINUATION_OUTLOOKS = {
    "CONTINUATION_MORE_LIKELY", "REVERSAL_AT_LEAST_AS_LIKELY",
    "THESIS_BROKEN",
}
INCOMPLETE_RESEARCH_PHRASES = {
    "no specific operating drivers",
    "no operating drivers",
    "from available search results",
    "based on available information",
    "insufficient information",
    "unable to identify",
    "could not identify",
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

        if catalyst_dependence not in ALLOWED_CATALYST_DEPENDENCE:
            raise ValueError(
                f"{symbol} has invalid catalyst_dependence: "
                f"{result.get('catalyst_dependence')!r}"
            )

        result["catalyst_dependence"] = catalyst_dependence

        mechanism_status = str(
            result.get("mechanism_status", "")
        ).strip().upper()
        if mechanism_status not in ALLOWED_MECHANISM_STATUSES:
            raise ValueError(
                f"{symbol} has invalid mechanism_status: "
                f"{result.get('mechanism_status')!r}"
            )
        result["mechanism_status"] = mechanism_status

        normalization_probability = str(
            result.get("normalization_probability", "")
        ).strip().upper()
        if normalization_probability not in ALLOWED_NORMALIZATION_PROBABILITIES:
            raise ValueError(
                f"{symbol} has invalid normalization_probability: "
                f"{result.get('normalization_probability')!r}"
            )
        result["normalization_probability"] = normalization_probability

        continuation_outlook = str(
            result.get("continuation_outlook", "")
        ).strip().upper()
        if continuation_outlook not in ALLOWED_CONTINUATION_OUTLOOKS:
            raise ValueError(
                f"{symbol} has invalid continuation_outlook: "
                f"{result.get('continuation_outlook')!r}"
            )
        result["continuation_outlook"] = continuation_outlook

        durable_drivers = result.get("durable_drivers")
        if not isinstance(durable_drivers, list) or not 1 <= len(durable_drivers) <= 2:
            raise ValueError(f"{symbol} must have 1-2 durable_drivers.")
        durable_drivers = [str(item).strip() for item in durable_drivers]
        if any(not item for item in durable_drivers):
            raise ValueError(f"{symbol} has an empty durable_driver.")
        if any(
            phrase in " ".join(durable_drivers).lower()
            for phrase in INCOMPLETE_RESEARCH_PHRASES
        ):
            raise ValueError(f"{symbol} has incomplete durable-driver research.")
        result["durable_drivers"] = durable_drivers

        temporary_drivers = result.get("temporary_drivers")
        if not isinstance(temporary_drivers, list) or len(temporary_drivers) > 2:
            raise ValueError(f"{symbol} must have 0-2 temporary_drivers.")
        temporary_drivers = [str(item).strip() for item in temporary_drivers]
        if any(not item for item in temporary_drivers):
            raise ValueError(f"{symbol} has an empty temporary_driver.")
        result["temporary_drivers"] = temporary_drivers
        if temporary_drivers and normalization_probability == "NOT_APPLICABLE":
            raise ValueError(
                f"{symbol} has temporary drivers but normalization is "
                "NOT_APPLICABLE."
            )
        if catalyst_dependence in {"MODERATE", "HIGH"} and not temporary_drivers:
            raise ValueError(
                f"{symbol} has {catalyst_dependence} catalyst dependence "
                "without a temporary driver."
            )
        if mechanism_status == "NONE" and temporary_drivers:
            raise ValueError(
                f"{symbol} mechanism_status NONE conflicts with temporary drivers."
            )
        if (
            mechanism_status == "NONE"
            and normalization_probability != "NOT_APPLICABLE"
        ):
            raise ValueError(
                f"{symbol} mechanism_status NONE requires "
                "normalization_probability NOT_APPLICABLE."
            )

        explanation = re.sub(
            r"\s*\[(?:cite|source|citation):[^\]]+\]",
            "",
            str(result.get("explanation", "")),
            flags=re.IGNORECASE,
        ).strip()

        if not explanation:
            raise ValueError(f"{symbol} has no explanation.")

        explanation_lower = explanation.lower()
        incomplete_phrase = next(
            (
                phrase for phrase in INCOMPLETE_RESEARCH_PHRASES
                if phrase in explanation_lower
            ),
            None,
        )
        if incomplete_phrase:
            raise ValueError(
                f"{symbol} explanation indicates incomplete research: "
                f"{incomplete_phrase!r}."
            )

        result["explanation"] = explanation
        if not str(result.get("reversal_mechanism", "")).strip():
            raise ValueError(f"{symbol} has no reversal mechanism.")
        if not str(result.get("normalization_effect", "")).strip():
            raise ValueError(f"{symbol} has no normalization analysis.")

        risk_materiality = str(
            result.get("risk_materiality", "")
        ).strip().upper()

        risk_materiality = {
            "MEDIUM": "MODERATE",
        }.get(risk_materiality, risk_materiality)

        if risk_materiality not in {"LOW", "MODERATE", "HIGH"}:
            raise ValueError(
                f"{symbol} has invalid risk_materiality: "
                f"{result.get('risk_materiality')!r}"
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

        reversal_evidence = result.get("reversal_evidence")
        if reversal_evidence is not None:
            reversal_evidence = str(reversal_evidence).strip() or None
        result["reversal_evidence"] = reversal_evidence

        for field in (
            "current_fact", "probability_evidence", "material_effect",
            "company_difference",
        ):
            value = result.get(field)
            if value is not None:
                value = str(value).strip() or None
            result[field] = value

        evidence_required_risks = {
            "MODERATE",
            "ELEVATED",
            "SEVERE",
        }

        if reversal_risk in evidence_required_risks:
            missing_evidence = [
                field for field in (
                    "current_fact", "probability_evidence", "material_effect"
                )
                if not result[field]
            ]
            if not reversal_evidence:
                missing_evidence.append("reversal_evidence")
            if missing_evidence:
                raise ValueError(
                    f"{symbol} assessed as {reversal_risk} without complete "
                    "evidence: " + ", ".join(missing_evidence)
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

        if reversal_risk == "MINIMAL":
            if (
                mechanism_status != "NONE"
                or catalyst_dependence != "LOW"
                or continuation_outlook != "CONTINUATION_MORE_LIKELY"
            ):
                raise ValueError(
                    f"{symbol} MINIMAL conflicts with its mechanism, catalyst, "
                    "or continuation fields."
                )
        elif reversal_risk == "LOW":
            if mechanism_status not in {"NONE", "HYPOTHETICAL"}:
                raise ValueError(
                    f"{symbol} LOW requires NONE or HYPOTHETICAL mechanism_status."
                )
            if continuation_outlook != "CONTINUATION_MORE_LIKELY":
                raise ValueError(
                    f"{symbol} LOW requires CONTINUATION_MORE_LIKELY."
                )
        elif reversal_risk == "MODERATE":
            if mechanism_status not in {"ACTIVE", "UNUSUALLY_PROBABLE"}:
                raise ValueError(
                    f"{symbol} MODERATE requires ACTIVE or "
                    "UNUSUALLY_PROBABLE mechanism_status."
                )
            if continuation_outlook != "CONTINUATION_MORE_LIKELY":
                raise ValueError(
                    f"{symbol} MODERATE requires CONTINUATION_MORE_LIKELY."
                )
            if normalization_probability not in {
                "NOT_APPLICABLE", "REASONABLY_PROBABLE"
            }:
                raise ValueError(
                    f"{symbol} MODERATE has inconsistent "
                    f"normalization_probability={normalization_probability}."
                )
        elif reversal_risk == "ELEVATED":
            if mechanism_status not in {"ACTIVE", "UNUSUALLY_PROBABLE"}:
                raise ValueError(
                    f"{symbol} ELEVATED requires ACTIVE or "
                    "UNUSUALLY_PROBABLE mechanism_status."
                )
            if continuation_outlook != "REVERSAL_AT_LEAST_AS_LIKELY":
                raise ValueError(
                    f"{symbol} ELEVATED requires REVERSAL_AT_LEAST_AS_LIKELY."
                )
        elif reversal_risk == "SEVERE":
            if mechanism_status not in {"ACTIVE", "UNUSUALLY_PROBABLE"}:
                raise ValueError(
                    f"{symbol} SEVERE requires ACTIVE or "
                    "UNUSUALLY_PROBABLE mechanism_status."
                )
            if continuation_outlook != "THESIS_BROKEN":
                raise ValueError(f"{symbol} SEVERE requires THESIS_BROKEN.")

        if not isinstance(result.get("sources"), list) or len(result["sources"]) != 2:
            raise ValueError(f"{symbol} must return exactly two source objects.")
        validate_sources(
            result.get("sources"),
            symbol,
            minimum=max(2, minimum_sources),
            maximum=2,
        )

        unique_source_urls = {
            str(source["url"]).strip()
            for source in result["sources"]
        }
        print(
            f"Validated JSON sources [{symbol}]: "
            f"{len(unique_source_urls)} distinct URLs"
        )


def shared_event_consistency_errors(results, prior_results=None):
    """Return current symbols whose shared-event classifications need retry."""
    current_symbols = {
        str(result.get("symbol", "")).strip().upper() for result in results
    }
    combined = list(prior_results or []) + list(results)
    groups = {}
    for result in combined:
        event_id = result.get("primary_risk_event_id")
        if not event_id:
            continue
        event_id = re.sub(
            r"[^A-Z0-9]+", "_", str(event_id).upper()
        ).strip("_")
        if event_id:
            groups.setdefault(event_id, []).append(result)

    errors = {}
    fields = (
        "mechanism_status", "normalization_probability",
        "continuation_outlook", "reversal_risk",
    )
    for event_id, members in groups.items():
        if len(members) < 2:
            continue
        signatures = [
            tuple(str(member.get(field, "")).strip().upper() for field in fields)
            for member in members
        ]
        baseline = Counter(signatures).most_common(1)[0][0]
        unsupported = [
            member for member, signature in zip(members, signatures)
            if signature != baseline
            and not str(member.get("company_difference") or "").strip()
        ]
        if not unsupported:
            continue
        group_symbols = [
            str(member.get("symbol", "")).strip().upper() for member in members
        ]
        message = (
            f"shared event {event_id} has inconsistent classification fields "
            "without a sourced company_difference; reassess the group baseline "
            f"for {', '.join(symbol for symbol in group_symbols if symbol)}"
        )
        for symbol in group_symbols:
            if symbol in current_symbols:
                errors[symbol] = message
    return errors


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
        max_attempts=None):
    """Call Gemini, require grounded research, parse JSON, and validate it."""
    models = [model_primary]
    if model_fallback and model_fallback != model_primary:
        models.append(model_fallback)

    attempt_limit = (
        max_retries if max_attempts is None else max(1, int(max_attempts))
    )
    last_error = None
    for model_name in models:
        for attempt in range(attempt_limit):
            attempt_prompt = prompt
            if attempt > 0:
                attempt_prompt += f"""
            
            MANDATORY RETRY CORRECTION
            The previous attempt failed for this exact reason:
            {last_error}
            
            Correct that failure. If company-specific search coverage was incomplete,
            execute both required searches for every supplied candidate before writing any
            JSON. Return every supplied candidate exactly once with every required field.
            Output only the single valid JSON object requested by the schema.
            """
            try:
                request_budget.consume(
                    f"{stage} ({model_name}, attempt {attempt + 1})"
                )
                response = client.models.generate_content(
                    model=model_name,
                    config=gemini_config,
                    contents=attempt_prompt
                )
                if not response.text:
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
                        raise ValueError(
                            "Gemini did not expose company-specific searches for: "
                            + ", ".join(missing_symbols)
                        )

                try:
                    data = parse_json_response(response.text)
                except json.JSONDecodeError as parse_error:
                    if not allow_partial_stock_results:
                        raise

                    partial_results = extract_partial_stock_results(
                        response.text
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

                resource_exhausted = (
                        error_code == 429
                        or "RESOURCE_EXHAUSTED" in error_text
                        or "429 TOO MANY REQUESTS" in error_text
                )

                print(
                    f"Error on attempt {attempt + 1}/{attempt_limit} "
                    f"during {stage} with {model_name}: {exc}"
                )

                if resource_exhausted:
                    print(
                        f"{model_name} returned 429 RESOURCE_EXHAUSTED; "
                        "skipping further retries for this model."
                    )
                    break

                if attempt < attempt_limit - 1:
                    delay = initial_delay * (3 ** attempt)
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)

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
        sentences = [escape(sector_context_sentence(market_context, sector))]
        for item in items:
            candidate = item["candidate"]
            research = item["research"]
            symbol_link = yahoo_link(candidate["Symbol"], candidate["Symbol"])
            name_link = yahoo_link(candidate["Symbol"], candidate["Name"])
            sentences.append(
                f"{symbol_link} ({name_link}) has "
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
model_primary = config["model_primary"]
model_fallback = config["model_fallback"]
gemini_batch_size = int(config.get("gemini_batch_size", 5))
max_candidates = int(config.get("max_candidates", 50))
target_selected_stocks = int(config.get("target_selected_stocks", 10))
max_stocks_per_sector = int(config.get("max_stocks_per_sector", 2))
max_moderate_per_risk_event = int(
    config.get("max_moderate_per_risk_event", 2)
)
thinking_budget = int(config.get("thinking_budget", 12288))
require_google_search = bool(config.get("require_google_search", True))
max_gemini_calls_per_run = int(config.get("max_gemini_calls_per_run", 12))
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
        (df['Avg Vol (3M)'] >= 100_000) &
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

client = initialize_gemini_client()
gemini_config = build_gemini_config(thinking_budget)
request_budget = GeminiRequestBudget(max_gemini_calls_per_run)

market_prompt = (
    config["prompt_market_context"]
    + f"\n\nCURRENT_DATE_UTC: {datetime.now(UTC).date().isoformat()}\n"
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
        validate_market_context(market_cache["market_context"])
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
        validator=validate_market_context,
        request_budget=request_budget,
        require_google_search=require_google_search,
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
stock_prompt_hash = stable_json_hash({
    "cache_version": cache_version,
    "model": model_primary,
    "prompt": config["prompt_stock_batch"],
})
stock_research_cache = load_json_object(stock_research_cache_file)
if stock_research_cache.get("version") != cache_version:
    stock_research_cache = {"version": cache_version, "entries": {}}
stock_research_cache.setdefault("entries", {})
stock_research_cache["entries"] = {
    key: entry
    for key, entry in stock_research_cache["entries"].items()
    if isinstance(entry, dict)
    and cache_entry_is_fresh(entry, gemini_research_cache_hours)
}
save_json_object_atomic(stock_research_cache_file, stock_research_cache)

selected = []
decision_ledger = []
sector_counts = {}
moderate_event_counts = {}
prior_research_decisions = []
models_used = [market_model]

for batch_start in range(0, len(candidate_records), gemini_batch_size):
    if len(selected) >= target_selected_stocks:
        break

    ranked_batch = candidate_records[
        batch_start:batch_start + gemini_batch_size
    ]

    # Stocks from sectors already full before this batch do not need research.
    research_candidates = [
        candidate
        for candidate in ranked_batch
        if sector_counts.get(candidate["Sector"], 0) < max_stocks_per_sector
    ]

    research_by_symbol = {}
    uncached_candidates = []
    cache_keys_by_symbol = {}

    for candidate in research_candidates:
        symbol = str(candidate["Symbol"]).upper()
        cache_key = stable_json_hash({
            "market_context_hash": market_context_hash,
            "stock_prompt_hash": stock_prompt_hash,
            "model": model_primary,
            "candidate": candidate,
        })
        cache_keys_by_symbol[symbol] = cache_key
        entry = stock_research_cache["entries"].get(cache_key)
        if entry and cache_entry_is_fresh(
                entry, gemini_research_cache_hours
        ):
            try:
                cached_result = entry["research"]
                cached_search_queries = entry.get(
                    "research_metadata", {}
                ).get("search_queries", [])
                minimum_sources = minimum_sources_for_candidate(
                    cached_search_queries, candidate
                )
                validate_stock_batch(
                    {"results": [cached_result]},
                    [candidate],
                    minimum_sources=minimum_sources,
                )
                research_by_symbol[symbol] = cached_result
                models_used.append(entry.get("model") or model_primary)
                print(f"Using validated stock research cache for {symbol}.")
                continue
            except (KeyError, TypeError, ValueError) as exc:
                print(f"Ignoring invalid stock cache entry for {symbol}: {exc}")
        uncached_candidates.append(candidate)

    if uncached_candidates:
        pending_candidates = list(uncached_candidates)
        validation_errors = {}

        for research_round in range(1, max_retries + 1):
            print(
                "\n...calling Gemini for stock batch: "
                + ", ".join(row["Symbol"] for row in pending_candidates)
                + "...\n"
            )

            retry_correction = ""
            if research_round > 1:
                retry_details = "\n".join(
                    f"  - {symbol}: {message}"
                    for symbol, message in validation_errors.items()
                )
                retry_correction = f"""

TARGETED RETRY CORRECTION
Valid research for other candidates has already been saved. Research and return
only the candidates supplied below. Correct these exact validation errors:
{retry_details}
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
                model_fallback=model_fallback,
                gemini_config=gemini_config,
                prompt=stock_prompt,
                stage=stage,
                validator=lambda data, expected=pending_candidates: (
                    validate_stock_batch_structure(data, expected)
                ),
                request_budget=request_budget,
                require_google_search=require_google_search,
                required_search_candidates=pending_candidates,
                allow_partial_stock_results=True,
                max_attempts=1,
            )
            models_used.append(batch_model)
            results_by_symbol = {
                str(result.get("symbol", "")).upper(): result
                for result in batch_data["results"]
            }
            shared_errors = shared_event_consistency_errors(
                batch_data["results"], prior_research_decisions
            )
            next_pending = []
            next_errors = {}

            for candidate in pending_candidates:
                symbol = str(candidate["Symbol"]).upper()
                result = results_by_symbol.get(symbol)
                if result is None:
                    next_pending.append(candidate)
                    next_errors[symbol] = "result was missing from the response"
                    continue

                if symbol in shared_errors:
                    next_pending.append(candidate)
                    next_errors[symbol] = shared_errors[symbol]
                    print(
                        f"Saving unrelated valid results; {symbol} requires a "
                        f"shared-event retry: {shared_errors[symbol]}"
                    )
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
                except (KeyError, TypeError, ValueError) as exc:
                    next_pending.append(candidate)
                    next_errors[symbol] = str(exc)
                    print(
                        f"Saving other valid results; {symbol} requires a "
                        f"targeted retry: {exc}"
                    )
                    continue

                research_by_symbol[symbol] = result
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
            if not next_pending:
                break

            pending_candidates = next_pending
            validation_errors = next_errors
        else:
            error_summary = "; ".join(
                f"{symbol}: {message}"
                for symbol, message in validation_errors.items()
            )
            raise RuntimeError(
                "Stock research remained invalid after targeted retries: "
                + error_summary
            )

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
            raise RuntimeError(
                f"No validated research was returned for eligible candidate {symbol}."
            )

        reversal_risk = str(research["reversal_risk"]).upper()
        explanation = str(research["explanation"]).strip()

        prior_research_decisions.append({
            "symbol": symbol,
            "reversal_risk": reversal_risk,
            "catalyst_dependence": research.get("catalyst_dependence"),
            "mechanism_status": research.get("mechanism_status"),
            "normalization_probability": research.get(
                "normalization_probability"
            ),
            "continuation_outlook": research.get("continuation_outlook"),
            "company_difference": research.get("company_difference"),
            "primary_risk_event_id": research.get("primary_risk_event_id"),
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

        if (
                reversal_risk == "MODERATE"
                and primary_event
                and moderate_event_counts.get(primary_event, 0)
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
                    f"The {primary_event} risk group already contained "
                    f"{max_moderate_per_risk_event} selected moderate-risk stocks."
                ),
            })
            continue

        selected.append({"candidate": candidate, "research": research})
        sector_counts[sector] = sector_count + 1

        if reversal_risk == "MODERATE" and primary_event:
            moderate_event_counts[primary_event] = (
                    moderate_event_counts.get(primary_event, 0) + 1
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
            "Durable Drivers": item["research"].get("durable_drivers", []),
            "Reversal Mechanism": item["research"].get(
                "reversal_mechanism"
            ),
            "Normalization Effect": item["research"].get(
                "normalization_effect"
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
            model_fallback=None,
            gemini_config=build_gemini_config(
                summary_thinking_budget, enable_search=False
            ),
            prompt=summary_prompt,
            stage="final HTML summary",
            validator=validate_summary_presence,
            request_budget=request_budget,
            require_google_search=False,
            max_attempts=1,
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
