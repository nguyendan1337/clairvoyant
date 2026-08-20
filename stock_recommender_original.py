import re
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from google import genai
from bs4 import BeautifulSoup
from google.genai import types
from datetime import datetime, timedelta, UTC
import requests, time, random, json, yaml, os



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



HTML_CACHE_FILE = "stock_pages_cache.json"
HTML_CACHE_EXPIRY_DAYS = 1



def load_html_cache():
    if not os.path.exists(HTML_CACHE_FILE):
        return {}

    with open(HTML_CACHE_FILE, "r") as f:
        cache = json.load(f)

    fresh_cache = {}
    now = datetime.now(UTC)

    for key, entry in cache.items():
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            if now - ts < timedelta(days=HTML_CACHE_EXPIRY_DAYS):
                fresh_cache[key] = entry
        except:
            continue

    return fresh_cache



def save_html_cache(cache):
    with open(HTML_CACHE_FILE, "w") as f:
        json.dump(cache, f)



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
        load_dotenv()
        api_key = os.getenv("GEMINI_KEY")
    # Enable Google Search tool
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    gemini_config = types.GenerateContentConfig(tools=[grounding_tool])
    return genai.Client(api_key=api_key), gemini_config



def call_gemini(client, model_primary, model_fallback, gemini_config, prompt):
    def try_model(model_name):
        # Get response from Gemini
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    config=gemini_config,
                    contents=prompt
                )
                if not response.text:
                    raise ValueError("Empty response from Gemini.")
                return response.text, model_name
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries} on model {model_name}: {e}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (3 ** attempt)  # exponential backoff
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Failed after {max_retries} attempts on model '{model_name}': {e}")

    # First try the specified model, if unavailable, fallback to fallback model
    try:
        return try_model(model_primary)
    except Exception as e:
        print(f"Switching to fallback model due to error: {e}")
        try:
            return try_model(model_fallback)
        except Exception as fallback_error:
            raise Exception(f"Failed after {max_retries} attempts on model '{model_fallback}': {e}")



CACHE_FILE = "yf_cache.json"
CACHE_EXPIRY_DAYS = 1



# ---------- CACHE HELPERS ----------
def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}

    with open(CACHE_FILE, "r") as f:
        cache = json.load(f)

    fresh_cache = {}
    now = datetime.now(UTC)

    for ticker, entry in cache.items():
        try:
            ts = datetime.fromisoformat(entry["timestamp"])
            if now - ts < timedelta(days=CACHE_EXPIRY_DAYS):
                fresh_cache[ticker] = entry
        except:
            continue

    return fresh_cache



def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)



# ---------- TOP QVM DATAFRAME CACHE ----------
# This cache stores the fully computed top-50 QVM DataFrame so that
# Gemini prompt testing can skip the expensive stock-data/QVM pipeline.
# The cache file can be persisted by GitHub Actions in the same way as
# the existing JSON caches.
TOP_QVM_CACHE_FILE = "top_qvm_stocks_cache.pkl"
TOP_QVM_CACHE_EXPIRY_HOURS = 6
TOP_QVM_CACHE_VERSION = 1


def load_top_qvm_cache():
    """Load the cached top-QVM DataFrame if it exists and is still fresh."""
    if not os.path.exists(TOP_QVM_CACHE_FILE):
        return None

    try:
        modified_time = datetime.fromtimestamp(
            os.path.getmtime(TOP_QVM_CACHE_FILE),
            tz=UTC
        )

        age = datetime.now(UTC) - modified_time
        if age >= timedelta(hours=TOP_QVM_CACHE_EXPIRY_HOURS):
            print(
                f"Top QVM cache is stale ({age.total_seconds() / 3600:.1f}h old)."
            )
            return None

        cached = pd.read_pickle(TOP_QVM_CACHE_FILE)

        if not isinstance(cached, dict):
            print("Invalid top QVM cache format. Rebuilding cache.")
            return None

        if cached.get("version") != TOP_QVM_CACHE_VERSION:
            print("Top QVM cache version mismatch. Rebuilding cache.")
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
            "data": df.copy()
        }
        pd.to_pickle(cache, TOP_QVM_CACHE_FILE)
        print(f"Saved top QVM cache to {TOP_QVM_CACHE_FILE}.")
    except Exception as e:
        print(f"Could not save top QVM cache: {e}")


# ---------- MAIN FUNCTION ----------
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
            'Quality': 0.45,
            'Value': 0.15,
            'Momentum': 0.40
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



def update_html_page(final_recommendations, df_html_table, template_name, display_page, model_used):
    # --- Extract table and summary blocks in any order ---
    table_match = re.search(r'(<table.*?</table>)', final_recommendations, flags=re.DOTALL | re.IGNORECASE)
    summary_match = re.search(r'(<div[^>]*class=["\']summary["\'][^>]*>.*?</div>)', final_recommendations, flags=re.DOTALL | re.IGNORECASE)

    gemini_table_html = table_match.group(1).strip() if table_match else ""
    gemini_summary = summary_match.group(1).strip() if summary_match else ""

    # --- Fallbacks ---
    if not gemini_table_html and "<table" in final_recommendations:
        # Try to recover a partial table if regex failed
        start = final_recommendations.find("<table")
        end = final_recommendations.find("</table>") + 8
        gemini_table_html = final_recommendations[start:end]
    if not gemini_summary and "<div" in final_recommendations:
        # Try to recover a generic div summary if regex failed
        start = final_recommendations.find("<div")
        end = final_recommendations.find("</div>") + 6
        gemini_summary = final_recommendations[start:end]

    # --- Read HTML template ---
    with open(template_name, "r", encoding="utf-8") as f:
        template = f.read()

    # --- Insert content ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_output = template.replace("<!--LAST_UPDATED_HERE-->", timestamp)
    html_output = html_output.replace("<!--RECOMMENDATIONS_TABLE_HERE-->", gemini_table_html)
    html_output = html_output.replace("<!--RECOMMENDATIONS_SUMMARY_HERE-->", gemini_summary)
    html_output = html_output.replace("<!--FULL_DF_TABLE_HERE-->", df_html_table)
    html_output = html_output.replace("<!--MODEL_USED_HERE-->", model_used)

    # --- Write final index.html ---
    with open(display_page, "w", encoding="utf-8") as f:
        f.write(html_output)



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
    df_scored = score_qvm(df_yf)

    # Take top 50 stocks for watchlist/Gemini evaluation.
    top_stocks = df_scored.head(50).copy()

    save_top_qvm_cache(top_stocks)

print("\nTop QVM Stocks:")
print(top_stocks.head(50)[['Symbol', 'QVMScore', '3M Return','1Y Return']])

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
with open('top_qvm_stocks.md', 'w') as f:
    f.write(
        top_stocks[cols_for_eval]
        .to_markdown(index=False)
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

# Filter df before sending to Gemini
df_gemini = top_stocks[essential_columns_for_gemini].copy()
df_gemini_str = df_gemini.to_string(index=False)
prompt = config["prompt"] + df_gemini_str

# --- Pass the top etfs to Gemini to get world context and final recommendations ---
print("\n...calling Gemini...\n")
client, gemini_config = initialize_gemini_client()

final_recommendations, model_used = call_gemini(client, model_primary, model_fallback, gemini_config, prompt)
print("\nGEMINI RESPONSE:\n")
print(final_recommendations)
print("Generated by model: " + model_used)

# # --- Update HTML page with recommendations ---
output_columns = [
    'Symbol',
    'Name',
    'Sector',
    '52 WkChange %',
    '3M Return',      # short-term momentum
    'QVMScore'        # overall quantitative score
]
df_html = df_gemini[output_columns].copy()
df_html = df_html.sort_values(by='QVMScore', ascending=False).reset_index(drop=True)
df_html["Symbol"] = df_html["Symbol"].apply(
    lambda x: f'<a href="https://finance.yahoo.com/quote/{x}/" target="_blank">{x}</a>'
)
df_html["Name"] = df_html.apply(
    lambda row: f'<a href="https://finance.yahoo.com/quote/{row["Symbol"].split(">")[1].split("<")[0]}/" target="_blank">{row["Name"]}</a>',
    axis=1
)
df_html_table = df_html.to_html(escape=False, index=False, classes="recommendations-table", border=0)
update_html_page(final_recommendations, df_html_table, "stock_page_template.html","index.html", model_used)

# print elapsed time
end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")
