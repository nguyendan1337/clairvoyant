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

    if s in ["N/A", "NONE", ""]:
        return None

    match = re.search(r'-?\d+\.?\d*', s)
    if not match:
        return None

    num = float(match.group())

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
    Convert columns in `cols` to numeric values.
    Removes '%' signs and extracts first number from string if needed.
    """
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('%', '', regex=True)
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


def fetch_all_stock_pages_from_url(url, min_52_week_change=20):
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
        html = fetch_single_stock_page(url, start=start, count=count, cache=cache)

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

    # First try the specified model, if unavailable, fallback to "-lite" version
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


# ---------- MAIN FUNCTION ----------
def append_qvm_data_yfinance(
        df: pd.DataFrame,
        max_info_calls: int = 150,   # 🔥 KEY: limit expensive calls
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

    # ---- STEP 1: Compute momentum FIRST (cheap) ----
    momentum_scores = {}

    for symbol in tickers_list:
        try:
            df_prices = price_data if len(tickers_list) == 1 else price_data[symbol]
            close = df_prices["Close"].dropna()

            ret_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 10 else None
            ret_9m = ((close.iloc[-1] / close.iloc[-189]) - 1) * 100 if len(close) > 189 else None
            ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else None
            ret_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else None
            ret_1m = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 21 else None

            # simple pre-score to rank
            score = np.nanmean([ret_3m, ret_6m, ret_9m])
            momentum_scores[symbol] = score

            data_map[symbol] = {
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

    # ---- STEP 3: Fetch info (cached + throttled) ----
    for symbol in tqdm(selected_for_info):
        try:
            if symbol in cache:
                info = cache[symbol]["info"]
            else:
                ticker_obj = yf.Ticker(symbol)
                info = ticker_obj.info

                # store minimal fields
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
                "EV_EBITDA": (ev / ebitda) if ev and ebitda else None,
                "EV_Revenue": (ev / revenue) if ev and revenue else None,
            })

        except Exception as e:
            print(f"Error on {symbol}: {e}")

    # ---- Save cache ----
    save_cache(cache)

    # ---- Map back to df ----
    all_columns = [
        "Sector", "ROE", "ROA", "ProfitMargin", "GrossMargin",
        "DebtToEquity", "CurrentRatio", "InterestCoverage",
        "PE", "PriceToBook", "PEG", "EV_EBITDA", "EV_Revenue",
        "1M Return", "3M Return", "6M Return", "9M Return", "1Y Return"
    ]

    for col in all_columns:
        df[col] = df["Symbol"].map(lambda x: data_map.get(x, {}).get(col))

    print("Done.")
    return df



def score_qvm(df, top_n=100, weights=None):
    """
    Score and rank stocks using QVM with custom weights.
    Handles missing columns and percentage returns.

    weights: dict with keys 'Quality', 'Value', 'Momentum' summing to 1.
             Example: {'Quality': 0.4, 'Value': 0.2, 'Momentum': 0.4}
    """

    df = df.copy()

    # Default equal weights if none provided
    if weights is None:
        weights = {'Quality': 0.33, 'Value': 0.33, 'Momentum': 0.33}

    # --- QUALITY SCORE ---
    quality_metrics = ['ROE', 'ROA', 'ProfitMargin', 'GrossMargin', 'CurrentRatio', 'InterestCoverage']
    existing_quality = [c for c in quality_metrics if c in df.columns]
    if existing_quality:
        # Normalize each metric to 0–100 for balance
        df_quality = df[existing_quality]
        df_quality_norm = df_quality.apply(lambda col: (col - col.min()) / (col.max() - col.min()) * 100 if col.max() != col.min() else 50)
        df['QualityScore'] = df_quality_norm.mean(axis=1, skipna=True)
    else:
        df['QualityScore'] = np.nan

    # --- VALUE SCORE ---
    value_metrics = ['PE', 'PEG', 'PriceToBook', 'EV_EBITDA', 'EV_Revenue']
    existing_value = [c for c in value_metrics if c in df.columns]
    if existing_value:
        # Invert so lower = better
        df_value_inv = df[existing_value].apply(lambda col: 1/col.replace(0, np.nan))
        # Normalize to 0–100
        df_value_norm = df_value_inv.apply(lambda col: (col - col.min()) / (col.max() - col.min()) * 100 if col.max() != col.min() else 50)
        df['ValueScore'] = df_value_norm.mean(axis=1, skipna=True)
    else:
        df['ValueScore'] = np.nan

    # --- MOMENTUM SCORE ---
    momentum_weights = {
        '1M Return': 0.05,
        '3M Return': 0.20,
        '6M Return': 0.40,
        '9M Return': 0.25,
        '1Y Return': 0.10
    }

    existing_momentum = [c for c in momentum_weights if c in df.columns]

    if existing_momentum:
        df_momentum = df[existing_momentum].copy()

        # 1️⃣ Remove extreme losers (3M Return < -10%)
        extreme_negative_cap = -10
        if '3M Return' in df_momentum.columns:
            df = df[df['3M Return'] >= extreme_negative_cap]
            df_momentum = df_momentum.loc[df.index]  # sync with filtered df

        # 2️⃣ Cap temporarily down stocks (3M Return >= -10% but negative)
        momentum_negative_cap = -5
        if '3M Return' in df_momentum.columns:
            df_momentum['3M Return'] = df_momentum['3M Return'].clip(lower=momentum_negative_cap)

        # 3️⃣ Compute weighted momentum (base score)
        weighted_momentum = sum(
            df_momentum[col] * weight
            for col, weight in momentum_weights.items()
            if col in df_momentum.columns
        )

        # 4️⃣ Smoothness penalty (penalize spikes / inconsistency)
        smoothness = 0

        if all(col in df_momentum.columns for col in ['1M Return', '3M Return']):
            smoothness -= abs(df_momentum['1M Return'] - df_momentum['3M Return'])

        if all(col in df_momentum.columns for col in ['3M Return', '6M Return']):
            smoothness -= abs(df_momentum['3M Return'] - df_momentum['6M Return'])

        if all(col in df_momentum.columns for col in ['6M Return', '9M Return']):
            smoothness -= abs(df_momentum['6M Return'] - df_momentum['9M Return'])

        if all(col in df_momentum.columns for col in ['9M Return', '1Y Return']):
            smoothness -= abs(df_momentum['9M Return'] - df_momentum['1Y Return'])

        smoothness_weight = 0.1

        # 5️⃣ Trend direction penalty (penalize declining momentum)
        trend_penalty = 0

        if all(col in df_momentum.columns for col in ['1M Return', '3M Return']):
            trend_penalty += (df_momentum['1M Return'] - df_momentum['3M Return'])

        if all(col in df_momentum.columns for col in ['3M Return', '6M Return']):
            trend_penalty += (df_momentum['3M Return'] - df_momentum['6M Return'])

        if all(col in df_momentum.columns for col in ['6M Return', '9M Return']):
            trend_penalty += (df_momentum['6M Return'] - df_momentum['9M Return'])

        # Only penalize negative trends (declining momentum)
        trend_penalty = trend_penalty.clip(upper=0)

        trend_weight = 0.2

        # Combine everything
        combined_momentum = (
                weighted_momentum
                + (smoothness * smoothness_weight)
                + (trend_penalty * trend_weight)
        )

        # 6️⃣ Normalize to 0–100
        min_val = combined_momentum.min()
        max_val = combined_momentum.max()

        if max_val != min_val:
            df['MomentumScore'] = (combined_momentum - min_val) / (max_val - min_val) * 100
        else:
            df['MomentumScore'] = 50
    else:
        df['MomentumScore'] = np.nan

    # --- COMPOSITE QVM SCORE WITH WEIGHTS ---
    df['QVMScore'] = (
            df['QualityScore'] * weights.get('Quality', 0) +
            df['ValueScore'] * weights.get('Value', 0) +
            df['MomentumScore'] * weights.get('Momentum', 0)
    )

    # --- Sort and select top_n stocks ---
    df = df.sort_values('QVMScore', ascending=False)
    df_top = df.head(min(top_n, len(df)))

    # --- Keep only essential columns ---
    essential_columns = [
        'Symbol', 'Name', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %',
        'Avg Vol (3M)', 'Sector', 'ROE', 'ROA', 'ProfitMargin', 'GrossMargin',
        'DebtToEquity', 'CurrentRatio', 'InterestCoverage', 'PE', 'PriceToBook',
        'PEG', 'EV_EBITDA', 'EV_Revenue', '1M Return', '3M Return', '6M Return',
        '9M Return', '1Y Return', 'QualityScore', 'ValueScore', 'MomentumScore', 'QVMScore'
    ]

    df_top = df_top[[c for c in essential_columns if c in df_top.columns]]

    return df_top.reset_index(drop=True)



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

df = fetch_all_stock_pages_from_url(url, min_52_week_change)
df = df.drop_duplicates()

# Filter rules
# df = df[
#     (df['Market Cap'] > 300_000_000) &  # remove microcaps < $300M
#     (df['P/E Ratio(TTM)'].notnull()) & (df['P/E Ratio(TTM)'] > 0) & (df['P/E Ratio(TTM)'] < 200) &  # avoid negative or extreme PE
#     (df['Avg Vol (3M)'] > 100_000) &  # avoid illiquid stocks
#     (df['52 WkChange %'].notnull())  # require some price history
#     ].copy()
df = df[
    (df['Market Cap'] >= 300_000_000) &          # much lower threshold
    (df['Price'] >= 5.0) &
    (df['Avg Vol (3M)'] >= 100_000) &
    ((df['P/E Ratio(TTM)'].isna()) | (df['P/E Ratio(TTM)'] > 0))
].copy()

print("\nTrash Filtered Stocks:")
print(df[['Symbol', 'Name', '52 WkChange %']].reset_index(drop=True))

minimal_cols = ['Symbol', 'Name', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %', 'Avg Vol (3M)']
df_minimal = df[minimal_cols].copy()

df_yf = append_qvm_data_yfinance(df_minimal)
df_scored = score_qvm(df_yf, weights={'Quality': 0.38, 'Value': 0.25, 'Momentum': 0.37})

# Take top 50–100 stocks for watchlist
top_stocks = df_scored.head(100)
print("\nTop QVM Stocks:")
print(top_stocks.head(20)[['Symbol', '52 WkChange %', '1M Return', '3M Return', 'QVMScore']])

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

    # Valuation anchor
    "PE",

    # Momentum anchors
    "3M Return",
    "52 WkChange %"
]

# Filter df before sending to Gemini
df_gemini = top_stocks[essential_columns_for_gemini].copy()
df_gemini_str = df_gemini.to_string(index=False)
prompt = config["prompt"] + df_gemini_str

# # --- Pass the top etfs to Gemini to get world context and final recommendations ---
# client, gemini_config = initialize_gemini_client()
#
# final_recommendations, model_used = call_gemini(client, model_primary, model_fallback, gemini_config, prompt)
# print("\nGEMINI RESPONSE:\n")
# print(final_recommendations)
# print("Generated by model: " + model_used)
#
# # # --- Update HTML page with recommendations ---
# output_columns = [
#     'Symbol',
#     'Name',
#     'Sector',         # Add this if available
#     '52 WkChange %',
#     '3M Return',      # short-term momentum
#     'QVMScore'        # overall quantitative score
# ]
# df_html = df_gemini[output_columns].copy()
# df_html = df_html.sort_values(by='QVMScore', ascending=False).reset_index(drop=True)
# df_html["Symbol"] = df_html["Symbol"].apply(
#     lambda x: f'<a href="https://finance.yahoo.com/quote/{x}/" target="_blank">{x}</a>'
# )
# df_html["Name"] = df_html.apply(
#     lambda row: f'<a href="https://finance.yahoo.com/quote/{row["Symbol"].split(">")[1].split("<")[0]}/" target="_blank">{row["Name"]}</a>',
#     axis=1
# )
# df_html_table = df_html.to_html(escape=False, index=False, classes="recommendations-table", border=0)
# update_html_page(final_recommendations, df_html_table, "stock_page_template.html","index.html", model_used)

# print elapsed time
end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")