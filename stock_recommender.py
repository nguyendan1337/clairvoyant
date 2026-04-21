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
            # Use your suffix extractor for Market Cap, Volume, etc.
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



def score_qvm(df, top_n=100, weights=None, min_quality=0):
    df = df.copy()

    if weights is None:
        weights = {
            'Quality': 0.38,
            'Value': 0.17,
            'Momentum': 0.45
        }

    # ===========================
    # 1. QUALITY SCORE (less harsh debt penalty)
    # ===========================
    quality_metrics = [
        'ROE', 'ROA', 'ProfitMargin',
        'GrossMargin', 'CurrentRatio',
        'InterestCoverage'
    ]
    q_cols = [c for c in quality_metrics if c in df.columns]

    if q_cols:
        q = df[q_cols].apply(lambda x: x.rank(pct=True))
        quality = q.mean(axis=1)

        # Softer, capped debt penalty
        if 'DebtToEquity' in df.columns:
            debt = df['DebtToEquity'].clip(upper=100)  # cap extreme outliers
            debt_penalty = debt.rank(pct=True)
            quality = quality - 0.22 * debt_penalty

        df['QualityScore'] = (quality * 100).clip(0, 100)
    else:
        df['QualityScore'] = 50

    if min_quality > 0:
        df = df[df['QualityScore'] >= min_quality].copy()

    # ===========================
    # 2. VALUE SCORE (NaN-safe, unchanged structure)
    # ===========================
    value_metrics = ['PE', 'PEG', 'PriceToBook', 'EV_EBITDA', 'EV_Revenue']
    v_cols = [c for c in value_metrics if c in df.columns]

    if v_cols:
        v = df[v_cols].apply(lambda x: x.rank(pct=True))
        df['ValueScore'] = v.mean(axis=1) * 100
    else:
        df['ValueScore'] = 50

    # ===========================
    # 3. MOMENTUM SCORE (reward real winners more)
    # ===========================
    mom_cols = ['1M Return','3M Return','6M Return','9M Return','1Y Return']
    m_cols = [c for c in mom_cols if c in df.columns]

    if m_cols:
        m = df[m_cols].copy().clip(lower=-100, upper=500)

        # Strong emphasis on actual returns
        raw_momentum = m.mean(axis=1).rank(pct=True) * 100

        # Trend persistence (lighter penalty)
        consistency = 0
        if '3M Return' in m and '6M Return' in m:
            consistency += (m['3M Return'] - m['6M Return']).abs()
        if '6M Return' in m and '1Y Return' in m:
            consistency += (m['6M Return'] - m['1Y Return']).abs()

        consistency = (100 - consistency.rank(pct=True) * 100)

        # Volatility penalty (reduced impact)
        volatility = m.std(axis=1).replace(0, np.nan)
        vol_penalty = volatility.rank(pct=True) * 100

        df['MomentumScore'] = (
                0.65 * raw_momentum +     # dominant factor
                0.20 * consistency -
                0.15 * vol_penalty       # reduced penalty
        )

        df['MomentumScore'] = df['MomentumScore'].clip(0, 100)
    else:
        df['MomentumScore'] = 50

    # ===========================
    # 4. FINAL SCORE
    # ===========================
    df['QVMScore'] = (
            df['QualityScore'] * weights['Quality'] +
            df['ValueScore'] * weights['Value'] +
            df['MomentumScore'] * weights['Momentum']
    )

    return df.sort_values('QVMScore', ascending=False).head(top_n).reset_index(drop=True)

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
df = df[
    (df['Market Cap'] >= 300_000_000) &
    (df['Price'] >= 5.0) &
    (df['Avg Vol (3M)'] >= 100_000) &
    ((df['P/E Ratio(TTM)'].isna()) | (df['P/E Ratio(TTM)'] > 0))
].copy()

print("\nTrash Filtered Stocks:")
print(df[['Symbol', 'Name', '52 WkChange %']].reset_index(drop=True))

minimal_cols = ['Symbol', 'Name', 'Market Cap', 'Price', 'P/E Ratio(TTM)', '52 WkChange %', 'Avg Vol (3M)']
df_minimal = df[minimal_cols].copy()

df_yf = append_qvm_data_yfinance(df_minimal)
# After your append_qvm_data_yfinance step
df_scored = score_qvm(
    df_yf,
    weights={'Quality': 0.38, 'Value': 0.17, 'Momentum': 0.45}
)

# Take top 50–100 stocks for watchlist
top_stocks = df_scored.head(50)
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

# essential_columns_for_gemini = [
#     # Identity
#     "Symbol", "Name", "Sector",
#
#     # Size / context
#     "Market Cap",
#
#     # Core QVM output (most important)
#     "QVMScore",
#
#     # Decomposed signals (compressed)
#     "QualityScore",
#     "ValueScore",
#     "MomentumScore",
#
#     # Key fundamentals
#     "ROE",
#     "ProfitMargin",
#     "DebtToEquity",
#     "EV_EBITDA",
#     "PEG",
#
#     # Valuation anchor
#     "PE",
#
#     # Momentum anchors
#     "3M Return",
#     "6M Return",
#     "9M Return",
#     "52 WkChange %"
# ]
#
# # Filter df before sending to Gemini
# df_gemini = top_stocks[essential_columns_for_gemini].copy()
# df_gemini_str = df_gemini.to_string(index=False)
# prompt = config["prompt"] + df_gemini_str
#
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