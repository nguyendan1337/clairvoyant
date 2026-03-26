import os
import re
import time
import yaml
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
from bs4 import BeautifulSoup
from datetime import datetime
from google.genai import types



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



def fetch_single_stock_page(url, start=0, count=100, retries=3, sleep=2):
    paged_url = f"{url}?start={start}&count={count}"
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
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt} failed for {paged_url}: {e}")
            if attempt < retries:
                time.sleep(sleep)
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
    target_col = '52 WkChange %'
    numeric_cols = [
        'Price', 'Change', 'Change %', 'Volume',
        'Avg Vol (3M)', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %'
    ]

    while True:
        html = fetch_single_stock_page(url, start=start, count=count)
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

        # Clean numeric columns (you need to define/implement this function)
        df_page = clean_numeric_columns(df_page, numeric_cols)

        # Drop clearly invalid rows early
        df_page = df_page[
            df_page[target_col].notna() &
            (df_page['Avg Vol (3M)'] > 0) &
            (df_page['Market Cap'] > 0)
            ]

        if df_page.empty:
            print(f"No valid rows after cleaning at start={start}. Stopping.")
            break

        # ─── Early stopping logic ────────────────────────────────
        max_change_on_page = df_page[target_col].max()
        if max_change_on_page < min_52_week_change:
            print(f"Page at start={start} has max {target_col} = {max_change_on_page:.2f}% "
                  f"which is below threshold {min_52_week_change}%. Stopping early.")
            break
        # ─────────────────────────────────────────────────────────

        all_pages.append(df_page)

        # Standard last-page check
        if len(df_page) < count:
            print(f"Last page reached at start={start} (fewer than {count} rows).")
            break

        start += count
        time.sleep(1.5)  # polite delay

    if not all_pages:
        return pd.DataFrame()

    # Final concatenation + sort + threshold filter (just in case)
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



def append_qvm_data_yfinance(df):
    """
    Enrich an existing DataFrame with QVM metrics using yfinance.
    Expects df to have a "Symbol" column.
    Appends Sector along with Quality, Value, and Momentum metrics.
    Optimized with batch Tickers API to reduce network calls.
    """
    df = df.copy()
    tickers = df["Symbol"].tolist()

    # ---- Download price history for momentum ----
    price_data = yf.download(
        tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    data_map = {}

    # ---- Fetch info for all tickers in batch ----
    tickers_batch = yf.Tickers(" ".join(tickers))
    for ticker in tickers:
        try:
            info = tickers_batch.tickers[ticker].info

            # ---- SECTOR ----
            sector = info.get("sector", "Unknown")

            # ---- QUALITY ----
            roe = info.get("returnOnEquity")
            roa = info.get("returnOnAssets")
            profit_margin = info.get("profitMargins")
            gross_margin = info.get("grossMargins")
            debt_to_equity = info.get("debtToEquity")
            current_ratio = info.get("currentRatio")
            interest_coverage = info.get("interestCoverage")

            # ---- VALUE ----
            pe = info.get("trailingPE")
            price_to_book = info.get("priceToBook")
            peg_ratio = info.get("pegRatio")
            ev = info.get("enterpriseValue")
            ebitda = info.get("ebitda")
            revenue = info.get("totalRevenue")

            ev_ebitda = (ev / ebitda) if ev and ebitda else None
            ev_revenue = (ev / revenue) if ev and revenue else None

            # ---- MOMENTUM ----
            try:
                if len(tickers) == 1:
                    df_prices = price_data
                else:
                    df_prices = price_data[ticker]

                close = df_prices["Close"].dropna()

                ret_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 0 else None
                ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else None
                ret_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else None
                ret_9m = ((close.iloc[-1] / close.iloc[-189]) - 1) * 100 if len(close) > 189 else None

            except Exception:
                ret_1y = ret_6m = ret_3m = ret_9m = None

            data_map[ticker] = {
                "Sector": sector,
                # Quality
                "ROE": roe,
                "ROA": roa,
                "ProfitMargin": profit_margin,
                "GrossMargin": gross_margin,
                "DebtToEquity": debt_to_equity,
                "CurrentRatio": current_ratio,
                "InterestCoverage": interest_coverage,
                # Value
                "PE": pe,
                "PriceToBook": price_to_book,
                "PEG": peg_ratio,
                "EV_EBITDA": ev_ebitda,
                "EV_Revenue": ev_revenue,
                # Momentum
                "3M Return": ret_3m,
                "6M Return": ret_6m,
                "9M Return": ret_9m,
                "1Y Return": ret_1y
            }

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            data_map[ticker] = {"Sector": "Unknown"}

    # ---- Map metrics back to df ----
    all_columns = [
        "Sector", "ROE", "ROA", "ProfitMargin", "GrossMargin", "DebtToEquity",
        "CurrentRatio", "InterestCoverage", "PE", "PriceToBook", "PEG",
        "EV_EBITDA", "EV_Revenue", "3M Return", "6M Return", "9M Return", "1Y Return"
    ]

    for col in all_columns:
        df[col] = df["Symbol"].map(lambda x: data_map.get(x, {}).get(col))

    return df



def score_qvm_clean(df, top_n=100):
    """
    Score and rank stocks using QVM, keeping only essential columns.
    Returns top_n stocks or fewer if the DataFrame is smaller.
    """

    df = df.copy()

    # --- QUALITY SCORE ---
    quality_metrics = ['ROE', 'ROA', 'ProfitMargin', 'GrossMargin', 'CurrentRatio', 'InterestCoverage']
    df['QualityScore'] = df[quality_metrics].mean(axis=1, skipna=True)

    # --- VALUE SCORE ---
    value_metrics = ['PE', 'PEG', 'PriceToBook', 'EV_EBITDA', 'EV_Revenue']

    # Invert each column individually to make lower = better
    df_value_inv = df[value_metrics].apply(lambda col: 1/col.replace(0, np.nan))
    df['ValueScore'] = df_value_inv.mean(axis=1, skipna=True)

    # --- MOMENTUM SCORE ---
    momentum_metrics = ['3M Return', '6M Return', '9M Return', '1Y Return']
    df['MomentumScore'] = df[momentum_metrics].mean(axis=1, skipna=True)

    # --- COMPOSITE QVM SCORE ---
    df['QVMScore'] = df[['QualityScore', 'ValueScore', 'MomentumScore']].mean(axis=1, skipna=True)

    # --- Sort and select top_n stocks ---
    df = df.sort_values('QVMScore', ascending=False)
    df_top = df.head(min(top_n, len(df)))

    # --- Keep only essential columns ---
    essential_columns = [
        'Symbol', 'Name', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %',
        'Avg Vol (3M)', 'Sector', 'ROE', 'ROA', 'ProfitMargin', 'GrossMargin',
        'DebtToEquity', 'CurrentRatio', 'InterestCoverage', 'PE', 'PriceToBook',
        'PEG', 'EV_EBITDA', 'EV_Revenue', '3M Return', '6M Return',
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

df = fetch_all_stock_pages_from_url(url, min_52_week_change)

# Filter rules (adjust thresholds as you like)
df = df[
    (df['Market Cap'] > 300_000_000) &  # remove microcaps < $300M
    (df['P/E Ratio(TTM)'].notnull()) & (df['P/E Ratio(TTM)'] > 0) & (df['P/E Ratio(TTM)'] < 200) &  # avoid negative or extreme PE
    (df['Avg Vol (3M)'] > 100_000) &  # avoid illiquid stocks
    (df['52 WkChange %'].notnull())  # require some price history
    ].copy()

print("\nTrash Filtered Stocks:")
print(df[['Symbol', 'Name', '52 WkChange %']].reset_index(drop=True))

# Assuming df is your full filtered DataFrame
minimal_cols = ['Symbol', 'Name', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %', 'Avg Vol (3M)']
df_minimal = df[minimal_cols].copy()

df_yf = append_qvm_data_yfinance(df_minimal)
df_scored = score_qvm_clean(df_yf)

# Take top 50–100 stocks for your watchlist
top_stocks = df_scored.head(100)
print("\nTop QVM Stocks:")
print(top_stocks.head(15)[['Symbol', '52 WkChange %', '3M Return', 'QVMScore']])

essential_columns_for_gemini = [
    'Symbol',
    'Name',
    'Sector',         # Add this if available
    'Market Cap',
    'P/E Ratio(TTM)',
    'ROE',
    '52 WkChange %',
    '3M Return',      # short-term momentum
    'QVMScore'        # overall quantitative score
]

# Filter your df before sending to Gemini
df_gemini = top_stocks[essential_columns_for_gemini].copy()
symbol_list = df_gemini['Symbol'].tolist()
print(symbol_list)

df_gemini_str = df_gemini.to_string(index=False)
prompt = config["prompt"] + df_gemini_str
max_retries = 4
initial_delay = 10
# # --- Pass the top etfs to Gemini to get world context and final recommendations ---
client, gemini_config = initialize_gemini_client()

final_recommendations, model_used = call_gemini(client, 'gemini-2.5-flash', 'gemini-2.5-flash-lite', gemini_config, prompt)
print("GEMINI RESPONSE:\n")
print(final_recommendations)
print("Generated by model: " + model_used)

end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")

# # --- Update HTML page with recommendations ---
output_columns = [
    'Symbol',
    'Name',
    'Sector',         # Add this if available
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
update_html_page(final_recommendations, df_html_table, "stock_page_template.html","stock_index.html", model_used)