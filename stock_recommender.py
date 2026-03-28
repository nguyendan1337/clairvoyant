import os
import re
import time
import yaml
import random
import requests
import numpy as np
import pandas as pd
import requests_cache
import yfinance as yf
from google import genai
import yfinance_cache as yfc
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
            response = scrape_session.get(paged_url, headers=headers, timeout=10)  # <-- cached
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
            print(f"Column '{target_col}' not found. Stopping.")
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
            print(f"Page at start={start} max {target_col} = {max_change_on_page:.2f}% < threshold. Stopping early.")
            break

        all_pages.append(df_page)

        if len(df_page) < count:
            print(f"Last page reached at start={start}.")
            break

        start += count
        time.sleep(1.5)   # still keep a small delay for politeness

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



def append_qvm_data_yfinance(
        df,
        chunk_size: int = 35,
        sleep_seconds: float = 3.0
):
    """
    Enrich DataFrame with QVM metrics using yfinance_cache (intelligent caching).
    Any failure (rate limit, empty data, network error, etc.) will raise an exception
    and stop the entire script.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    df = df.copy()
    tickers = df["Symbol"].tolist()

    if not tickers:
        raise ValueError("No symbols found in the 'Symbol' column.")

    data_map = {}

    print(f"Starting enrichment for {len(tickers)} tickers (chunk_size={chunk_size})")

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

        print(f"→ Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} tickers)")

        try:
            # Price history (original yfinance - fast and reliable)
            price_data = yf.download(
                chunk,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True
            )

            if price_data is None or (isinstance(price_data, pd.DataFrame) and price_data.empty):
                raise RuntimeError(f"Price download returned empty data for chunk starting at index {i}")

            # Fundamentals using yfinance_cache (intelligent caching)
            for ticker in chunk:
                try:
                    # Create Ticker - yfinance_cache handles caching automatically
                    tkr = yfc.Ticker(ticker)

                    info = tkr.info

                    if not info or not isinstance(info, dict):
                        raise RuntimeError(f"Empty or invalid .info returned for {ticker}")

                    # === Extract data ===
                    sector = info.get("sector", "Unknown")

                    # Quality
                    roe = info.get("returnOnEquity")
                    roa = info.get("returnOnAssets")
                    profit_margin = info.get("profitMargins")
                    gross_margin = info.get("grossMargins")
                    debt_to_equity = info.get("debtToEquity")
                    current_ratio = info.get("currentRatio")
                    interest_coverage = info.get("interestCoverage")

                    # Value
                    pe = info.get("trailingPE")
                    price_to_book = info.get("priceToBook")
                    peg_ratio = info.get("pegRatio")
                    ev = info.get("enterpriseValue")
                    ebitda = info.get("ebitda")
                    revenue = info.get("totalRevenue")

                    ev_ebitda = (ev / ebitda) if ev and ebitda and ebitda != 0 else None
                    ev_revenue = (ev / revenue) if ev and revenue and revenue != 0 else None

                    # Momentum
                    try:
                        df_prices = price_data[ticker] if len(chunk) > 1 else price_data
                        close = df_prices["Close"].dropna()

                        if len(close) == 0:
                            raise RuntimeError(f"No price data available for {ticker}")

                        ret_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
                        ret_6m = ((close.iloc[-1] / close.iloc[-126]) - 1) * 100 if len(close) > 126 else None
                        ret_3m = ((close.iloc[-1] / close.iloc[-63]) - 1) * 100 if len(close) > 63 else None
                        ret_9m = ((close.iloc[-1] / close.iloc[-189]) - 1) * 100 if len(close) > 189 else None

                    except Exception as price_err:
                        raise RuntimeError(f"Failed to calculate momentum for {ticker}: {price_err}")

                    data_map[ticker] = {
                        "Sector": sector,
                        "ROE": roe, "ROA": roa, "ProfitMargin": profit_margin,
                        "GrossMargin": gross_margin, "DebtToEquity": debt_to_equity,
                        "CurrentRatio": current_ratio, "InterestCoverage": interest_coverage,
                        "PE": pe, "PriceToBook": price_to_book, "PEG": peg_ratio,
                        "EV_EBITDA": ev_ebitda, "EV_Revenue": ev_revenue,
                        "3M Return": ret_3m, "6M Return": ret_6m,
                        "9M Return": ret_9m, "1Y Return": ret_1y
                    }

                except Exception as ticker_err:
                    raise RuntimeError(f"Critical failure processing ticker '{ticker}': {ticker_err}") from ticker_err

        except Exception as chunk_err:
            raise RuntimeError(f"Critical failure in chunk {chunk_num} (starting with {chunk[0]}): {chunk_err}") from chunk_err

        # Polite delay with jitter
        sleep_time = sleep_seconds + random.uniform(0.5, 2.0)
        time.sleep(sleep_time)

    # Map results back
    cols = ["Sector", "ROE", "ROA", "ProfitMargin", "GrossMargin", "DebtToEquity",
            "CurrentRatio", "InterestCoverage", "PE", "PriceToBook", "PEG",
            "EV_EBITDA", "EV_Revenue", "3M Return", "6M Return", "9M Return", "1Y Return"]

    for col in cols:
        df[col] = df["Symbol"].map(lambda x: data_map.get(x, {}).get(col))

    print(f"Successfully enriched {len(tickers)} tickers.")
    return df



def safe_fillna(series, value):
    filled = series.fillna(value)
    return filled.infer_objects(copy=False)

# def sanity_filter(df):
#     """
#     Soft sanity filter - removes only the most obvious trash.
#     Designed to keep most momentum stocks while dropping clear junk.
#     """
#     df = df.copy()
#     original_count = len(df)
#     print(f"Starting sanity_filter with {original_count} rows")
#
#     # 1. Basic cleaning
#     df = df.replace([np.inf, -np.inf], np.nan)
#
#     numeric_cols = ["ROE", "ROA", "ProfitMargin", "GrossMargin", "DebtToEquity",
#                     "CurrentRatio", "InterestCoverage", "PE", "PriceToBook", "PEG",
#                     "EV_EBITDA", "EV_Revenue", "3M Return", "6M Return",
#                     "9M Return", "1Y Return"]
#
#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     # 2. Very light filters - only remove extreme trash
#     conditions = []
#
#     # A. Extremely bad profitability (very rare to keep these)
#     if 'ROE' in df.columns:
#         conditions.append(safe_fillna(df['ROE'], -1) > -1.0)        # allow up to -100% ROE
#
#     if 'ProfitMargin' in df.columns:
#         conditions.append(safe_fillna(df['ProfitMargin'], -1) > -1.0)
#
#     # B. Extreme leverage
#     if 'DebtToEquity' in df.columns:
#         conditions.append(safe_fillna(df['DebtToEquity'], 20) < 20)   # allow up to 20x
#
#     # C. Valuation sanity (mainly avoid clearly broken numbers)
#     if 'PE' in df.columns:
#         conditions.append(df['PE'].isna() | (df['PE'] > 0))           # drop negative PE only
#
#     if 'EV_EBITDA' in df.columns:
#         conditions.append(df['EV_EBITDA'].isna() | (df['EV_EBITDA'] > 0))
#
#     # D. Momentum - this is the most important one for your screener
#     # Require at least ONE momentum period not completely terrible
#     momentum_cols = ["3M Return", "6M Return", "9M Return", "1Y Return"]
#     existing_momentum = [col for col in momentum_cols if col in df.columns]
#
#     if existing_momentum:
#         momentum_condition = pd.Series(False, index=df.index)
#         for col in existing_momentum:
#             # Allow mildly negative, but drop horrible losers
#             momentum_condition = momentum_condition | (safe_fillna(df[col], -100) > -30)
#         conditions.append(momentum_condition)
#
#     # Apply all conditions (soft AND)
#     if conditions:
#         final_condition = conditions[0]
#         for cond in conditions[1:]:
#             final_condition = final_condition & cond
#
#         df = df[final_condition]
#
#     kept_percent = (len(df) / original_count * 100) if original_count > 0 else 0
#     print(f"After core filters: {len(df)} rows remaining ({kept_percent:.1f}% kept)")
#
#     # 3. Winsorization - clip extreme outliers (safe to keep)
#     def winsorize(series, lower=0.01, upper=0.99):
#         if series.dropna().empty:
#             return series
#         return series.clip(lower=series.quantile(lower), upper=series.quantile(upper))
#
#     cols_to_clip = [col for col in numeric_cols if col in df.columns]
#     for col in cols_to_clip:
#         df[col] = winsorize(df[col])
#
#     # 4. Very light missing data filter
#     core_cols = ["ROE", "ProfitMargin", "DebtToEquity", "PE", "3M Return"]
#     core_cols = [col for col in core_cols if col in df.columns]
#
#     if core_cols:
#         min_required = max(2, len(core_cols) - 2)   # very lenient: allow missing most fields
#         df = df.dropna(thresh=min_required, subset=core_cols)
#
#     print(f"After missing data filter: {len(df)} rows")
#     print(f"Final rows after sanity_filter: {len(df)}")
#
#     return df



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
        '3M Return': 0.2,
        '6M Return': 0.35,
        '9M Return': 0.25,
        '1Y Return': 0.2
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

        # Compute weighted momentum
        weighted_momentum = sum(df_momentum[col] * weight for col, weight in momentum_weights.items() if col in df_momentum.columns)

        # Normalize to 0–100
        min_val = weighted_momentum.min()
        max_val = weighted_momentum.max()
        if max_val != min_val:
            df['MomentumScore'] = (weighted_momentum - min_val) / (max_val - min_val) * 100
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



# ====================== MAIN EXECUTION ======================
if __name__ == "__main__":
    yfc.options.cache_dir = "cache/yfinance"
    print("Cache dir:", getattr(yfc.options, "cache_dir", "NOT SET"))
    # Record the start time
    start_time = time.perf_counter()

    # Global cached session for all scraping (persistent)
    scrape_session = requests_cache.CachedSession(
        'cache/stock_screener_cache',     # separate cache file
        backend='sqlite',
        expire_after=3600 * 6,     # 6 hours - good for screener data
        allowable_methods=('GET',)
    )

    with open("stock_config.yml") as f:
        config = yaml.safe_load(f)
    url = config["url"]
    min_52_week_change = config["min_52_week_change"]
    max_retries = config["max_retries"]
    initial_delay = config["initial_delay"]

    df = fetch_all_stock_pages_from_url(url, min_52_week_change)
    df = df.drop_duplicates()

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
    df = df[minimal_cols].copy()

    df = append_qvm_data_yfinance(df)
    print("\nGrabbed from yfinance:")
    print(df.head(10))
    # df = sanity_filter(df)
    df = score_qvm(df, weights={'Quality': 0.38, 'Value': 0.25, 'Momentum': 0.37})

    # Take top 50–100 stocks for your watchlist
    top_stocks = df.head(100)
    print("\nTop QVM Stocks:")
    print(top_stocks.head(15)[['Symbol', '52 WkChange %', '3M Return', 'QVMScore']])

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

        # Key fundamentals (just enough for reasoning)
        "ROE",
        "ProfitMargin",
        "DebtToEquity",

        # Valuation anchor (pick ONE, not many)
        "PE",

        # Momentum anchors (keep short + medium term)
        "3M Return",
        "52 WkChange %"
    ]

    df_gemini = top_stocks[essential_columns_for_gemini].copy()
    df_gemini_str = df_gemini.to_string(index=False)
    prompt = config["prompt"] + df_gemini_str

    # # --- Pass the top etfs to Gemini to get world context and final recommendations ---
    client, gemini_config = initialize_gemini_client()

    final_recommendations, model_used = call_gemini(client, 'gemini-2.5-flash', 'gemini-2.5-flash-lite', gemini_config, prompt)
    print("\nGEMINI RESPONSE:\n")
    print(final_recommendations)
    print("Generated by model: " + model_used)

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
    update_html_page(final_recommendations, df_html_table, "stock_page_template.html","index.html", model_used)

    #print elapsed time
    end_time = time.perf_counter()
    print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")