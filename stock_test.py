import os
import re
import time
import yaml
import requests
import pandas as pd
from google import genai
from bs4 import BeautifulSoup
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
        'Price', 'Change', 'Change %', 'Volume', 'Avg Vol (3M)',
        'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %'
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

        # Clean numeric columns
        df_page = clean_numeric_columns(df_page, numeric_cols)

        # Drop invalid rows
        df_page = df_page[
            df_page['52 WkChange %'].notna() &
            (df_page['Avg Vol (3M)'] > 0) &
            (df_page['Market Cap'] > 0)
            ]

        if not df_page.empty:
            all_pages.append(df_page)

        # Stop if fewer rows returned than count → last page
        if len(df_page) < count:
            print(f"Last page reached at start={start}.")
            break

        start += count
        time.sleep(1.5)

    if not all_pages:
        return pd.DataFrame()

    # Concatenate and filter by threshold at the end
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
                    raise Exception(f"Failed after {max_retries} attempts on model '{model}': {e}")

    # First try the specified model, if unavailable, fallback to "-lite" version
    try:
        return try_model(model_primary)
    except Exception as e:
        print(f"Switching to fallback model due to error: {e}")
        try:
            return try_model(model_fallback)
        except Exception as fallback_error:
            raise Exception(f"Failed after {max_retries} attempts on model '{model}': {e}")



# Main execution
# Record the start time
start_time = time.perf_counter()

with open("config_test.yml") as f:
    config = yaml.safe_load(f)
url = config["url"]
min_52_week_change = config["min_52_week_change"]

df = fetch_all_stock_pages_from_url(url, min_52_week_change)
df = df[
    # Momentum baseline
    (df['52 WkChange %'] >= 30) &
    (df['52 WkChange %'] < 300)
    ]

df = df[
    # Liquidity OR strong momentum
    (df['Avg Vol (3M)'] >= 500_000) |
    (df['52 WkChange %'] >= 80)
    ]

df = df[
    # Size OR strong momentum
    (df['Market Cap'] >= 2e9) |
    (df['52 WkChange %'] >= 70)
    ]

df = df[
    # P/E logic (from before)
    (
            (df['P/E Ratio(TTM)'] > 0) & (df['P/E Ratio(TTM)'] <= 60)
    ) |
    (
            (df['P/E Ratio(TTM)'].isna() | (df['P/E Ratio(TTM)'] <= 0)) &
            (df['52 WkChange %'] >= 50)
    )
    ]

print("\nFinal DataFrame:")
print(df[['Symbol', 'Name', '52 WkChange %']].reset_index(drop=True))

# Assuming df is your full filtered DataFrame
minimal_cols = ['Symbol', 'Name', 'Market Cap', 'P/E Ratio(TTM)', '52 WkChange %', 'Avg Vol (3M)']
df_minimal = df[minimal_cols].copy()

table_md = df_minimal.to_markdown(index=False)

prompt = f"""
You are an expert quantitative + fundamental investment analyst specializing in Quality-Value-Momentum (QVM) strategies.

CRITICAL CONSTRAINT: The data comes ONLY from a daily Yahoo Finance scrape. I am providing ONLY these minimal columns — NO deep fundamentals (no ROE, ROA, debt, margins, FCF, accruals, EV), NO short-term momentum (only 52-week), and NO sector column.

Here is today's filtered list of stocks (after basic liquidity + size + positive 52-week momentum + reasonable P/E filtering). Columns are exactly:

| Symbol | Name | Market Cap | P/E Ratio(TTM) | 52 WkChange % | Avg Vol (3M) |
{table_md}

Your job is to COMPENSATE for the severely limited data by aggressively using your real-time web search / Google Search grounding tools to fill in the missing QVM pieces.

Task:

1. Assign each stock to its correct GICS sector (Technology, Healthcare, Financials, etc., or "Other"). Use your knowledge or quick web search if uncertain.

2. Build a simple proxy QVM score using ONLY the provided columns:
   - 60% weight: 52 WkChange % (momentum)
   - 25% weight: inverse/low P/E Ratio(TTM) (value)
   - 15% weight: combination of Market Cap (size) + Avg Vol (3M) (liquidity/quality proxy)
   Rank and select the top 50–70 stocks by this proxy score.

3. For those top 50–70 stocks (and for major sectors), perform targeted web searches to add the missing Quality & Value layers:
   - Recent ROE/ROA, debt-to-equity, gross/operating margins, FCF/debt coverage, accrual/earnings quality
   - Latest earnings report highlights (beats/misses, guidance, red flags like high debt or margin erosion)
   - Any major news, catalysts, or risks from the last 7–30 days

4. Group stocks by sector. For each major sector (top 5–7 by count or average proxy score):
   - Stock count in the list
   - Top 3–5 strongest candidates after your research
   - Brief current sector outlook + key recent news/tailwinds/headwinds (last 1–4 weeks)

5. Output your ranked Top 10 final recommendations from the entire list sorted by 52 WkChange %.
   Format strictly so that the columns align and are easy to read:
   Rank | Symbol | Sector | 52 WkChange % | 1-sentence rationale (include key web-discovered quality/value insight)

Be very critical: Reject momentum traps (e.g., weak fundamentals, high debt, poor earnings quality, accounting issues). Only recommend stocks that would plausibly pass a real QVM filter after your research.

CRITICAL OUTPUT RULES:
- Do NOT show any reasoning, steps, analysis, sector groups, research summaries, or intermediate results.
- Do NOT explain why you chose these stocks or describe your process.
- Do NOT include tables, headers, introductions, conclusions, or any extra text.
"""


max_retries = 4
initial_delay = 10
# --- Pass the top etfs to Gemini to get world context and final recommendations ---
client, gemini_config = initialize_gemini_client()

final_recommendations, model_used = call_gemini(client, 'gemini-2.5-flash', 'gemini-2.5-flash-lite', gemini_config, prompt)
print("GEMINI RESPONSE:\n")
print(final_recommendations)
print("Generated by model: " + model_used)
end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")
# table_md = df.to_markdown(index=False)
# df.to_csv("52wk_gainers_20plus.csv", index=False)