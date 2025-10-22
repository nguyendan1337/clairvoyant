import os
import re
import time
import yaml
import requests
import pandas as pd
from google import genai
from bs4 import BeautifulSoup
from google.genai import types
from dotenv import load_dotenv


def extract_first_number(s):
    """Extract the first float number from a string."""
    match = re.search(r'-?\d+\.?\d*', str(s))
    return float(match.group()) if match else None

def fetch_etf_page(url, start=0, count=100, retries=3, sleep=2):
    """Fetch one ETF page with retries and delay, supports pagination for Yahoo Finance."""
    paged_url = f"{url}?start={start}&count={count}"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(paged_url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(sleep)
            else:
                print(f"Failed to fetch {paged_url}: {e}")
                return None

def parse_etf_table(html):
    """Parse HTML table into a DataFrame."""
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')
    if not table:
        return pd.DataFrame()
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    rows = []
    for tr in table.find_all('tr')[1:]:
        tds = [td.get_text(strip=True) for td in tr.find_all('td')]
        if len(tds) == len(headers):
            rows.append(tds)
    return pd.DataFrame(rows, columns=headers)


# Record the start time
start_time = time.perf_counter()

# --- List of Yahoo ETF pages ---
yahoo_urls = [
    "https://finance.yahoo.com/markets/etfs/gainers/",
    "https://finance.yahoo.com/markets/etfs/top-performing/",
    "https://finance.yahoo.com/markets/etfs/best-historical-performance/",
    "https://finance.yahoo.com/markets/etfs/top/"
]

all_dfs = []

for url in yahoo_urls:
    print(f"Fetching ETFs from: {url}")
    start = 0
    count = 100
    while True:
        html = fetch_etf_page(url, start=start, count=count)
        if not html:
            print("No HTML returned. Stopping this page.")
            break

        df_page = parse_etf_table(html)
        if df_page.empty:
            print("No more ETFs found. Finished this page.")
            break

        if not df_page.empty:
            all_dfs.append(df_page)
        if len(df_page) < count:
            break  # Last page reached

        start += count
        time.sleep(1.5)  # polite pause between requests

if not all_dfs:
    print("No ETF data retrieved from any page.")
    exit()

# Combine all pages
df = pd.concat(all_dfs, ignore_index=True)
df = df.drop_duplicates(subset='Symbol', keep='first').reset_index(drop=True)

# --- Clean numeric fields if present ---
numeric_cols = ['Price', '50 DayAverage', '200 DayAverage', '52 WkChange %', '3 MonthReturn']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('%', '', regex=True).apply(extract_first_number)

# --- Remove leveraged/risky ETFs ---
leveraged_keywords = [
    '2x', '3x', 'Ultra', 'Leveraged', 'Bull', 'Bear', 'Double', 'Triple',
    'Enhanced', 'PLUS', 'Short', 'Inverse', 'UltraShort', 'Defined Volatility',
    'Daily', 'Option', 'ETN', 'VIX', 'Junior', 'WeeklyPay', 'Target Income',
    'Covered Call', 'Strategy', 'Derivative', 'Income', 'Bond', 'Futures',
    'Smallcap', 'Microcap', 'Factor', 'Volatility', 'ESG', 'Carbon'
]
mask = ~df['Name'].str.contains('|'.join(leveraged_keywords), case=False, na=False)
df = df[mask]

# --- Apply growth filters ---
filtered_df = df[
    (df['52 WkChange %'] > 25) &
    (df['Price'] >= df['50 DayAverage'] * 0.98) &
    (df['Price'] >= df['200 DayAverage'] * 1.05) &
    (df['3 MonthReturn'] > 7)
    ]

# --- Sort by 52-week performance ---
sorted_df = filtered_df.sort_values(by='52 WkChange %', ascending=False).copy()
sorted_df.insert(0, 'Rank', range(1, len(sorted_df) + 1))

# --- Display results ---
print("\nCompiled Top ETFs from Yahoo Finance (sorted by 52 Week Change %):")
top_etfs = sorted_df[['Rank', 'Name', 'Symbol', '52 WkChange %', '3 MonthReturn', 'Price', '50 DayAverage', '200 DayAverage']].to_string(index=False)
print(top_etfs)

# Record the end time
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds\n\n")


# Send results to Google Gemini for analysis and recommendations

# Load .env file
load_dotenv()  # looks for .env in current directory
# Set API key from .env
api_key = os.getenv("GEMINI_KEY")
# Initialize the GenAI client
client = genai.Client(api_key=api_key)
# Enable Google Search tool
grounding_tool = types.Tool(google_search=types.GoogleSearch())
with open("config.yaml") as f: prompt = yaml.safe_load(f)["prompt"] + top_etfs
gemini_config = types.GenerateContentConfig(tools=[grounding_tool])
# Get response from Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=gemini_config
)

print("GEMINI RESPONSE:")
print(response.text)
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds\n\n")
