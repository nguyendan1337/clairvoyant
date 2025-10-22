import os
import re
import time
import yaml
import requests
import pandas as pd
from google import genai
from datetime import datetime
from bs4 import BeautifulSoup
from google.genai import types


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

# Read configuration
with open("config.yml") as f:
    config = yaml.safe_load(f)
urls = config["urls"]
leveraged_keywords = config["leveraged_keywords"]
min_52_week_change = config["min_52_week_change"]
min_3_month_return = config["min_3_month_return"]
day_50_average_buffer = config["day_50_average_buffer"]
day_200_average_buffer = config["day_200_average_buffer"]

# --- Fetch and parse ETF data from all URLs ---
all_dfs = []

for url in urls:
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
mask = ~df['Name'].str.contains('|'.join(leveraged_keywords), case=False, na=False)
df = df[mask]

# --- Apply growth filters ---
filtered_df = df[
    (df['52 WkChange %'] > min_52_week_change) &
    (df['Price'] >= df['50 DayAverage'] * day_50_average_buffer) &
    (df['Price'] >= df['200 DayAverage'] * day_200_average_buffer) &
    (df['3 MonthReturn'] > min_3_month_return)
    ]

# --- Sort by 52-week performance ---
sorted_df = filtered_df.sort_values(by='52 WkChange %', ascending=False).copy()
sorted_df.insert(0, 'Rank', range(1, len(sorted_df) + 1))

# --- Display results ---
print("\nCompiled Top ETFs from Yahoo Finance (sorted by 52 Week Change %):")
top_etfs = sorted_df[['Rank', 'Name', 'Symbol', '52 WkChange %', '3 MonthReturn', 'Price', '50 DayAverage', '200 DayAverage']]
table_html = top_etfs.to_html(index=False, classes="data-table", border=0)
top_etfs = top_etfs.to_string(index=False)
print(top_etfs)

# Record the end time
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds\n\n")
print("Getting Google Gemini response...\n")

# Send results to Google Gemini for analysis and recommendations
api_key = os.getenv("GEMINI_KEY")
if not api_key:
    # fallback to local .env for development
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_KEY")
client = genai.Client(api_key=api_key)
# Enable Google Search tool
grounding_tool = types.Tool(google_search=types.GoogleSearch())
prompt = config["prompt"] + top_etfs
gemini_config = types.GenerateContentConfig(tools=[grounding_tool])
# Get response from Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=gemini_config
)

print("GEMINI RESPONSE:\n")
print(response.text)
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds\n\n")

# --- Extract HTML table and summary from Gemini response ---
# Simple approach: split by first </table>
table_match = re.search(r'(<table.*?/table>)', response.text, flags=re.DOTALL)
if table_match:
    gemini_table_html = table_match.group(1)
    # Anything after the table is the summary
    gemini_summary = response.text.split(gemini_table_html)[-1].strip()
else:
    gemini_table_html = ""
    gemini_summary = response.text  # fallback if no table found

# --- Read HTML template ---
with open("template.html", "r", encoding="utf-8") as f:
    template = f.read()

# --- Insert content ---
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
html_output = template.replace("<!--LAST_UPDATED_HERE-->", timestamp)
html_output = html_output.replace("<!--RECOMMENDATIONS_TABLE_HERE-->", gemini_table_html)
html_output = html_output.replace("<!--RECOMMENDATIONS_SUMMARY_HERE-->", gemini_summary)
html_output = html_output.replace("<!--FULL_DF_TABLE_HERE-->", table_html)

# --- Write final index.html ---
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_output)