import os
import re
import time
import yaml
import requests
import pandas as pd
from google import genai
import concurrent.futures
from datetime import datetime
from bs4 import BeautifulSoup
from google.genai import types



def extract_first_number(s):
    """Extract the first float number from a string."""
    match = re.search(r'-?\d+\.?\d*', str(s))
    return float(match.group()) if match else None



def fetch_single_stock_page(url, start=0, count=100, retries=3, sleep=2):
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



def parse_stock_table(html):
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



def fetch_all_stock_pages_from_url(url):
    """Fetch and parse all pages for a single ETF list URL."""
    all_pages = []
    start = 0
    count = 100

    while True:
        html = fetch_single_stock_page(url, start=start, count=count)
        if not html:
            print(f"No HTML returned for {url} at start={start}. Stopping this page.")
            break

        df_page = parse_stock_table(html)
        if df_page.empty:
            print(f"No more ETFs found for {url}. Finished this page.")
            break

        all_pages.append(df_page)

        if len(df_page) < count:
            break  # last page reached

        start += count
        time.sleep(1.5)  # polite delay between paginated calls

    return pd.concat(all_pages, ignore_index=True) if all_pages else pd.DataFrame()



def concurrently_fetch_stock_data_from_all_urls():
    """Fetch ETF data from multiple URLs concurrently."""
    all_dfs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(fetch_all_stock_pages_from_url, url): url for url in urls}

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                df_url = future.result()
                if not df_url.empty:
                    print(f"Completed fetching from {url}, {len(df_url)} ETFs found.")
                    all_dfs.append(df_url)
                else:
                    print(f"No data found for {url}.")
            except Exception as e:
                print(f"Error fetching data from {url}: {e}")

    if not all_dfs:
        print("No ETF data retrieved from any page.")
        exit()

    # Combine all dataframes and clean up
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.drop_duplicates(subset='Symbol', keep='first').reset_index(drop=True)
    return df



def cleanup_filter_sort_data(df):
    # --- Clean numeric fields if present ---
    numeric_cols = ['Price', '50 DayAverage', '200 DayAverage', '52 WkChange %', '3 MonthReturn']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('%', '', regex=True).apply(extract_first_number)

    # --- Remove leveraged/risky/foreign ETFs ---
    mask = ~df['Name'].str.contains('|'.join(excluded_keywords), case=False, na=False)
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
    all_etfs_table = top_etfs.to_html(index=False, classes="data-table", border=0)
    top_etfs = top_etfs.to_string(index=False)
    print(top_etfs)

    return top_etfs, all_etfs_table



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



def call_gemini(client, model, gemini_config, prompt):
    def try_model(model):
        # Get response from Gemini
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    config=gemini_config,
                    contents=prompt
                )
                if not response.text:
                    raise ValueError("Empty response from Gemini.")
                return response.text, model
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries} on model {model}: {e}")
                if attempt < max_retries - 1:
                    delay = initial_delay * (3 ** attempt)  # exponential backoff
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(f"Failed after {max_retries} attempts on model '{model}': {e}")

    # First try the specified model, if unavailable, fallback to "-lite" version
    try:
        return try_model(model)
    except Exception as e:
        if not model.endswith("-lite"):
            print(f"Switching to fallback model due to error: {e}")
            return try_model(model+"-lite")
        else:
            raise Exception(f"Failed after {max_retries} attempts on model '{model}': {e}")



def update_html_page(final_recommendations, all_etfs_table, model_used):
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
    with open("template.html", "r", encoding="utf-8") as f:
        template = f.read()

    # --- Insert content ---
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_output = template.replace("<!--LAST_UPDATED_HERE-->", timestamp)
    html_output = html_output.replace("<!--RECOMMENDATIONS_TABLE_HERE-->", gemini_table_html)
    html_output = html_output.replace("<!--RECOMMENDATIONS_SUMMARY_HERE-->", gemini_summary)
    html_output = html_output.replace("<!--FULL_DF_TABLE_HERE-->", all_etfs_table)
    html_output = html_output.replace("<!--MODEL_USED_HERE-->", model_used)

    # --- Write final index.html ---
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_output)



# Record the start time
start_time = time.perf_counter()

# Read configuration
with open("config.yml") as f:
    config = yaml.safe_load(f)
urls = config["urls"]
excluded_keywords = config["excluded_keywords"]
min_52_week_change = config["min_52_week_change"]
min_3_month_return = config["min_3_month_return"]
day_50_average_buffer = config["day_50_average_buffer"]
day_200_average_buffer = config["day_200_average_buffer"]
max_retries = config["max_retries"]
initial_delay = config["initial_delay"]
model = config["model"]

# --- Fetch and parse Stock data from all URLs ---
df = concurrently_fetch_stock_data_from_all_urls()

# --- Get the top performing etfs by cleaning, filtering by keywords and performance, and sorting the data ---
top_etfs, all_etfs_table = cleanup_filter_sort_data(df)

# Record the end time
end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")
print("Getting Google Gemini responses...\n")

# --- Pass the top etfs to Gemini to get world context and final recommendations ---
client, gemini_config = initialize_gemini_client()

prompt_selection = config["prompt_selection"] + top_etfs
reduced_list, model_used = call_gemini(client, model, gemini_config, prompt_selection)
print("GEMINI RESPONSE:\n")
print(reduced_list)
print("Generated by model: " + model_used)

prompt_recommendation = config["prompt_recommendation"] + reduced_list
final_recommendations, model_used = call_gemini(client, model, gemini_config, prompt_recommendation)
print(final_recommendations)
print("Generated by model: " + model_used)

end_time = time.perf_counter()
print(f"Elapsed time: {str(round(end_time - start_time))} seconds\n\n")

# --- Update HTML page with recommendations ---
update_html_page(final_recommendations, all_etfs_table, model_used)
