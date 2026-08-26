"""Discover, enrich, score, and publish high-momentum ETF recommendations.

Yahoo universe discovery and market data use yfinance. Gemini adds current
context to the quantitative results and the final output is rendered to HTML.
"""

import os
import re
import time
import random
import json
import yaml
from datetime import datetime, timedelta, UTC
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from google import genai
from google.genai import types
from tqdm import tqdm
from yfinance import ETFQuery


YF_CACHE_FILE = 'etf_yf_cache.json'
YF_CACHE_EXPIRY_DAYS = 1
TOP_QVM_CACHE_FILE = 'top_qvm_etfs_cache.pkl'
TOP_QVM_CACHE_EXPIRY_HOURS = 6
TOP_QVM_CACHE_VERSION = 5


def initialize_gemini_client():
    api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parent / '.env'
        load_dotenv(dotenv_path=env_path)
        api_key = os.getenv('GEMINI_KEY')
    if not api_key:
        raise ValueError('GEMINI_KEY not found in environment or .env')

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    gemini_config = types.GenerateContentConfig(tools=[grounding_tool])
    return genai.Client(api_key=api_key), gemini_config


def call_gemini(client, model_primary, model_fallback, gemini_config, prompt):
    def try_model(model_name):
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    config=gemini_config,
                    contents=prompt,
                )
                if not response.text:
                    raise ValueError('Empty response from Gemini.')
                return response.text, model_name
            except Exception as exc:
                print(
                    f'Gemini attempt {attempt + 1}/{max_retries} '
                    f'failed on {model_name}: {exc}'
                )
                if attempt < max_retries - 1:
                    delay = initial_delay * (3 ** attempt)
                    print(f'Retrying in {delay}s...')
                    time.sleep(delay)
                else:
                    raise

    try:
        return try_model(model_primary)
    except Exception as exc:
        print(f'Switching to fallback model due to error: {exc}')
        return try_model(model_fallback)


def update_html_page(
    final_recommendations,
    df_html_table,
    template_name,
    display_page,
    model_used,
):
    table_match = re.search(
        r'(<table.*?</table>)',
        final_recommendations,
        flags=re.DOTALL | re.IGNORECASE,
    )
    summary_match = re.search(
        r'(<div[^>]*class=["\']summary["\'][^>]*>.*?</div>)',
        final_recommendations,
        flags=re.DOTALL | re.IGNORECASE,
    )
    gemini_table_html = table_match.group(1).strip() if table_match else ''
    gemini_summary = summary_match.group(1).strip() if summary_match else ''

    if not gemini_table_html and '<table' in final_recommendations:
        start = final_recommendations.find('<table')
        end = final_recommendations.find('</table>', start)
        if end != -1:
            gemini_table_html = final_recommendations[start:end + 8]
    if not gemini_summary and '<div' in final_recommendations:
        start = final_recommendations.find('<div')
        end = final_recommendations.find('</div>', start)
        if end != -1:
            gemini_summary = final_recommendations[start:end + 6]

    with open(template_name, 'r', encoding='utf-8') as f:
        template = f.read()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_output = template.replace('<!--LAST_UPDATED_HERE-->', timestamp)
    html_output = html_output.replace(
        '<!--RECOMMENDATIONS_TABLE_HERE-->',
        gemini_table_html,
    )
    html_output = html_output.replace(
        '<!--RECOMMENDATIONS_SUMMARY_HERE-->',
        gemini_summary,
    )
    html_output = html_output.replace('<!--FULL_DF_TABLE_HERE-->', df_html_table)
    html_output = html_output.replace('<!--MODEL_USED_HERE-->', model_used)
    with open(display_page, 'w', encoding='utf-8') as f:
        f.write(html_output)


def extract_number_with_suffix(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {'N/A', 'NONE', '-', '', '—', 'NAN'}:
        return None
    s = s.replace(',', '').replace('$', '')
    match = re.search('-?[\\d.]+', s)
    if not match:
        return None
    try:
        number = float(match.group())
    except ValueError:
        return None
    if 'T' in s:
        number *= 1000000000000.0
    elif 'B' in s:
        number *= 1000000000.0
    elif 'M' in s:
        number *= 1000000.0
    elif 'K' in s:
        number *= 1000.0
    return number


def clean_percent(value):
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().replace(',', '').replace('+', '').replace('%', '')
    try:
        return float(s)
    except ValueError:
        return None


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def load_json_cache(path, expiry_days):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except Exception as exc:
        print(f'Could not load {path}: {exc}')
        return {}
    fresh_cache = {}
    now = datetime.now(UTC)
    for key, entry in cache.items():
        try:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            if now - timestamp < timedelta(days=expiry_days):
                fresh_cache[key] = entry
        except Exception:
            continue
    return fresh_cache


def save_json_cache(path, cache):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f)
    except Exception as exc:
        print(f'Could not save {path}: {exc}')


def load_top_qvm_cache():
    print(f'Checking top ETF QVM cache: {TOP_QVM_CACHE_FILE}', flush=True)
    if not os.path.exists(TOP_QVM_CACHE_FILE):
        print('Top ETF QVM cache does not exist.', flush=True)
        return None
    try:
        cached = pd.read_pickle(TOP_QVM_CACHE_FILE)
        if not isinstance(cached, dict):
            print('Invalid top ETF QVM cache format. Rebuilding.')
            return None
        if cached.get('version') != TOP_QVM_CACHE_VERSION:
            print('Top ETF QVM cache version mismatch. Rebuilding.')
            return None
        created_at = cached.get('created_at')
        if not created_at:
            print('Top ETF QVM cache has no creation timestamp. Rebuilding.')
            return None
        created_time = datetime.fromisoformat(created_at)
        if created_time.tzinfo is None:
            created_time = created_time.replace(tzinfo=UTC)
        age = datetime.now(UTC) - created_time
        if age >= timedelta(hours=TOP_QVM_CACHE_EXPIRY_HOURS):
            print(f'Top ETF QVM cache is stale ({age.total_seconds() / 3600:.1f}h old).')
            return None
        df = cached.get('data')
        if not isinstance(df, pd.DataFrame):
            print('Top ETF QVM cache data is invalid. Rebuilding.')
            return None
        if df.empty:
            print('Top ETF QVM cache is empty. Rebuilding.')
            return None
        print(
            f'Using cached top {len(df)} ETF QVM candidates '
            f'({age.total_seconds() / 3600:.1f}h old).'
        )
        return df
    except Exception as exc:
        print(f'Could not load top ETF QVM cache: {exc}')
        return None


def save_top_qvm_cache(df):
    try:
        cache = {
            'version': TOP_QVM_CACHE_VERSION,
            'created_at': datetime.now(UTC).isoformat(),
            'data': df.copy(),
        }
        pd.to_pickle(cache, TOP_QVM_CACHE_FILE)
        print(f'Saved top ETF QVM cache to {TOP_QVM_CACHE_FILE}.')
    except Exception as exc:
        print(f'Could not save top ETF QVM cache: {exc}')


def fetch_etf_universe(min_52_week_change, max_retries=4, page_size=250):
    """Fetch every US-region ETF meeting the 52-week return threshold."""
    query = ETFQuery(
        'and',
        [
            ETFQuery('eq', ['region', 'us']),
            ETFQuery(
                'gte',
                ['fiftytwowkpercentchange', min_52_week_change],
            ),
        ],
    )
    rows, offset = ([], 0)
    while True:
        result = None
        for attempt in range(1, max_retries + 1):
            try:
                result = yf.screen(
                    query,
                    offset=offset,
                    size=page_size,
                    sortField='fiftytwowkpercentchange',
                    sortAsc=False,
                )
                break
            except Exception as exc:
                print(f'ETF screener attempt {attempt}/{max_retries} failed at offset {offset}: {exc}')
                if attempt < max_retries:
                    time.sleep(2 + random.uniform(0, 1))
        if result is None:
            raise RuntimeError(f'Yahoo ETF screener failed at offset {offset}.')
        page = result.get('quotes') or []
        if not page:
            break
        rows.extend(page)
        print(f'Fetched {len(page)} ETF screener rows at offset {offset}.')
        offset += len(page)
        total = result.get('total')
        if len(page) < page_size or (total is not None and offset >= total):
            break
    if not rows:
        raise RuntimeError('Yahoo ETF screener returned no matching ETFs.')
    df = pd.DataFrame(rows)
    if 'symbol' not in df.columns:
        raise RuntimeError('Yahoo ETF screener response has no symbol field.')
    df['Name'] = df.get('longName', pd.Series(index=df.index, dtype=object))
    if 'shortName' in df.columns:
        df['Name'] = df['Name'].fillna(df['shortName'])
    df = df.rename(
        columns={
            'symbol': 'Symbol',
            'fiftyTwoWeekChangePercent': '52 WkChange %',
            'regularMarketPrice': 'Price',
            'averageDailyVolume3Month': 'Avg Vol (3M)',
            'trailingThreeMonthReturns': '3 MonthReturn',
        }
    )
    if '52 WkChange %' not in df.columns and 'fiftyTwoWeekChange' in df.columns:
        df = df.rename(columns={'fiftyTwoWeekChange': '52 WkChange %'})
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    df = df[df['Symbol'].str.match('^[A-Z0-9.-]+$', na=False)]
    df = df.drop_duplicates('Symbol', keep='first')
    if '52 WkChange %' in df.columns:
        df['52 WkChange %'] = pd.to_numeric(df['52 WkChange %'], errors='coerce')
        df = df[df['52 WkChange %'] >= min_52_week_change]
        df = df.sort_values('52 WkChange %', ascending=False)
    return df.reset_index(drop=True)


def apply_basic_etf_filters(df, excluded_keywords, min_52_week_change, min_3_month_return):
    df = df.copy()
    if 'Name' not in df.columns:
        df['Name'] = df['Symbol']
    numeric_cols = ['Price', '50 DayAverage', '200 DayAverage', '52 WkChange %', '3 MonthReturn']
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if col in {'52 WkChange %', '3 MonthReturn'}:
            df[col] = df[col].apply(clean_percent)
        else:
            df[col] = df[col].apply(extract_number_with_suffix)
    if excluded_keywords:
        pattern = '|'.join((re.escape(keyword) for keyword in excluded_keywords))
        df = df[~df['Name'].astype(str).str.contains(pattern, case=False, na=False, regex=True)]
    if '52 WkChange %' in df.columns:
        df = df[df['52 WkChange %'].notna() & (df['52 WkChange %'] >= min_52_week_change)]
    if '3 MonthReturn' in df.columns:
        df = df[df['3 MonthReturn'].isna() | (df['3 MonthReturn'] >= min_3_month_return)]
    return df.drop_duplicates(subset='Symbol').reset_index(drop=True)


def download_price_history(symbols, period='13mo'):
    if not symbols:
        return {}
    print(f'Downloading ETF price history for {len(symbols)} symbols...')
    try:
        price_data = yf.download(
            symbols,
            period=period,
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f'yfinance bulk download failed: {exc}')
        return {}
    histories = {}
    for symbol in symbols:
        try:
            if len(symbols) == 1:
                close = price_data['Close'].dropna()
            else:
                if (
                    not hasattr(price_data, 'columns')
                    or symbol not in price_data.columns.get_level_values(0)
                ):
                    continue
                close = price_data[symbol]['Close'].dropna()
            if not close.empty:
                histories[symbol] = close
        except Exception:
            continue
    return histories


def return_from_history(close, periods):
    if close is None:
        return None
    if len(close) <= periods:
        return None
    try:
        return float((close.iloc[-1] / close.iloc[-periods] - 1) * 100)
    except Exception:
        return None


def compute_momentum_metrics(close):
    if close is None:
        return {}
    if len(close) < 20:
        return {}
    return {
        'Price': safe_float(close.iloc[-1]),
        '1M Return': return_from_history(close, 21),
        '3M Return': return_from_history(close, 63),
        '6M Return': return_from_history(close, 126),
        '9M Return': return_from_history(close, 189),
        '1Y Return': return_from_history(close, 252),
        'Volatility 1Y': safe_float(
            close.pct_change().std() * np.sqrt(252) * 100
        ),
        '50D Average': safe_float(close.rolling(50).mean().iloc[-1]),
        '200D Average': safe_float(close.rolling(200).mean().iloc[-1]),
    }


def append_etf_yfinance_data(
    df,
    max_info_calls=500,
    delay=0.5,
    priority_symbols=None,
):
    df = df.copy()
    df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    symbols = df['Symbol'].dropna().astype(str).str.upper().unique().tolist()
    histories = download_price_history(symbols, period='13mo')
    data_map = {}
    momentum_strength = {}
    print('Computing ETF momentum metrics...')
    for symbol in symbols:
        metrics = compute_momentum_metrics(histories.get(symbol))
        data_map[symbol] = metrics
        momentum_components = [
            metrics.get('3M Return'),
            metrics.get('6M Return'),
            metrics.get('9M Return'),
        ]
        valid = [value for value in momentum_components if value is not None]
        if valid:
            momentum_strength[symbol] = float(np.mean(valid))
        else:
            momentum_strength[symbol] = -np.inf
    sorted_symbols = sorted(
        momentum_strength,
        key=lambda symbol: momentum_strength[symbol],
        reverse=True,
    )
    priority_symbols = {
        str(symbol).strip().upper()
        for symbol in (priority_symbols or [])
    }
    selected_for_info = list(
        dict.fromkeys(
            list(priority_symbols) + sorted_symbols[:max_info_calls]
        )
    )
    print(
        'Fetching yfinance metadata for top '
        f'{len(selected_for_info)} momentum candidates...'
    )
    cache = load_json_cache(YF_CACHE_FILE, YF_CACHE_EXPIRY_DAYS)
    for symbol in tqdm(selected_for_info):
        try:
            if symbol in cache:
                info = cache[symbol]['info']
            else:
                ticker = yf.Ticker(symbol)
                raw_info = ticker.info or {}
                info_fields = [
                    'longName',
                    'shortName',
                    'quoteType',
                    'category',
                    'fundFamily',
                    'legalType',
                    'currency',
                    'country',
                    'exchange',
                    'totalAssets',
                    'netAssets',
                    'annualReportExpenseRatio',
                    'netExpenseRatio',
                    'expenseRatio',
                    'averageVolume',
                    'averageVolume10days',
                    'beta3Year',
                    'threeYearAverageReturn',
                    'fiveYearAverageReturn',
                    'ytdReturn',
                    'yield',
                    'trailingPE',
                    'priceToBook',
                    'fundInceptionDate',
                ]
                info = {
                    field: raw_info.get(field)
                    for field in info_fields
                }
                cache[symbol] = {
                    'info': info,
                    'timestamp': datetime.now(UTC).isoformat(),
                }
                time.sleep(delay + random.uniform(0, 0.3))
            data_map[symbol].update(
                {
                    'Name': info.get('longName') or info.get('shortName'),
                    'Category': info.get('category'),
                    'FundFamily': info.get('fundFamily'),
                    'LegalType': info.get('legalType'),
                    'Country': info.get('country'),
                    'Exchange': info.get('exchange'),
                    'AUM': safe_float(
                        info.get('totalAssets') or info.get('netAssets')
                    ),
                    'ExpenseRatio': safe_float(
                        info.get('annualReportExpenseRatio')
                        or info.get('netExpenseRatio')
                        or info.get('expenseRatio')
                    ),
                    'AverageVolume': safe_float(
                        info.get('averageVolume')
                        or info.get('averageVolume10days')
                    ),
                    'Beta3Y': safe_float(info.get('beta3Year')),
                    '3Y Average Return': safe_float(
                        info.get('threeYearAverageReturn')
                    ),
                    '5Y Average Return': safe_float(
                        info.get('fiveYearAverageReturn')
                    ),
                    'YTD Return': safe_float(info.get('ytdReturn')),
                    'Yield': safe_float(info.get('yield')),
                    'TrailingPE': safe_float(info.get('trailingPE')),
                    'PriceToBook': safe_float(info.get('priceToBook')),
                }
            )
        except Exception as exc:
            print(f'Error retrieving yfinance metadata for {symbol}: {exc}')
    save_json_cache(YF_CACHE_FILE, cache)
    columns = [
        'Name',
        'Category',
        'FundFamily',
        'LegalType',
        'Country',
        'Exchange',
        'Price',
        'AUM',
        'ExpenseRatio',
        'AverageVolume',
        'Beta3Y',
        '3Y Average Return',
        '5Y Average Return',
        'YTD Return',
        'Yield',
        'TrailingPE',
        'PriceToBook',
        'Volatility 1Y',
        '50D Average',
        '200D Average',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
    ]
    for col in columns:
        df[col] = df['Symbol'].map(lambda symbol: data_map.get(symbol, {}).get(col))
    discovery = df.set_index('Symbol').to_dict('index')
    if '52 WkChange %' in df.columns:
        df['52 WkChange %'] = df.apply(
            lambda row: (
                row['52 WkChange %']
                if pd.notna(row['52 WkChange %'])
                else discovery.get(row['Symbol'], {}).get(
                    '52 WkChange %'
                )
            ),
            axis=1,
        )
    return (df, histories)


def get_benchmark_data(histories, benchmark_symbols):
    missing = [symbol for symbol in benchmark_symbols if symbol not in histories]
    if missing:
        print(f'Downloading missing benchmark history: {missing}')
        missing_histories = download_price_history(missing, period='13mo')
        histories.update(missing_histories)
    benchmark_metrics = {}
    for symbol in benchmark_symbols:
        benchmark_metrics[symbol] = compute_momentum_metrics(histories.get(symbol))
    return benchmark_metrics


def get_discovery_thresholds(config, benchmark_metrics, benchmark_symbols):
    one_year_returns = [
        benchmark_metrics.get(symbol, {}).get('1Y Return')
        for symbol in benchmark_symbols
    ]
    three_month_returns = [
        benchmark_metrics.get(symbol, {}).get('3M Return')
        for symbol in benchmark_symbols
    ]
    one_year_returns = [value for value in one_year_returns if value is not None]
    three_month_returns = [
        value for value in three_month_returns if value is not None
    ]
    configured_1y = config['min_52_week_change']
    configured_3m = config['min_3_month_return']
    one_year_tolerance = config.get('benchmark_1y_discovery_tolerance', 10.0)
    three_month_tolerance = config.get('benchmark_3m_discovery_tolerance', 5.0)
    discovery_1y = configured_1y
    discovery_3m = configured_3m
    if one_year_returns:
        discovery_1y = min(
            configured_1y,
            min(one_year_returns) - one_year_tolerance,
        )
    if three_month_returns:
        discovery_3m = min(
            configured_3m,
            min(three_month_returns) - three_month_tolerance,
        )
    return discovery_1y, discovery_3m


def get_benchmark_superiority_mask(scored_etfs, benchmark_df, benchmark_symbols):
    candidates = ~scored_etfs['Symbol'].isin(benchmark_symbols)
    complete_history = (
        scored_etfs['3M Return'].notna()
        & scored_etfs['1Y Return'].notna()
    )
    if benchmark_df.empty or benchmark_df['QVMScore'].isna().all():
        raise ValueError('Benchmark QVM scores are required for candidate filtering.')
    qvm_superior = scored_etfs['QVMScore'] > benchmark_df['QVMScore'].max()
    benchmark_3m_floor = benchmark_df['3M Return'].mean()
    benchmark_1y_floor = benchmark_df['1Y Return'].mean()
    competitive_horizons = (
        scored_etfs['3M Return'].ge(benchmark_3m_floor)
        & scored_etfs['1Y Return'].ge(benchmark_1y_floor)
    )

    benchmark_mean_cols = [
        f'{period} Excess vs Benchmarks'
        for period in ['1M', '3M', '6M', '9M', '1Y']
        if f'{period} Excess vs Benchmarks' in scored_etfs.columns
    ]
    if len(benchmark_mean_cols) < 3:
        raise ValueError('At least three benchmark-relative periods are required.')
    majority_periods = (
        scored_etfs[benchmark_mean_cols].gt(0).sum(axis=1)
        >= int(np.ceil(len(benchmark_mean_cols) / 2))
    )

    relative_to_each = []
    for benchmark in benchmark_symbols:
        cols = [
            f'{period} Excess vs {benchmark}'
            for period in ['1M', '3M', '6M', '9M', '1Y']
            if f'{period} Excess vs {benchmark}' in scored_etfs.columns
        ]
        if cols:
            relative_to_each.append(scored_etfs[cols].mean(axis=1) > 0)
    if len(relative_to_each) != len(benchmark_symbols):
        raise ValueError('Relative-return data is required for every benchmark.')
    positive_average_vs_each = pd.concat(relative_to_each, axis=1).all(axis=1)
    return (
        candidates
        & complete_history
        & qvm_superior
        & competitive_horizons
        & majority_periods
        & positive_average_vs_each
    )


def add_benchmark_relative_metrics(df, benchmark_metrics, benchmark_symbols):
    df = df.copy()
    periods = ['1M', '3M', '6M', '9M', '1Y']
    for period in periods:
        col = f'{period} Return'
        if col not in df.columns:
            continue
        benchmark_returns = []
        for benchmark in benchmark_symbols:
            value = benchmark_metrics.get(benchmark, {}).get(col)
            if value is not None:
                benchmark_returns.append(value)
        if not benchmark_returns:
            continue
        benchmark_mean = float(np.mean(benchmark_returns))
        df[f'{period} Excess vs Benchmarks'] = df[col] - benchmark_mean
        for benchmark in benchmark_symbols:
            benchmark_return = benchmark_metrics.get(benchmark, {}).get(col)
            if benchmark_return is not None:
                df[f'{period} Excess vs {benchmark}'] = df[col] - benchmark_return
    excess_cols = [
        f'{period} Excess vs Benchmarks'
        for period in periods
        if f'{period} Excess vs Benchmarks' in df.columns
    ]
    if excess_cols:
        df['BenchmarkRelativeScoreRaw'] = df[excess_cols].mean(axis=1)
        df['BenchmarkRelativeScore'] = (
            df['BenchmarkRelativeScoreRaw'].rank(pct=True) * 100
        ).fillna(50)
    return df


def winsorized_rank(series, ascending=True):
    s = pd.to_numeric(series, errors='coerce').copy()
    valid = s.dropna()
    if len(valid) >= 20:
        lower = valid.quantile(0.02)
        upper = valid.quantile(0.98)
        s = s.clip(lower, upper)
    return s.rank(pct=True, ascending=ascending)


def score_etf_qvm(df, top_n=100, weights=None, min_quality=35):
    df = df.copy()
    if weights is None:
        weights = {'Quality': 0.25, 'Value': 0.15, 'Momentum': 0.6}
    numeric_columns = [
        'AUM',
        'ExpenseRatio',
        'AverageVolume',
        'Beta3Y',
        '3Y Average Return',
        '5Y Average Return',
        'YTD Return',
        'Yield',
        'TrailingPE',
        'PriceToBook',
        'Volatility 1Y',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
        'BenchmarkRelativeScore',
    ]
    for col in numeric_columns:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    if df.empty:
        return pd.DataFrame()
    quality_components = []
    if 'AUM' in df.columns:
        aum = np.log1p(df['AUM'].clip(lower=0))
        quality_components.append(winsorized_rank(aum, ascending=True))
    if 'AverageVolume' in df.columns:
        volume = np.log1p(df['AverageVolume'].clip(lower=0))
        quality_components.append(winsorized_rank(volume, ascending=True))
    if 'ExpenseRatio' in df.columns:
        expense = df['ExpenseRatio'].clip(lower=0)
        quality_components.append(winsorized_rank(expense, ascending=False))
    if '3Y Average Return' in df.columns:
        quality_components.append(winsorized_rank(df['3Y Average Return'], ascending=True))
    if '5Y Average Return' in df.columns:
        quality_components.append(winsorized_rank(df['5Y Average Return'], ascending=True))
    if quality_components:
        q = pd.concat(quality_components, axis=1)
        valid_count = q.notna().sum(axis=1)
        quality = q.mean(axis=1, skipna=True)
        confidence = np.where(
            valid_count >= 3,
            1.0,
            np.where(
                valid_count == 2,
                0.85,
                np.where(valid_count == 1, 0.65, 0.5),
            ),
        )
        quality = 0.5 + (quality - 0.5) * confidence
        quality = quality.where(valid_count > 0, 0.5)
        df['QualityScore'] = (quality * 100).clip(0, 100)
    else:
        df['QualityScore'] = 50.0
    if min_quality > 0:
        df = df[df['QualityScore'] >= min_quality].copy()
    if df.empty:
        return pd.DataFrame()
    value_components = []
    if 'ExpenseRatio' in df.columns:
        value_components.append(winsorized_rank(df['ExpenseRatio'].clip(lower=0), ascending=False))
    if 'TrailingPE' in df.columns:
        pe = df['TrailingPE'].where(df['TrailingPE'] > 0)
        value_components.append(winsorized_rank(pe, ascending=False))
    if 'PriceToBook' in df.columns:
        pb = df['PriceToBook'].where(df['PriceToBook'] > 0)
        value_components.append(winsorized_rank(pb, ascending=False))
    if value_components:
        v = pd.concat(value_components, axis=1)
        valid_count = v.notna().sum(axis=1)
        value = v.mean(axis=1, skipna=True)
        confidence = np.where(valid_count >= 2, 1.0, np.where(valid_count == 1, 0.65, 0.5))
        value = 0.5 + (value - 0.5) * confidence
        value = value.where(valid_count > 0, 0.5)
        df['ValueScore'] = (value * 100).clip(0, 100)
    else:
        df['ValueScore'] = 50.0
    momentum_columns = ['1M Return', '3M Return', '6M Return', '9M Return', '1Y Return']
    m_cols = [col for col in momentum_columns if col in df.columns]
    if m_cols:
        m = df[m_cols].copy().clip(lower=-100, upper=500)
        mean_return = m.mean(axis=1, skipna=True)
        raw_momentum = mean_return.rank(pct=True) * 100
        valid_count = m.notna().sum(axis=1)
        positive_windows = (m > 0).sum(axis=1)
        trend_alignment = positive_windows / valid_count.replace(0, np.nan) * 100
        trend_alignment = trend_alignment.fillna(50)
        consistency_columns = [
            col
            for col in ['3M Return', '6M Return', '1Y Return']
            if col in df.columns
        ]
        if len(consistency_columns) >= 2:
            consistency = df[consistency_columns].rank(pct=True, axis=0)
            consistency_score = consistency.mean(axis=1, skipna=True) * 100
            dispersion = consistency.std(axis=1, skipna=True)
            consistency_score = (consistency_score - dispersion * 25).clip(0, 100).fillna(50)
        else:
            consistency_score = raw_momentum
        if 'Volatility 1Y' in df.columns:
            volatility = df['Volatility 1Y'].clip(lower=1)
            risk_adjusted_return = mean_return / volatility
            risk_adjusted_score = (risk_adjusted_return.rank(pct=True) * 100).fillna(50)
        else:
            risk_adjusted_score = raw_momentum
        if '3M Return' in df.columns:
            recent_score = (df['3M Return'].rank(pct=True) * 100).fillna(50)
        else:
            recent_score = raw_momentum
        if 'BenchmarkRelativeScore' in df.columns:
            benchmark_score = df['BenchmarkRelativeScore'].fillna(50)
        else:
            benchmark_score = pd.Series(50, index=df.index)
        momentum = (
            0.25 * raw_momentum
            + 0.25 * trend_alignment
            + 0.20 * consistency_score
            + 0.10 * risk_adjusted_score
            + 0.10 * recent_score
            + 0.10 * benchmark_score
        )
        df['MomentumScore'] = momentum.fillna(50).clip(0, 100)
    else:
        df['MomentumScore'] = 50.0
    if '3M Return' in df.columns:
        recent_return = df['3M Return'].fillna(0)
        recent_penalty = np.where(recent_return < 0, np.minimum(20, -recent_return * 0.5), 0)
        df['MomentumScore'] = (df['MomentumScore'] - recent_penalty).clip(0, 100)
    df['QVMScore'] = (
        df['QualityScore'] * weights['Quality']
        + df['ValueScore'] * weights['Value']
        + df['MomentumScore'] * weights['Momentum']
    )
    if 'Volatility 1Y' in df.columns:
        extreme_volatility = df['Volatility 1Y'] > 70
        df.loc[extreme_volatility, 'QVMScore'] *= 0.92
    return (
        df.sort_values(
            ['QVMScore', 'MomentumScore'],
            ascending=[False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )


start_time = time.perf_counter()
with open('etf_config.yml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
excluded_keywords = config['excluded_keywords']
min_52_week_change = config['min_52_week_change']
min_3_month_return = config['min_3_month_return']
max_retries = config['max_retries']
initial_delay = config['initial_delay']
model_primary = config['model_primary']
model_fallback = config['model_fallback']
benchmark_etfs = [
    str(symbol).strip().upper()
    for symbol in config['benchmark_etfs']
]
if not benchmark_etfs:
    raise ValueError('benchmark_etfs must contain at least one symbol.')
max_info_calls = config.get('max_info_calls', 500)
top_n = config.get('qvm_top_n', 50)
qvm_weights = config.get('qvm_weights', {'Quality': 0.25, 'Value': 0.15, 'Momentum': 0.6})
min_quality = config.get('min_quality', 35)
top_etfs = load_top_qvm_cache()
if top_etfs is None:
    seed_histories = download_price_history(benchmark_etfs, period='13mo')
    seed_benchmark_metrics = get_benchmark_data(seed_histories, benchmark_etfs)
    discovery_1y, discovery_3m = get_discovery_thresholds(
        config,
        seed_benchmark_metrics,
        benchmark_etfs,
    )
    print(
        'Benchmark-relative discovery floors: '
        f'1Y {discovery_1y:.2f}%, 3M {discovery_3m:.2f}%.'
    )
    df = fetch_etf_universe(discovery_1y, max_retries)
    print(f'\nRaw ETF universe: {len(df)}')
    df = apply_basic_etf_filters(
        df,
        excluded_keywords,
        discovery_1y,
        discovery_3m,
    )
    print(f'After basic ETF filtering: {len(df)}')
    benchmark_rows = pd.DataFrame({'Symbol': benchmark_etfs, 'Name': benchmark_etfs})
    df = pd.concat([df, benchmark_rows], ignore_index=True)
    df = df.drop_duplicates(subset='Symbol', keep='first').reset_index(drop=True)
    df_yf, histories = append_etf_yfinance_data(
        df,
        max_info_calls=max_info_calls,
        priority_symbols=benchmark_etfs,
    )
    benchmark_metrics = get_benchmark_data(histories, benchmark_etfs)
    df_yf = add_benchmark_relative_metrics(df_yf, benchmark_metrics, benchmark_etfs)
    is_benchmark = df_yf['Symbol'].isin(benchmark_etfs)
    if 'Price' in df_yf.columns:
        price_eligible = (
            df_yf['Price'].notna()
            & (df_yf['Price'] >= config.get('min_price', 5.0))
        )
        df_yf = df_yf[is_benchmark | price_eligible].copy()
        is_benchmark = df_yf['Symbol'].isin(benchmark_etfs)
    if 'AverageVolume' in df_yf.columns:
        volume_eligible = (
            df_yf['AverageVolume'].isna()
            | (
                df_yf['AverageVolume']
                >= config.get('min_average_volume', 100000)
            )
        )
        df_yf = df_yf[is_benchmark | volume_eligible].copy()
        is_benchmark = df_yf['Symbol'].isin(benchmark_etfs)
    if 'AUM' in df_yf.columns:
        aum_eligible = (
            df_yf['AUM'].isna()
            | (df_yf['AUM'] >= config.get('min_aum', 100000000))
        )
        df_yf = df_yf[is_benchmark | aum_eligible].copy()

    scored_etfs = score_etf_qvm(
        df_yf,
        top_n=len(df_yf),
        weights=qvm_weights,
        min_quality=0,
    )
    benchmark_df = scored_etfs[
        scored_etfs['Symbol'].isin(benchmark_etfs)
    ].copy()
    print(
        'Final benchmark-relative floors: '
        f'3M {benchmark_df["3M Return"].mean():.2f}%, '
        f'1Y {benchmark_df["1Y Return"].mean():.2f}%.'
    )
    superiority_mask = get_benchmark_superiority_mask(
        scored_etfs,
        benchmark_df,
        benchmark_etfs,
    )
    candidate_mask = (
        superiority_mask
        & (scored_etfs['QualityScore'] >= min_quality)
    )
    candidate_etfs = scored_etfs[candidate_mask].head(top_n)
    print(
        f'Benchmark-superior ETF candidates: {len(candidate_etfs)} '
        f'(maximum requested: {top_n}).'
    )
    if not benchmark_df.empty:
        top_etfs = pd.concat(
            [candidate_etfs, benchmark_df],
            ignore_index=True,
        )
        top_etfs = top_etfs.drop_duplicates(subset='Symbol', keep='first')
    else:
        top_etfs = candidate_etfs
    top_etfs = top_etfs.sort_values(
        'QVMScore',
        ascending=False,
        na_position='last',
    ).reset_index(drop=True)
    top_etfs['Role'] = np.where(
        top_etfs['Symbol'].isin(benchmark_etfs),
        'BENCHMARK',
        'CANDIDATE',
    )
    save_top_qvm_cache(top_etfs)
print('\nTop ETF QVM candidates:')
display_columns = [
    col
    for col in [
        'Role',
        'Symbol',
        'QVMScore',
        'QualityScore',
        'ValueScore',
        'MomentumScore',
        'BenchmarkRelativeScore',
        '3M Return',
        '6M Return',
        '1Y Return',
    ]
    if col in top_etfs.columns
]
print(top_etfs[display_columns].head(50).to_string(index=False))
inspection_columns = [
    col
    for col in [
        'Role',
        'Symbol',
        'Name',
        'Category',
        'QVMScore',
        'QualityScore',
        'ValueScore',
        'MomentumScore',
        'BenchmarkRelativeScore',
        'AUM',
        'ExpenseRatio',
        'AverageVolume',
        'Volatility 1Y',
        'TrailingPE',
        'PriceToBook',
        '1M Return',
        '3M Return',
        '6M Return',
        '9M Return',
        '1Y Return',
    ]
    if col in top_etfs.columns
]
inspection_columns += [
    f'{period} Excess vs {benchmark}'
    for benchmark in benchmark_etfs
    for period in ['1M', '3M', '6M', '9M', '1Y']
    if f'{period} Excess vs {benchmark}' in top_etfs.columns
]
with open('top_qvm_etfs.md', 'w', encoding='utf-8') as f:
    f.write(top_etfs[inspection_columns].to_markdown(index=False))

gemini_columns = [
    col
    for col in [
        'Role', 'Symbol', 'Name', 'Category', 'FundFamily', 'QVMScore',
        'QualityScore', 'ValueScore', 'MomentumScore',
        'BenchmarkRelativeScore', 'AUM', 'ExpenseRatio', 'AverageVolume',
        'Volatility 1Y', 'TrailingPE', 'PriceToBook', '1M Return',
        '3M Return', '6M Return', '9M Return', '1Y Return',
    ]
    if col in top_etfs.columns
]
gemini_columns += [
    f'{period} Excess vs {benchmark}'
    for benchmark in benchmark_etfs
    for period in ['1M', '3M', '6M', '9M', '1Y']
    if f'{period} Excess vs {benchmark}' in top_etfs.columns
]
df_gemini = top_etfs[gemini_columns].copy()
dataset = df_gemini.to_string(index=False)
prompt = f'{config["prompt"].rstrip()}\n\nDATASET:\n{dataset}'

print('\n...calling Gemini...\n')
client, gemini_config = initialize_gemini_client()
final_recommendations, model_used = call_gemini(
    client,
    model_primary,
    model_fallback,
    gemini_config,
    prompt,
)
print('\nGEMINI RESPONSE:\n')
print(final_recommendations)
print(f'Generated by model: {model_used}')

html_columns = [
    col
    for col in [
        'Symbol', 'Name', 'Category', '3M Return', '1Y Return', 'QVMScore',
    ]
    if col in df_gemini.columns
]
df_html = df_gemini[html_columns].copy()
df_html = df_html.sort_values('QVMScore', ascending=False).reset_index(drop=True)
df_html = df_html.rename(
    columns={
        'Name': 'ETF Name',
        'Category': 'Sector',
        '3M Return': '3M Return (%)',
        '1Y Return': '1Y Return (%)',
        'QVMScore': 'QVM Score',
    }
)
if {'Symbol', 'ETF Name'}.issubset(df_html.columns):
    df_html['ETF Name'] = df_html.apply(
        lambda row: (
            f'<a href="https://finance.yahoo.com/quote/{row["Symbol"]}/" '
            f'target="_blank">{row["ETF Name"]}</a>'
        ),
        axis=1,
    )
if 'Symbol' in df_html.columns:
    df_html['Symbol'] = df_html['Symbol'].apply(
        lambda symbol: (
            f'<a href="https://finance.yahoo.com/quote/{symbol}/" '
            f'target="_blank">{symbol}</a>'
        )
    )
df_html_table = df_html.to_html(
    escape=False,
    index=False,
    classes='recommendations-table',
    border=0,
)
update_html_page(
    final_recommendations,
    df_html_table,
    'etf_page_template.html',
    'etf_index.html',
    model_used,
)
end_time = time.perf_counter()
print(f'Elapsed time: {round(end_time - start_time)} seconds\n')
