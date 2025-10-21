etf_text="hehehe"
prompt = f"""
I have a table of ETFs below:

{etf_text}

Please do the following:
1. Group the ETFs by theme (e.g., technology, healthcare, finance).
2. For each theme, select the 5 best-performing ETFs based on 52-week change %.
3. Present the results nicely in a readable table format with columns: Theme, Rank, Name, Symbol, 52 WkChange %, 3 MonthReturn, Price, 50 DayAverage, 200 DayAverage.
"""
print(prompt)