import os
import subprocess
from datetime import datetime

# --- 1. Generate the HTML content ---
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
html = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>Hello from Python</title>
  </head>
  <body>
    <h1>Hello World 👋</h1>
    <p>This page was written and published automatically by Python.</p>
    <p>Last updated: {timestamp}</p>
  </body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Wrote index.html successfully.")

# --- 2. Commit and push to GitHub ---
try:
    subprocess.run(["git", "add", "index.html"], check=True)
    subprocess.run(["git", "commit", "-m", f"Update page at {timestamp}"], check=True)
    subprocess.run(["git", "push"], check=True)
    print("🚀 Successfully pushed to GitHub.")
except subprocess.CalledProcessError as e:
    print("⚠️ Git operation failed:", e)
