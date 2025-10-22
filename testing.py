# generate_page.py

# Your website’s HTML content
html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>Hello from Python</title>
  </head>
  <body>
    <h1>Hello World 👋</h1>
    <p>This page was written by a Python script.</p>
  </body>
</html>
"""

# Write it to index.html (the main GitHub Pages file)
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("✅ Wrote index.html successfully.")
