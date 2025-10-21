import requests
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()  # looks for .env in current directory

# 1️⃣ Set your API key (replace with your actual key or use env var)
api_key = os.getenv("GROQ_KEY")   # or: api_key = "sk-..."

# 2️⃣ Define endpoint & payload (using OpenAI‑compatible route)
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
count = 2
payload = {
    "model": "groq/compound",   # a model available on the free tier
    "messages": [{"role": "user", "content": f"Tell me {count} fun facts."}],
    "max_tokens": 100
}

# 3️⃣ Make the request
response = requests.post(url, json=payload, headers=headers)
print(response.json())
print(response.json()["choices"][0]["message"]["content"])