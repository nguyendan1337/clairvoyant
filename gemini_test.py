import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file and get GEMINI_KEY
load_dotenv()
api_key = os.getenv("GEMINI_KEY")
# Initialize the GenAI client
client = genai.Client(api_key=api_key)
# Enable Google Search tool
grounding_tool = types.Tool(google_search=types.GoogleSearch())
config = types.GenerateContentConfig(tools=[grounding_tool])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tell me about today's stock news.",
    config=config
)

print(response.text)