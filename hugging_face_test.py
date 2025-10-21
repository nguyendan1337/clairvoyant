import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env file
load_dotenv()  # looks for .env in current directory

client = InferenceClient(
    provider="fal-ai",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)

# Save to file
image.save("astronaut_horse.png")

# Display
image.show()