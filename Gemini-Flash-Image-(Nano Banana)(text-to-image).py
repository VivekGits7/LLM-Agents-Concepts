import os
import time
from io import BytesIO

from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables from .env file
load_dotenv()

# 🔑 API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
client = genai.Client(api_key=api_key)

prompt = (
    """
    Energetic poster showing creative professionals (writer, student, designer) connected to a central AI hologram, vibrant startup-style design, orange and teal gradient, inspiring tagline space.
    """
)

print("🖼️ Generating image with Gemini 2.5 Flash Image model...")
time.sleep(2)
response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt],
)

print("✅ Image generation complete. Processing response...")
time.sleep(1)
if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None and part.inline_data.data is not None:
            generated_image = Image.open(BytesIO(part.inline_data.data))
            generated_image.save("generated_image.png")
            print("✅ Image saved as generated_image.png")
else:
    print("No valid response received from the model")