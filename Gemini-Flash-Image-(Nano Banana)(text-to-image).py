import os
import time
from dotenv import load_dotenv

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# 🔑 API Key from .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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
for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("generated_image.png")
        print("✅ Image saved as generated_image.png")