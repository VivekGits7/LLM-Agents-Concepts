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

print("🖼️ Loading existing image for editing...")
time.sleep(1)
prompt = (
    "Generate a professional LinkedIn profile photo from my image. Use soft natural lighting and a smooth gradient background in light gray or white. Enhance clarity, contrast, and detail while keeping it natural. The result should look like a high-quality corporate headshot suitable for LinkedIn."
)

print("🖼️ Editing image with Gemini 2.5 Flash Image model...")
image = Image.open("/Video-and-Image-Generation/VeoGenAIModels/ProfilePhoto.jpg")

response = client.models.generate_content(
    model="gemini-2.5-flash-image",
    contents=[prompt, image],
)

for part in response.candidates[0].content.parts:
    if part.text is not None:
        print(part.text)
    elif part.inline_data is not None:
        image = Image.open(BytesIO(part.inline_data.data))
        image.save("generated_image.png")
