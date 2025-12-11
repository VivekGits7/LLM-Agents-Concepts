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

if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print(part.text)
        elif part.inline_data is not None and part.inline_data.data is not None:
            edited_image = Image.open(BytesIO(part.inline_data.data))
            edited_image.save("generated_image.png")
else:
    print("No valid response received from the model")
