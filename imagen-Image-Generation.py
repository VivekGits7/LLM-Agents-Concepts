import os

from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# 🔑 API Key from .env
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = "Create a image of a Code Checker AI Application logo with a futuristic design, vibrant colors, and sleek typography as poster."

response = client.models.generate_images(
    model='imagen-4.0-generate-001',
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images= 1,
    )
)
for generated_image in response.generated_images:
  generated_image.image.show()