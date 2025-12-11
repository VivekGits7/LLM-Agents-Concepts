import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

# 🔑 API Key from .env
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")
client = genai.Client(api_key=api_key)

prompt = "Create a image of a Code Checker AI Application logo with a futuristic design, vibrant colors, and sleek typography as poster."

response = client.models.generate_images(
    model='imagen-4.0-generate-001',
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images=1,
    )
)

if response.generated_images is not None:
    for generated_image in response.generated_images:
        if generated_image.image is not None:
            generated_image.image.show()
else:
    print("No images were generated")