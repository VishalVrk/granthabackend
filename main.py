from fastapi import FastAPI, File, UploadFile
import openai
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("OpenAI API Key not found. Check your .env file.")

@app.post("/extract_text/")
async def extract_text(image: UploadFile = File(...)):
    # Read the image
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Convert image to RGB and save temporarily
    image_pil = image_pil.convert("RGB")
    image_pil.save("temp.jpg", "JPEG")

    # OpenAI GPT-4 Vision API call (Updated for v1.0.0+)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are an OCR assistant. Extract text from images accurately."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract and return the text from this image."},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64," + image_to_base64("temp.jpg")}
                ]
            }
        ],
        max_tokens=300
    )

    # Extract text from API response
    extracted_text = response.choices[0].message.content

    return {"extracted_text": extracted_text}

def image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
