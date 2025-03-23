from fastapi import FastAPI, File, UploadFile
import openai
import io
import os
import base64
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Get OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API Key not found. Set OPENAI_API_KEY in Render environment variables.")

@app.post("/extract_text/")
async def extract_text(image: UploadFile = File(...)):
    # Read and process image
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert image to base64 (OpenAI API expects raw bytes)
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # OpenAI GPT-4 Vision API call
    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "system", "content": "You are an OCR assistant. Extract text from images accurately."},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract and return the text from this image."},
                {"type": "image", "image": image_base64}
            ]}
        ],
        max_tokens=300
    )

    # Extract text from response
    extracted_text = response["choices"][0]["message"]["content"]
    
    return {"extracted_text": extracted_text}
