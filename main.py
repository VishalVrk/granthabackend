from fastapi import FastAPI, File, UploadFile
import openai
from PIL import Image
import io
import os
import base64
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("OpenAI API Key not found. Check your .env file.")

def image_to_base64(image_pil):
    """ Convert PIL Image to Base64 """
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/extract_text/")
async def extract_text(image: UploadFile = File(...)):
    # Read image
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert image to Base64
    image_base64 = image_to_base64(image_pil)

    # OpenAI GPT-4 Turbo Vision API call
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are an OCR assistant. Extract text from images accurately."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract and return the text from this image."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                ]
            }
        ],
        max_tokens=300
    )

    # Extract text from API response
    extracted_text = response.choices[0].message.content

    return {"extracted_text": extracted_text}