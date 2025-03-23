from fastapi import FastAPI, File, UploadFile
import openai
from PIL import Image
import io
import os
import base64
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    raise ValueError("OpenAI API Key not found. Check your .env file.")

def image_to_base64(image_pil):
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

@app.post("/extract_text/")
async def extract_text(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_base64 = image_to_base64(image_pil)

    # Debug: Print first 100 characters of Base64 to verify
    print(image_base64[:100])

    response = client.chat.completions.create(
        model="gpt-4-turbo-vision",  # Updated model
        messages=[
            {"role": "system", "content": "You are an OCR assistant. Extract text from images accurately."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract and return the text from this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=300
    )

    # Debug: Print full response from OpenAI
    print(response)

    extracted_text = response.choices[0].message.content if response.choices else "No text extracted."

    return {"extracted_text": extracted_text}
