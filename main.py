from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
import openai
from PIL import Image
import io
import os
import base64
from dotenv import load_dotenv

app = FastAPI()

# Allow all origins, methods, and headers (Modify this for security if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in ancient scripts. Convert handwritten Malayalam or Tamil text into Grantha script and provide its basic meaning."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transliterate this text into Grantha script and provide its basic meaning."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        max_tokens=100  # Increase tokens to accommodate meaning
    )

    grantha_text = response.choices[0].message.content.strip()
    print(grantha_text)
    print(response)

    return {"grantha_text": grantha_text}