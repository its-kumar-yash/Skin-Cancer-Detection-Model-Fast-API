import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from io import BytesIO
from PIL import Image
from typing import Tuple
import tensorflow as tf

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("model/1")
CLASS_NAMES = ['basal cell carcinoma', 'melanoma', 'squamous cell carcinoma']

@app.get("/ping")
async def ping():
    return "Hello World!"

def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((180, 180), resample=Image.BICUBIC)
    image = np.array(img_resized)
    return image, img_resized.size

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image, img_size = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            "class": predicted_class,
            "confidence": float(confidence),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Skin Disease Detection API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)