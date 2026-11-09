import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow import keras

app = FastAPI(title="VisionGuard Model API", version="1.0.0")

MODEL_PATH = "models/champion.keras"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        print(f"Loaded champion model from '{MODEL_PATH}'")
    else:
        print(f"Warning: '{MODEL_PATH}' not found at {os.path.abspath(MODEL_PATH)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)
    
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    
    return {
        "filename": file.filename,
        "predicted_class_index": pred_idx,
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in preds]
    }

