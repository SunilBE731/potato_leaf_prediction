from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "potato_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight", "not_potato"]

TREATMENTS = {
    "Early_Blight": "• Remove infected leaves\n• Use copper-based fungicides\n• Maintain spacing for airflow\n• Avoid overhead irrigation",
    "Late_Blight": "• Spray Mancozeb or Chlorothalonil\n• Remove and destroy infected plants\n• Improve drainage in soil\n• Do not wet foliage",
    "Healthy": "Your plant is healthy! Continue regular watering and pest checks.",
}

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img_batch = preprocess_image(image_bytes)

        predictions = model.predict(img_batch)[0]
        class_index = np.argmax(predictions)
        class_name = CLASS_NAMES[class_index]
        confidence = float(predictions[class_index])

        if class_name == "not_potato":
            return {
                "status": "error",
                "message": "This is NOT a potato leaf. Please upload a potato leaf image.",
                "confidence": confidence
            }

        return {
            "status": "success",
            "class": class_name,
            "confidence": confidence,
            "treatment": TREATMENTS[class_name]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
