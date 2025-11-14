import base64
import io
import numpy as np
import joblib
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uvicorn
from firebase_config import database

app = FastAPI(title="Soil Classification API")

# Load model and class indices
model = load_model("soil_mobilenet.h5")
class_indices = joblib.load("soil_class_indices.pkl")
classes = {v: k for k, v in class_indices.items()}
groundwater_model = load_model("groundwater_prediction_model.h5", compile=False)
rain_model = load_model("rainfall_prediction_model.h5", compile=False)

class RainfallInput(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    soil_moisture: float
    ultrasonic_distance: float

class GroundwaterRequest(BaseModel):
    temperature: float
    humidity: float
    pressure: float
    soil_moisture: float

# Request model
class ImageRequest(BaseModel):
    image: str  # Base64-encoded image string


@app.get("/")
def home():
    return {"message": "Soil Classification API is running ✅"}


@app.get("/latest-sensor-data")
def get_latest_sensor_data():
    ref = database.reference("sensor_data")
    data = ref.order_by_key().limit_to_last(1).get()

    if not data:
        return {"error": "No data found"}

    latest = list(data.values())[0]
    return {"latest_data": latest}


@app.get("/all-sensor-data")
def get_all_sensor_data():
    ref = database.reference("sensor_data")
    data = ref.get()

    if not data:
        return {"error": "No data found"}

    formatted = [{"id": key, **value} for key, value in data.items()]
    return {"data": formatted}

@app.post("/predict_base64")
def predict_base64(request: ImageRequest):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        class_id = np.argmax(preds)
        confidence = float(np.max(preds))

        return {
            "prediction": classes[class_id],
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_groundwater")
def predict_groundwater(request: GroundwaterRequest):
    try:
        # Arrange features EXACTLY as in training
        input_data = np.array([[
            request.temperature,
            request.humidity,
            request.pressure,
            request.soil_moisture
        ]])

        # Scale using saved scaler
        # scaled = scaler_g.transform(input_data)

        # Predict
        pred = groundwater_model.predict(input_data)
        groundwater_value = float(pred[0][0])

        return {
            "groundwater_level_pred": groundwater_value,
            "note": "groundwater_level unit = same as training dataset"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-rainfall")
def predict_rainfall(data: RainfallInput):
    # Convert input data to array
    input_features = np.array([[
        data.temperature,
        data.humidity,
        data.pressure,
        data.soil_moisture,
        data.ultrasonic_distance
    ]])
    
    # Scale features
    # input_scaled = scaler_r.transform(input_features)
    
    # Predict
    pred = rain_model.predict(input_features)
    
    # Return prediction as float
    return {"predicted_rainfall": float(pred[0][0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
