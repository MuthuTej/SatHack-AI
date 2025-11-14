import base64
import io
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import uvicorn
from firebase_config import database
import pickle


app = FastAPI(title="Soil Classification API")

# Load model and class indices
model = load_model("soil_mobilenet.h5")
class_indices = joblib.load("soil_class_indices.pkl")
classes = {v: k for k, v in class_indices.items()}

model_water = load_model('water_model.h5', compile=False)
model_crop = load_model('crop_model.h5', compile=False)
model_time = load_model('time_model.h5', compile=False)


with open('crop_encoders.pkl', 'rb') as f:
    crop_encoders = pickle.load(f)
with open('water_encoders.pkl', 'rb') as f:
    water_encoders = pickle.load(f)
with open('time_encoders.pkl', 'rb') as f:
    time_encoders = pickle.load(f)
    
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
    
@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input dictionary to DataFrame
        df_input = pd.DataFrame([data])

        # Transform categorical features safely
        try:
            X_cat = time_encoders['encoder_cat'].transform(
                df_input[['soil_type', 'month']].astype(str)
            )
        except ValueError as e:
            return {"error": f"Unknown category encountered: {str(e)}"}

        # Transform numeric features
        X_num = time_encoders['scaler'].transform(
            df_input[['temperature', 'pressure', 'groundwater_level', 'rainfall']]
        )

        # Combine numeric and categorical features
        X_input = np.concatenate([X_num, X_cat], axis=1)

        # Crop prediction
        pred_crop = model_crop.predict(X_input)
        crop_label = crop_encoders['encoder_crop'].inverse_transform(pred_crop)[0][0]

        # Water prediction
        pred_water = model_water.predict(X_input)[0][0]

        # Irrigation time prediction
        pred_time = model_time.predict(X_input)
        time_label = time_encoders['encoder_time'].inverse_transform(pred_time)[0][0]

        return {
            "crop": crop_label,
            "water_needed": float(pred_water),
            "irrigation_time": time_label
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
