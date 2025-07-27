# FastAPI application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import mlflow.pyfunc
import logging
import json
from datetime import datetime
import pandas as pd
import joblib
import uuid
import os
from mlflow.tracking import MlflowClient

# Configure logging
LOG_FILE = "logs/api_predictions.log"
os.makedirs("logs", exist_ok=True) # Ensure logs directory exists
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


# Global variables for model and scaler
ml_model = None
scaler = None
model_name = "CaliforniaHousingRegressor" # As registered in MLflow
model_version_str = "unknown"

# --- FastAPI App with Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, scaler, model_version_str
    # Start Prometheus metrics server on a different port (e.g., 8001)
    # (See Part 5 for Prometheus setup)
    # metrics_port = 8001
    # threading.Thread(target=start_prometheus_server, args=(metrics_port,), daemon=True).start()

    try:
        #To run fastapi locally set tracking URI to localhost
        # mlflow.set_tracking_uri("http://127.0.0.1:5000")

        #To run fastapi in docker set tracking URI to mlflow
        tracking_uri = "http://mlflow-server:5000"
        mlflow.set_tracking_uri(tracking_uri)

        logger.info("Attempting to load model from MLflow registry...")

        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        if not model_versions:
            logger.error(f"No versions found for model '{model_name}' in MLflow registry.")
            yield
            return

        # Pick the latest version
        latest_version = max(model_versions, key=lambda m: int(m.version))
        model_uri = f"models:/{model_name}/{latest_version.version}"  #
        model_version_str = f"{model_name}_v{latest_version.version}"
        ml_model = mlflow.pyfunc.load_model(model_uri)

        # To check which model got loaded
        impl = ml_model._model_impl
        try:
            raw_model = impl.get_raw_model()
            logger.info("Actual model class: %s", type(raw_model).__name__)
        except Exception as e:
            logger.warning("Could not determine model class: %s", str(e))

        logger.info(f"Successfully loaded model from {model_uri}")

        # Load the scaler
        scaler_path = "scaler.pkl" # Make sure this file is available in the Docker image
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.error(f"Scaler file not found at {scaler_path}. Data will not be scaled.")
            scaler = None

    except Exception as e:
        logger.exception(f"Error loading model or scaler: {e}")
        ml_model = None

    yield
    # Optional: Add cleanup here

app = FastAPI(title="California Housing Price Predictor API", lifespan=lifespan)

# --- Input and Output Models ---

# Pydantic model for request validation (California Housing features)
class PredictionRequest(BaseModel):
    MedInc: float = Field(..., description="Median income in block group", example=3.87)
    HouseAge: float = Field(..., description="Median house age in block group", example=25.0)
    AveRooms: float = Field(..., description="Average number of rooms per household", example=5.0)
    AveBedrms: float = Field(..., description="Average number of bedrooms per household", example=1.0)
    Population: float = Field(..., description="Block group population", example=1200.0)
    AveOccup: float = Field(..., description="Average number of household members", example=2.5)
    Latitude: float = Field(..., description="Block group latitude", example=34.0)
    Longitude: float = Field(..., description="Block group longitude", example=-118.0)

class PredictionResponse(BaseModel):
    predicted_value: float
    timestamp: str
    model_version: str

# --- Routes ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the California Housing Price Prediction API. Use /predict for predictions."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded or failed to load during startup.")
    if scaler is None:
        raise HTTPException(status_code=500, detail="Scaler not loaded. Cannot preprocess input.")

    # Convert Pydantic model to DataFrame, preserving feature order
    try:
        input_df = pd.DataFrame([request.dict()])

        # Apply the same scaling as during training
        # Ensure the order of columns matches the training data features
        # Get feature names from the scaler if possible or hardcode based on original data
        feature_order = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
        input_df = input_df[feature_order]

        scaled_input = scaler.transform(input_df)
        scaled_df = pd.DataFrame(scaled_input, columns=feature_order)

        prediction = ml_model.predict(scaled_df)[0]
        timestamp = datetime.now().isoformat()

        # Log the request and response
        log_data = {
            "timestamp": timestamp,
            "request_id": str(uuid.uuid4()),
            "input_data": request.dict(), # Log original request for readability
            "scaled_data": scaled_df.to_dict(orient="records")[0],
            "prediction": float(prediction),
            "model_version": model_version_str
        }
        logger.info(json.dumps(log_data))

        return PredictionResponse(
            predicted_value=float(prediction),
            timestamp=timestamp,
            model_version=model_version_str
        )
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed due to internal error.")

# --- Run Server ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
