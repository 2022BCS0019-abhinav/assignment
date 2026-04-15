from fastapi import FastAPI
from app.models import PredictionRequest
from app.model import predict_churn
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Prediction API (MLOps) is running"}

@app.post("/predict-risk")
def predict(request: PredictionRequest):
    logging.info("Received ML pipeline request")

    risk = predict_churn(request.customer, request.tickets)

    logging.info(f"Pipeline Prediction result: {risk}")
    return {"risk": risk}