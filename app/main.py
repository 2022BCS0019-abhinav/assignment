import logging
from fastapi import FastAPI
from app.models import PredictionRequest
from app.rules import predict_risk

logging.basicConfig(level=logging.INFO)

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict-risk")
def predict(request: PredictionRequest):
    logging.info("Received prediction request")
    risk = predict_risk(request.customer, request.tickets)
    logging.info(f"Prediction result: {risk}")
    return {"risk": risk}