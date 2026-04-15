import joblib
import numpy as np

# Load pipeline
pipeline = joblib.load("model/pipeline.pkl")

def predict_churn(customer, tickets):
    # Feature engineering SAME as training

    tenure = 12  # dummy (we don’t have it from API)
    monthly = customer.monthly_charges
    total = 500  # dummy

    charge_diff = monthly - total / (tenure + 1)
    avg_spend = total / (tenure + 1)

    ticket_count = len(tickets)
    complaint_count = sum(1 for t in tickets if t.type == "complaint")

    features = np.array([[
        tenure,
        monthly,
        total,
        charge_diff,
        avg_spend,
        ticket_count,
        complaint_count
    ]])

    prediction = pipeline.predict(features)[0]

    return "HIGH" if prediction == 1 else "LOW"