import pandas as pd
import numpy as np

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/telco.csv")

print("Before Cleaning:")
print(df.info())

# -----------------------------
# Cleaning
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print("\nAfter Cleaning:")
print(df.info())

# -----------------------------
# Feature Engineering
# -----------------------------
df["ChargeDiff"] = df["MonthlyCharges"] - df["TotalCharges"] / (df["tenure"] + 1)
df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

np.random.seed(42)
df["ticket_count_30d"] = np.random.randint(0, 10, size=len(df))
df["complaint_count"] = np.random.randint(0, 5, size=len(df))

print("\nFeature Columns Added:")
print(df[["ChargeDiff", "AvgMonthlySpend", "ticket_count_30d", "complaint_count"]].head())

# -----------------------------
# Model Training (Pipeline)
# -----------------------------
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ChargeDiff",
    "AvgMonthlySpend",
    "ticket_count_30d",
    "complaint_count"
]

X = df[features]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -----------------------------
# Save Pipeline
# -----------------------------
import joblib

joblib.dump(pipeline, "model/pipeline.pkl")

print("\nPipeline saved successfully!")