import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.joblib"


def predict_single(input_data: dict):
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([input_data])

    churn_probability = model.predict_proba(df)[0][1]
    prediction = int(churn_probability >= 0.5)

    return {
        "churn_probability": float(round(churn_probability, 4)),
        "prediction": "Yes" if prediction == 1 else "No",
        "risk_level": (
            "High" if churn_probability >= 0.7 else
            "Medium" if churn_probability >= 0.4 else
            "Low"
        )
    }