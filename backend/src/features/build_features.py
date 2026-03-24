import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in service_cols:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

    return df