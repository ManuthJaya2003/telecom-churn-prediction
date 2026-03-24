import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.preprocess import load_data, split_data
from src.features.build_features import add_engineered_features


import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "backend", "models", "churn_model.joblib")


def build_preprocessor(X: pd.DataFrame):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    return preprocessor


def main():
    df = load_data(DATA_PATH)
    df = add_engineered_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            eval_metric="logloss"
        )
    }

    best_model_name = None
    best_pipeline = None
    best_auc = 0.0

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_model_name} with ROC-AUC: {best_auc:.4f}")
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH)
    print(f"Saved model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()