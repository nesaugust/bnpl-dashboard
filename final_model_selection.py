"""
Clean EDA and Machine Learning script for BNPL credit risk prediction.

This script:
1. Loads BNPL_dataset.csv
2. Performs basic preprocessing
3. Encodes categorical variables
4. Trains multiple classification models
5. Selects the best model based on ROC-AUC
6. Saves the best model and feature columns for dashboard use
"""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


DATA_PATH = "BNPL_dataset.csv"
TARGET_COL = "default_flag"
MODEL_PATH = "best_bnpl_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.json"
RESULTS_PATH = "model_results.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the dataset."""
    df = pd.read_csv(path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for machine learning."""
    df = df.copy()

    # Convert transaction_date into useful date features
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df["transaction_month"] = df["transaction_date"].dt.month
        df["transaction_year"] = df["transaction_date"].dt.year
        df["transaction_day"] = df["transaction_date"].dt.day
        df = df.drop(columns=["transaction_date"])

    # One-hot encode categorical columns
    categorical_cols = [
        "employment_type",
        "product_category",
        "location",
        "customer_segment",
    ]

    existing_categorical_cols = [
        col for col in categorical_cols if col in df.columns
    ]

    df = pd.get_dummies(
        df,
        columns=existing_categorical_cols,
        drop_first=True,
    )

    # Convert boolean columns to integers
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Drop ID columns that should not be used for prediction
    df = df.drop(columns=["user_id"], errors="ignore")

    return df


def split_features_target(df: pd.DataFrame):
    """Split dataframe into features and target."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' was not found in the dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y


def build_models() -> dict:
    """Create machine learning models."""
    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
        ),
        "Neural Network": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=300,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


def get_prediction_probability(model, X_test):
    """Return positive-class probabilities where available."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    raise ValueError("Model does not support probability or decision scores.")


def train_and_evaluate_models(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train models and compare their performance."""
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = get_prediction_probability(model, X_test)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_proba),
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="ROC-AUC", ascending=False)

    return results_df


def print_best_model_analysis(best_model, best_model_name, X_test, y_test):
    """Print classification report and confusion matrix for the best model."""
    y_pred_best = best_model.predict(X_test)
    y_proba_best = get_prediction_probability(best_model, X_test)

    print("\nBest Model Analysis")
    print("-" * 50)
    print(f"Best Model: {best_model_name}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_proba_best):.4f}")


def save_artifacts(best_model, feature_columns, results_df):
    """Save model, feature columns, and model comparison results."""
    joblib.dump(best_model, MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as file:
        json.dump(list(feature_columns), file, indent=4)

    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nFiles saved successfully:")
    print(f"- {MODEL_PATH}")
    print(f"- {FEATURE_COLUMNS_PATH}")
    print(f"- {RESULTS_PATH}")


def main():
    """Run the full EDA and machine learning pipeline."""
    df = load_data(DATA_PATH)

    print("Dataset Preview:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDataset Info:")
    print(df.dtypes)

    print("\nDescriptive Statistics:")
    print(df.describe())

    df_ml = preprocess_data(df)

    X, y = split_features_target(df_ml)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = build_models()

    results_df = train_and_evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nModel Comparison:")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = models[best_model_name]

    print_best_model_analysis(
        best_model,
        best_model_name,
        X_test,
        y_test,
    )

    save_artifacts(
        best_model,
        X.columns,
        results_df,
    )


if __name__ == "__main__":
    main()
