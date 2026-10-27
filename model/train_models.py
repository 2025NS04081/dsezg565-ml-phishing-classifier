"""
Model Training and Evaluation - Phishing Website Classification
Done by: Aravinth
BITS ID: 2025NS04081
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV_PATH = PROJECT_ROOT / "model" / "train_data.csv"
TEST_CSV_PATH = PROJECT_ROOT / "test_data.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"
METRICS_CSV_PATH = PROJECT_ROOT / "model" / "metrics_results.csv"

ID_COLUMN = "url"
TARGET_COLUMN = "Result"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=RANDOM_STATE),
}


def load_splits():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    feature_cols = [c for c in train_df.columns if c not in (ID_COLUMN, TARGET_COLUMN)]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test, feature_cols


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    X_train, y_train, X_test, y_test, feature_cols = load_splits()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SAVED_MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_cols, SAVED_MODELS_DIR / "feature_columns.joblib")

    results = []
    for name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics["Model"] = name
        results.append(metrics)

        file_name = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, SAVED_MODELS_DIR / file_name)

        print(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "Model"))

    results_df = pd.DataFrame(results)[["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    results_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\nSaved metrics comparison to {METRICS_CSV_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
