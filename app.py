"""
Phishing Website Classification - Streamlit App
Done by: Aravinth
BITS ID: 2025NS04081
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAVED_MODELS_DIR = PROJECT_ROOT / "model" / "saved_models"
METRICS_CSV_PATH = PROJECT_ROOT / "model" / "metrics_results.csv"
SAMPLE_TEST_CSV_PATH = PROJECT_ROOT / "test_data.csv"

ID_COLUMN = "url"
TARGET_COLUMN = "Result"
LABEL_MAP = {0: "legitimate", 1: "phishing"}
ACCENT = "#7C3AED"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

CUSTOM_CSS = f"""
<style>
.hero {{
    padding: 0.85rem 1.5rem;
    border-radius: 10px;
    background: linear-gradient(120deg, #4338CA 0%, {ACCENT} 55%, #DB2777 100%);
    color: white;
    margin-bottom: 1rem;
}}
.hero h1 {{ margin: 0; font-size: 1.3rem; }}
.hero p {{ margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.85rem; }}
div[data-testid="stMetric"] {{
    background: rgba(124, 58, 237, 0.08);
    border: 1px solid rgba(124, 58, 237, 0.25);
    border-radius: 10px;
    padding: 0.6rem 0.8rem 0.3rem 0.8rem;
}}
</style>
"""


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(SAVED_MODELS_DIR / "scaler.joblib")
    feature_cols = joblib.load(SAVED_MODELS_DIR / "feature_columns.joblib")
    models = {name: joblib.load(SAVED_MODELS_DIR / f) for name, f in MODEL_FILES.items()}
    return scaler, feature_cols, models


@st.cache_data
def load_precomputed_metrics():
    return pd.read_csv(METRICS_CSV_PATH)


def get_best_overall_model(df: pd.DataFrame) -> pd.Series:
    # just averaging all 6 metrics to get one overall score, picked the row with highest
    scored = df.copy()
    scored["Composite Score"] = df.drop(columns="Model").mean(axis=1)
    return scored.sort_values("Composite Score", ascending=False).iloc[0]


def style_metrics_table(df: pd.DataFrame):
    best_per_col = df.drop(columns="Model").idxmax()
    def highlight_best(col):
        return ["background-color: rgba(124, 58, 237, 0.25); font-weight: 600"
                if i == best_per_col[col.name] else "" for i in col.index]
    return df.set_index("Model").style.format("{:.4f}").apply(highlight_best, axis=0)


def run_predictions(df: pd.DataFrame, model, scaler, feature_cols):
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing {len(missing)} required feature column(s), e.g. {missing[:5]}")
        return None

    X_scaled = scaler.transform(df[feature_cols])
    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)[:, 1]

    results = pd.DataFrame()
    if ID_COLUMN in df.columns:
        results[ID_COLUMN] = df[ID_COLUMN]
    results["predicted_status"] = [LABEL_MAP[p] for p in preds]
    results["phishing_probability"] = probs.round(4)
    if TARGET_COLUMN in df.columns:
        results["actual_status"] = df[TARGET_COLUMN].map(LABEL_MAP)

    return results, preds, probs


def render_model_agreement(df: pd.DataFrame, models: dict, scaler, feature_cols):
    # added this to see if models disagree on same rows, helps compare them better
    X_scaled = scaler.transform(df[feature_cols])

    vote_table = pd.DataFrame(index=df.index)
    if ID_COLUMN in df.columns:
        vote_table[ID_COLUMN] = df[ID_COLUMN]

    per_model_rate = {}
    for name, model in models.items():
        preds = model.predict(X_scaled)
        vote_table[name] = [LABEL_MAP[p] for p in preds]
        per_model_rate[name] = preds.mean()

    vote_cols = list(models.keys())
    phishing_votes = (vote_table[vote_cols] == "phishing").sum(axis=1)
    vote_table["consensus"] = np.where(phishing_votes > len(models) / 2, "phishing", "legitimate")
    vote_table["agreement"] = np.maximum(phishing_votes, len(models) - phishing_votes).astype(str) + f"/{len(models)}"

    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown("**Per-row vote breakdown (all 5 models)**")
        st.dataframe(vote_table, width="stretch", height=320)

    with col_right:
        st.markdown("**Predicted phishing rate per model**")
        rate_df = pd.Series(per_model_rate, name="phishing_rate").sort_values()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(rate_df)))
        ax.barh(rate_df.index, rate_df.values, color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction predicted phishing")
        for i, v in enumerate(rate_df.values):
            ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

        full_agreement_pct = (phishing_votes.isin([0, len(models)])).mean()
        st.metric("Rows where all 5 models agree", f"{full_agreement_pct:.1%}")


def render_feature_importance(model_name: str, model, feature_cols: list[str]):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = "Decision Tree feature importances (Gini-based)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = "Logistic Regression — |coefficient| (standardized features)"
    else:
        st.info(
            f"{model_name} doesn't expose a simple feature-importance score "
            "(kNN is distance-based, Naive Bayes is probabilistic, and our SVM uses an RBF kernel). "
            "Switch to Logistic Regression or Decision Tree above to see this chart."
        )
        return

    top_idx = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(feature_cols)[top_idx], importances[top_idx], color=ACCENT)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Phishing Website Classifier", page_icon=":shield:", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>Phishing Website Classification</h1>
            <p>M.Tech (DSE) · Machine Learning (DSEZG565) · Assignment 2 — Aravinth (2025NS04081)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scaler, feature_cols, models = load_artifacts()

    st.sidebar.header("Settings")
    model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
    use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead", value=uploaded_file is None)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_sample and SAMPLE_TEST_CSV_PATH.exists():
        df = pd.read_csv(SAMPLE_TEST_CSV_PATH)
    else:
        st.info("Upload a CSV file to get predictions, or check 'Use bundled test_data.csv'.")
        df = None

    tab_compare, tab_predict, tab_agreement, tab_importance = st.tabs(
        ["Model Comparison", "Predictions", "Model Agreement", "Feature Importance"]
    )

    with tab_compare:
        precomputed_metrics = load_precomputed_metrics()
        best_model = get_best_overall_model(precomputed_metrics)

        st.subheader("Best Performing Model (Held-Out Test Set)")
        kpi_cols = st.columns(4)
        kpi_cols[0].metric("Model", best_model["Model"])
        kpi_cols[1].metric("Composite Score", f"{best_model['Composite Score']:.4f}")
        kpi_cols[2].metric("Accuracy", f"{best_model['Accuracy']:.4f}")
        kpi_cols[3].metric("MCC", f"{best_model['MCC']:.4f}")
        st.caption("Composite Score = average of Accuracy, AUC, Precision, Recall, F1, and MCC across all 6 metrics.")

        st.subheader("All 5 Models — Metrics on Held-Out Test Set")
        st.dataframe(style_metrics_table(precomputed_metrics), width="stretch")
        st.caption("Highlighted cell = best score for that metric across all 5 models.")

    if df is None:
        return

    output = run_predictions(df, model, scaler, feature_cols)
    if output is None:
        return
    results, preds, probs = output

    with tab_predict:
        st.subheader(f"Predictions — {model_name}")
        st.dataframe(results, width="stretch")

        if TARGET_COLUMN in df.columns:
            y_true = df[TARGET_COLUMN]

            st.subheader("Evaluation Metrics on Uploaded Data")
            metrics = {
                "Accuracy": accuracy_score(y_true, preds),
                "AUC": roc_auc_score(y_true, probs),
                "Precision": precision_score(y_true, preds),
                "Recall": recall_score(y_true, preds),
                "F1": f1_score(y_true, preds),
                "MCC": matthews_corrcoef(y_true, preds),
            }
            cols = st.columns(len(metrics))
            for col, (name, value) in zip(cols, metrics.items()):
                col.metric(name, f"{value:.4f}")

            col_left, col_right = st.columns(2)
            with col_left:
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                cm = confusion_matrix(y_true, preds)
                ConfusionMatrixDisplay(cm, display_labels=["legitimate", "phishing"]).plot(ax=ax, cmap="Purples")
                st.pyplot(fig)

            with col_right:
                st.subheader("Classification Report")
                report = classification_report(
                    y_true, preds, target_names=["legitimate", "phishing"], output_dict=True
                )
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.3f}"), width="stretch")
        else:
            st.info("Uploaded CSV has no ground-truth 'Result' column, so metrics/confusion matrix are not shown.")

    with tab_agreement:
        st.subheader("Do all 5 models agree on the same rows?")
        st.caption(
            "Instead of judging one model at a time, this runs every model on your uploaded data "
            "simultaneously and highlights consensus vs. disagreement — useful for spotting which "
            "URLs are genuinely ambiguous."
        )
        render_model_agreement(df, models, scaler, feature_cols)

    with tab_importance:
        st.subheader(f"What drives {model_name}'s decisions?")
        render_feature_importance(model_name, model, feature_cols)


if __name__ == "__main__":
    main()

