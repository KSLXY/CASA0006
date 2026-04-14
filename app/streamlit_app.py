from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "metrics.json"
SAMPLE_PATH = ROOT_DIR / "data" / "sample" / "merged_sample.csv"


@st.cache_resource
def load_model_payload():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_sample_data():
    if not SAMPLE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_PATH)


def main() -> None:
    st.set_page_config(
        page_title="London Road Severity Explorer",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("London Road Collision Severity Project")
    st.caption("Portfolio project upgraded from CASA0006 coursework.")

    payload = load_model_payload()
    metrics = load_metrics()
    sample_df = load_sample_data()

    tabs = st.tabs(["Overview", "Data", "Model Performance", "Predict"])

    with tabs[0]:
        st.subheader("Project Summary")
        st.markdown(
            """
            - **Goal**: Predict road collision severity (1/2/3) from weather and collision context.
            - **Data**: London weather + UK DfT collision records.
            - **Modeling**: Logistic Regression, Random Forest, HistGradientBoosting with model selection by Macro F1.
            - **Deployment**: Streamlit Community Cloud.
            """
        )

    with tabs[1]:
        st.subheader("Sample Dataset Preview")
        if sample_df.empty:
            st.warning("Sample data not found.")
        else:
            st.dataframe(sample_df.head(20), use_container_width=True)
            st.write(f"Rows: {len(sample_df)}")

    with tabs[2]:
        st.subheader("Training Metrics")
        if metrics is None:
            st.info("No metrics file found. Run training first.")
        else:
            selected = metrics.get("selected_model", "N/A")
            st.write(f"Selected model: **{selected}**")
            selected_metrics = metrics.get("selected_model_metrics", {})
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{selected_metrics.get('accuracy', 0):.3f}")
            c2.metric("Macro F1", f"{selected_metrics.get('f1_macro', 0):.3f}")
            all_metrics = metrics.get("all_model_metrics", [])
            if all_metrics:
                table = pd.DataFrame(
                    [{"model_name": m["model_name"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]} for m in all_metrics]
                )
                st.dataframe(table, use_container_width=True)

    with tabs[3]:
        st.subheader("Single Prediction")
        if payload is None:
            st.info("No trained model found. Run `python -m src.train --config configs/default.yaml` first.")
        else:
            feature_columns = payload["feature_columns"]
            defaults = payload.get("feature_defaults", {})
            class_labels = payload.get("class_labels", ["Slight", "Serious", "Fatal"])
            inputs = {}
            for col in feature_columns:
                default = float(defaults.get(col, 0.0))
                inputs[col] = st.number_input(col, value=default, format="%.4f")

            if st.button("Predict severity"):
                X = pd.DataFrame([inputs], columns=feature_columns)
                pipeline = payload["pipeline"]
                pred_idx = int(pipeline.predict(X)[0])
                probs = pipeline.predict_proba(X)[0]
                st.success(f"Predicted class: {class_labels[pred_idx]} (severity={pred_idx + 1})")
                prob_table = pd.DataFrame(
                    {"class_label": class_labels, "probability": probs}
                ).sort_values("probability", ascending=False)
                st.dataframe(prob_table, use_container_width=True)


if __name__ == "__main__":
    main()

