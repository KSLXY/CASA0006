from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "artifacts" / "model.joblib"
METRICS_PATH = ROOT_DIR / "artifacts" / "metrics.json"
METRICS_CV_PATH = ROOT_DIR / "artifacts" / "metrics_cv.json"
SAMPLE_PATH = ROOT_DIR / "data" / "sample" / "merged_sample.csv"
MODEL_COMPARE_PATH = ROOT_DIR / "artifacts" / "model_compare.csv"
ERROR_CASES_PATH = ROOT_DIR / "artifacts" / "error_cases.csv"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "artifacts" / "feature_importance.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"


@st.cache_resource
def load_model_payload():
    if not MODEL_PATH.exists():
        return None, None
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_metrics_cv():
    if not METRICS_CV_PATH.exists():
        return None
    return json.loads(METRICS_CV_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_sample_data():
    if not SAMPLE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data
def load_model_comparison():
    if MODEL_COMPARE_PATH.exists():
        return pd.read_csv(MODEL_COMPARE_PATH)
    return pd.DataFrame()


@st.cache_data
def load_error_cases():
    if ERROR_CASES_PATH.exists():
        return pd.read_csv(ERROR_CASES_PATH)
    return pd.DataFrame()


@st.cache_data
def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return pd.DataFrame()


def _safe_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="London Road Severity Explorer",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("London Road Collision Severity Project")
    st.caption("From coursework to portfolio: data quality, model comparison, and deployment.")

    payload, model_load_error = load_model_payload()
    metrics = load_metrics()
    metrics_cv = load_metrics_cv()
    sample_df = load_sample_data()
    model_compare_df = load_model_comparison()
    error_cases_df = load_error_cases()
    feature_importance_df = load_feature_importance()

    if model_load_error:
        st.error(
            "Model artifact could not be loaded. "
            "This is often a Python-version mismatch issue on cloud. "
            f"Details: {model_load_error}"
        )

    selected_model = metrics.get("selected_model", "N/A") if metrics else "N/A"
    rows_total = int(metrics.get("rows_total", 0)) if metrics else 0
    rows_used = int(metrics.get("rows_used_for_training", 0)) if metrics else 0
    rows_removed = int(metrics.get("rows_removed_invalid_target", 0)) if metrics else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows total", rows_total)
    k2.metric("Rows used", rows_used)
    k3.metric("Invalid target removed", rows_removed)
    k4.metric("Best model", selected_model)

    tabs = st.tabs(
        [
            "Overview",
            "Data",
            "Data Quality",
            "Model Performance",
            "Reliability",
            "Error Analysis",
            "Predict",
            "Limitations & Next Step",
        ]
    )

    with tabs[0]:
        st.subheader("Problem Story")
        st.markdown(
            """
            - **City context**: Road-collision severity is a key public-safety signal.
            - **Data challenge**: Real-world labels include invalid values (e.g. `-10`), so cleaning decisions matter.
            - **Modeling decision**: We compare baseline and tree models, and select by **Macro F1** to keep minority classes visible.
            - **Deployment outcome**: This demo is online and reproducible from GitHub + Streamlit.
            """
        )
        if metrics:
            st.info(f"Performance note: **{metrics.get('performance_note', 'sample demo performance')}**")

    with tabs[1]:
        st.subheader("Sample Dataset Preview")
        if sample_df.empty:
            st.warning("Sample data not found.")
        else:
            st.dataframe(sample_df.head(20), use_container_width=True)
            st.write(f"Rows: {len(sample_df)}")
        st.markdown(
            """
            **Data sources**
            - London weather data (Kaggle public dataset)
            - UK DfT road collision records
            """
        )

    with tabs[2]:
        st.subheader("Data Quality")
        if metrics is None:
            st.info("No metrics file found. Run training first.")
        else:
            st.metric("Removed invalid target labels", int(metrics.get("rows_removed_invalid_target", 0)))
            missing_rate = metrics.get("missing_rate_by_feature", {})
            if missing_rate:
                df_missing = (
                    pd.DataFrame(
                        [{"feature": k, "missing_rate": v} for k, v in missing_rate.items()]
                    )
                    .sort_values("missing_rate", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(df_missing, use_container_width=True)
                st.caption("Missing-rate is computed before imputation and used as data quality evidence.")
            else:
                st.write("Missing-rate details unavailable.")

    with tabs[3]:
        st.subheader("Model Comparison & Evidence")
        if metrics is None:
            st.info("No metrics file found. Run training first.")
        else:
            selected = metrics.get("selected_model", "N/A")
            st.write(f"Selected model: **{selected}**")
            selected_metrics = metrics.get("selected_model_metrics", {})
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{selected_metrics.get('accuracy', 0):.3f}")
            c2.metric("Macro F1", f"{selected_metrics.get('f1_macro', 0):.3f}")
            if model_compare_df.empty:
                all_metrics = metrics.get("all_model_metrics", [])
                if all_metrics:
                    model_compare_df = pd.DataFrame(
                        [{"model_name": m["model_name"], "accuracy": m["accuracy"], "f1_macro": m["f1_macro"]} for m in all_metrics]
                    )
            if not model_compare_df.empty:
                st.dataframe(model_compare_df, use_container_width=True)
                st.bar_chart(
                    model_compare_df.set_index("model_name")[["accuracy", "f1_macro"]],
                    use_container_width=True,
                )
            _safe_image(FIGURES_DIR / "model_comparison.png", "Model Comparison Figure")
            _safe_image(FIGURES_DIR / "confusion_matrix.png", "Confusion Matrix (Selected Model)")
            _safe_image(FIGURES_DIR / "feature_importance.png", "Feature Influence")
            if not feature_importance_df.empty:
                st.markdown("Top feature influence table")
                st.dataframe(feature_importance_df.head(10), use_container_width=True)

    with tabs[4]:
        st.subheader("Reliability Checks")
        if metrics_cv is None:
            st.info("No reliability artifact found. Run training first.")
        else:
            cv_rows = metrics_cv.get("cv", {}).get("rows", [])
            if cv_rows:
                cv_df = pd.DataFrame(cv_rows)
                st.markdown("Stratified K-Fold results")
                st.dataframe(cv_df, use_container_width=True)
            time_holdout = metrics_cv.get("time_holdout", {})
            st.markdown("Time-based holdout (simulation of real deployment)")
            if time_holdout.get("available"):
                st.json(time_holdout)
            else:
                st.warning(time_holdout.get("reason", "Time holdout unavailable."))
            if metrics:
                st.metric("Fatal recall (test split)", f"{metrics.get('fatal_recall_on_test_split', 0):.3f}")

    with tabs[5]:
        st.subheader("Error Analysis")
        if error_cases_df.empty:
            st.warning(
                "No error rows in current sample split. "
                "Treat this as optimistic sample behavior, not final model quality."
            )
        else:
            display_cols = [
                c
                for c in [
                    "actual_class_label",
                    "predicted_class_label",
                    "mean_temp",
                    "precipitation",
                    "pressure",
                    "number_of_vehicles",
                    "number_of_casualties",
                ]
                if c in error_cases_df.columns
            ]
            st.dataframe(error_cases_df[display_cols], use_container_width=True)
        if metrics:
            observations = metrics.get("error_analysis_observations", [])
            if observations:
                st.markdown("Key observations")
                for item in observations:
                    st.write(f"- {item}")

    with tabs[6]:
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

    with tabs[7]:
        st.subheader("Limitations")
        st.markdown(
            """
            - The current metrics are based on a **small sample demo dataset**.
            - Temporal and spatial features are still limited.
            - External enrichment data (OSM/air-quality/IMD) is not fully connected yet.
            """
        )
        st.subheader("Next Step")
        st.markdown(
            """
            - Add time-window features and road/location context.
            - Use cross-validation and class-balance strategies.
            - Expand error analysis with full-data runs and stability checks.
            """
        )


if __name__ == "__main__":
    main()
