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


def _render_module_explainer(section_key: str, t):
    explainers = {
        "overview": t(
            """
            **How to read this module**
            1. We define the safety problem first (severity classification, not simple accident counting).
            2. We show why data governance is necessary before modeling.
            3. We frame model output as risk-support evidence rather than automatic decision.
            """,
            """
            **本模块怎么看**
            1. 先定义问题：这是“严重度分类”，不是简单事故数量统计。
            2. 先做数据治理再建模，避免脏标签直接污染结果。
            3. 输出定位为风险辅助证据，不是自动执法决策。
            """,
        ),
        "data": t(
            """
            **How to read this module**
            1. Inspect raw fields and identify available explanatory signals.
            2. Check whether key fields (date/time/severity) are present and usable.
            3. Treat this table as the input boundary for feature engineering.
            """,
            """
            **本模块怎么看**
            1. 先确认原始字段是否能解释事故严重度。
            2. 检查 date/time/severity 等关键字段是否完整可用。
            3. 把本页视为后续特征工程的输入边界。
            """,
        ),
        "quality": t(
            """
            **How to read this module**
            1. Invalid target removal quantifies label risk control.
            2. Missing-rate table shows where imputation may introduce uncertainty.
            3. High-quality modeling starts from explicit data quality evidence.
            """,
            """
            **本模块怎么看**
            1. 异常标签剔除数量用于量化标签治理效果。
            2. 缺失率表反映插补可能带来的不确定性来源。
            3. 模型可信度首先来自数据质量证据。
            """,
        ),
        "performance": t(
            """
            **How to read this module**
            1. Accuracy gives overall correctness; Macro F1 protects minority classes.
            2. Compare all candidate models before selecting one.
            3. Confusion matrix and feature influence explain *why* model behaves this way.
            """,
            """
            **本模块怎么看**
            1. Accuracy 看整体，Macro F1 保护少数类（尤其 fatal）。
            2. 先比较候选模型，再做选择，不拍脑袋定模型。
            3. 通过混淆矩阵和特征影响解释“模型为什么这样预测”。
            """,
        ),
        "reliability": t(
            """
            **How to read this module**
            1. K-fold mean/std reflects result stability.
            2. Time-based holdout simulates future deployment behavior.
            3. Fatal recall is a safety-sensitive risk metric to monitor separately.
            """,
            """
            **本模块怎么看**
            1. K折的均值和方差用于评估结果稳定性。
            2. 时间外推切分模拟真实上线后的时间漂移风险。
            3. Fatal 召回率是安全场景的关键风险指标，需要单独看。
            """,
        ),
        "error": t(
            """
            **How to read this module**
            1. Error cases locate where model fails, not where it succeeds.
            2. Analyze feature patterns around misclassification.
            3. Convert findings into next-step feature or data collection actions.
            """,
            """
            **本模块怎么看**
            1. 误差分析关注“模型错在哪里”，而不只是对在哪里。
            2. 观察误判样本周边的特征模式。
            3. 把误差结论转化为下一步数据补全与特征改进动作。
            """,
        ),
        "predict": t(
            """
            **How to read this module**
            1. This is a scenario simulator, not a production decision engine.
            2. Modify features to test sensitivity (rain/peak-hour/vehicles).
            3. Use probability distribution, not only top-1 class, for interpretation.
            """,
            """
            **本模块怎么看**
            1. 这里是场景模拟器，不是生产级自动决策引擎。
            2. 可调雨量、高峰时段、车辆数观察模型敏感性。
            3. 解释时看概率分布，不只看第一名类别。
            """,
        ),
        "limits": t(
            """
            **How to read this module**
            1. We explicitly separate demo evidence from policy-level conclusions.
            2. Known limitations are documented to prevent over-claiming.
            3. Next-step items are concrete and execution-oriented.
            """,
            """
            **本模块怎么看**
            1. 明确区分“演示证据”与“政策级结论”。
            2. 主动写出局限，防止过度宣称。
            3. 下一步必须是可执行任务，而不是泛泛而谈。
            """,
        ),
    }
    st.info(explainers.get(section_key, ""))


def main() -> None:
    def t(en: str, zh: str) -> str:
        return zh if lang == "中文" else en

    st.set_page_config(
        page_title="London Road Severity Explorer",
        page_icon=":bar_chart:",
        layout="wide",
    )
    lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"], index=0)
    st.title(t("London Road Collision Severity Project", "伦敦交通事故严重度预测项目"))
    st.caption(
        t(
            "From coursework to portfolio: data quality, model comparison, and deployment.",
            "从课程作业到作品集：数据治理、模型对比与在线部署。",
        )
    )

    payload, model_load_error = load_model_payload()
    metrics = load_metrics()
    metrics_cv = load_metrics_cv()
    sample_df = load_sample_data()
    model_compare_df = load_model_comparison()
    error_cases_df = load_error_cases()
    feature_importance_df = load_feature_importance()

    if model_load_error:
        st.error(
            t(
                "Model artifact could not be loaded. This is often a Python-version mismatch issue on cloud. ",
                "模型文件加载失败，这通常与云端 Python 版本不一致有关。"
            )
            + f"\n\nDetails: {model_load_error}"
        )

    selected_model = metrics.get("selected_model", "N/A") if metrics else "N/A"
    rows_total = int(metrics.get("rows_total", 0)) if metrics else 0
    rows_used = int(metrics.get("rows_used_for_training", 0)) if metrics else 0
    rows_removed = int(metrics.get("rows_removed_invalid_target", 0)) if metrics else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(t("Rows total", "总样本数"), rows_total)
    k2.metric(t("Rows used", "训练使用样本"), rows_used)
    k3.metric(t("Invalid target removed", "剔除异常标签"), rows_removed)
    k4.metric(t("Best model", "最佳模型"), selected_model)

    tabs = st.tabs(
        [
            "Overview",
            t("Data", "数据"),
            t("Data Quality", "数据质量"),
            t("Model Performance", "模型表现"),
            t("Reliability", "可靠性"),
            t("Error Analysis", "误差分析"),
            t("Predict", "预测"),
            t("Limitations & Next Step", "局限与下一步"),
        ]
    )

    with tabs[0]:
        st.subheader(t("Problem Story", "项目主线"))
        _render_module_explainer("overview", t)
        st.markdown(
            t(
                """
                - **City context**: Road-collision severity is a key public-safety signal.
                - **Data challenge**: Real-world labels include invalid values (e.g. `-10`), so cleaning decisions matter.
                - **Modeling decision**: We compare baseline and tree models, and select by **Macro F1** to keep minority classes visible.
                - **Deployment outcome**: This demo is online and reproducible from GitHub + Streamlit.
                """,
                """
                - **城市问题**：交通事故严重度是公共安全的核心信号。
                - **数据挑战**：真实标签存在异常值（如 `-10`），必须先治理再建模。
                - **建模决策**：采用多模型对比，并以 **Macro F1** 作为主指标，避免忽视少数类。
                - **交付结果**：项目可在 GitHub + Streamlit 在线复现与展示。
                """,
            )
        )
        if metrics:
            st.info(
                t(
                    f"Performance note: **{metrics.get('performance_note', 'sample demo performance')}**",
                    f"结果说明：**{metrics.get('performance_note', 'sample demo performance')}**",
                )
            )

    with tabs[1]:
        st.subheader(t("Sample Dataset Preview", "样本数据预览"))
        _render_module_explainer("data", t)
        if sample_df.empty:
            st.warning(t("Sample data not found.", "未找到样本数据。"))
        else:
            st.dataframe(sample_df.head(20), use_container_width=True)
            st.write(t(f"Rows: {len(sample_df)}", f"样本行数：{len(sample_df)}"))
        st.markdown(
            t(
                """
                **Data sources**
                - London weather data (Kaggle public dataset)
                - UK DfT road collision records
                """,
                """
                **数据来源**
                - 伦敦天气（Kaggle 公开数据）
                - 英国 DfT 道路碰撞数据
                """,
            )
        )

    with tabs[2]:
        st.subheader(t("Data Quality", "数据质量"))
        _render_module_explainer("quality", t)
        if metrics is None:
            st.info(t("No metrics file found. Run training first.", "未找到指标文件，请先运行训练。"))
        else:
            st.metric(t("Removed invalid target labels", "剔除异常目标标签"), int(metrics.get("rows_removed_invalid_target", 0)))
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
                st.caption(
                    t(
                        "Missing-rate is computed before imputation and used as data quality evidence.",
                        "缺失率在填补前统计，用于展示数据质量证据。",
                    )
                )
            else:
                st.write(t("Missing-rate details unavailable.", "缺失率明细不可用。"))

    with tabs[3]:
        st.subheader(t("Model Comparison & Evidence", "模型对比与证据"))
        _render_module_explainer("performance", t)
        if metrics is None:
            st.info(t("No metrics file found. Run training first.", "未找到指标文件，请先运行训练。"))
        else:
            selected = metrics.get("selected_model", "N/A")
            st.write(t(f"Selected model: **{selected}**", f"选中模型：**{selected}**"))
            selected_metrics = metrics.get("selected_model_metrics", {})
            c1, c2 = st.columns(2)
            c1.metric(t("Accuracy", "准确率"), f"{selected_metrics.get('accuracy', 0):.3f}")
            c2.metric(t("Macro F1", "宏平均F1"), f"{selected_metrics.get('f1_macro', 0):.3f}")
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
            _safe_image(FIGURES_DIR / "model_comparison.png", t("Model Comparison Figure", "模型对比图"))
            _safe_image(FIGURES_DIR / "confusion_matrix.png", t("Confusion Matrix (Selected Model)", "混淆矩阵（选中模型）"))
            _safe_image(FIGURES_DIR / "feature_importance.png", t("Feature Influence", "特征影响图"))
            if not feature_importance_df.empty:
                st.markdown(t("Top feature influence table", "特征影响Top表"))
                st.dataframe(feature_importance_df.head(10), use_container_width=True)

    with tabs[4]:
        st.subheader(t("Reliability Checks", "可靠性检验"))
        _render_module_explainer("reliability", t)
        if metrics_cv is None:
            st.info(t("No reliability artifact found. Run training first.", "未找到可靠性结果，请先运行训练。"))
        else:
            cv_rows = metrics_cv.get("cv", {}).get("rows", [])
            if cv_rows:
                cv_df = pd.DataFrame(cv_rows)
                st.markdown(t("Stratified K-Fold results", "分层K折结果"))
                st.dataframe(cv_df, use_container_width=True)
            time_holdout = metrics_cv.get("time_holdout", {})
            st.markdown(t("Time-based holdout (simulation of real deployment)", "时间外推切分（模拟真实部署）"))
            if time_holdout.get("available"):
                st.json(time_holdout)
            else:
                st.warning(time_holdout.get("reason", t("Time holdout unavailable.", "时间外推不可用。")))
            if metrics:
                st.metric(t("Fatal recall (test split)", "Fatal召回率（测试集）"), f"{metrics.get('fatal_recall_on_test_split', 0):.3f}")

    with tabs[5]:
        st.subheader(t("Error Analysis", "误差分析"))
        _render_module_explainer("error", t)
        if error_cases_df.empty:
            st.warning(
                t(
                    "No error rows in current sample split. Treat this as optimistic sample behavior, not final model quality.",
                    "当前样本切分下没有误差样本。这通常表示样本规模较小，不能据此断言模型泛化能力。",
                )
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
                st.markdown(t("Key observations", "关键观察"))
                for item in observations:
                    st.write(f"- {item}")

    with tabs[6]:
        st.subheader(t("Single Prediction", "单条预测"))
        _render_module_explainer("predict", t)
        if payload is None:
            st.info(t("No trained model found. Run `python -m src.train --config configs/default.yaml` first.", "未找到训练模型，请先运行训练命令。"))
        else:
            feature_columns = payload["feature_columns"]
            defaults = payload.get("feature_defaults", {})
            class_labels = payload.get("class_labels", ["Slight", "Serious", "Fatal"])
            inputs = {}
            for col in feature_columns:
                default = float(defaults.get(col, 0.0))
                inputs[col] = st.number_input(col, value=default, format="%.4f")

            if st.button(t("Predict severity", "预测严重度")):
                X = pd.DataFrame([inputs], columns=feature_columns)
                pipeline = payload["pipeline"]
                pred_idx = int(pipeline.predict(X)[0])
                probs = pipeline.predict_proba(X)[0]
                st.success(
                    t(
                        f"Predicted class: {class_labels[pred_idx]} (severity={pred_idx + 1})",
                        f"预测类别：{class_labels[pred_idx]}（严重度={pred_idx + 1}）",
                    )
                )
                prob_table = pd.DataFrame(
                    {"class_label": class_labels, "probability": probs}
                ).sort_values("probability", ascending=False)
                st.dataframe(prob_table, use_container_width=True)

    with tabs[7]:
        st.subheader(t("Limitations", "局限"))
        _render_module_explainer("limits", t)
        st.markdown(
            t(
                """
                - The current metrics are based on a **small sample demo dataset**.
                - Temporal and spatial features are still limited.
                - External enrichment data (OSM/air-quality/IMD) is not fully connected yet.
                """,
                """
                - 当前指标来自**小样本演示数据**，可能偏乐观。
                - 时空特征仍然有限。
                - 外部补充数据（OSM/空气质量/IMD）尚未完全打通。
                """,
            )
        )
        st.subheader(t("Next Step", "下一步"))
        st.markdown(
            t(
                """
                - Add time-window features and road/location context.
                - Use cross-validation and class-balance strategies.
                - Expand error analysis with full-data runs and stability checks.
                """,
                """
                - 增加时间窗口和道路/空间上下文特征。
                - 强化交叉验证与类别不平衡策略。
                - 在全量数据上扩展误差分析与稳定性检验。
                """,
            )
        )


if __name__ == "__main__":
    main()
