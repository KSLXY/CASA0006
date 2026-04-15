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
DATA_QUALITY_REPORT_PATH = ROOT_DIR / "artifacts" / "data_quality_report.json"
LEAKAGE_REPORT_PATH = ROOT_DIR / "artifacts" / "leakage_check_report.json"
THRESHOLD_REPORT_PATH = ROOT_DIR / "artifacts" / "threshold_report.csv"
CALIBRATION_REPORT_PATH = ROOT_DIR / "artifacts" / "calibration_report.json"
SEARCH_REPORT_PATH = ROOT_DIR / "artifacts" / "hyperparameter_search.json"
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


@st.cache_data
def load_json_if_exists(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


@st.cache_data
def load_threshold_report():
    if THRESHOLD_REPORT_PATH.exists():
        return pd.read_csv(THRESHOLD_REPORT_PATH)
    return pd.DataFrame()


def _safe_image(path: Path, caption: str):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    return False


def _render_module_explainer(section_key: str, t):
    explainers = {
        "overview": t(
            """
            **Project scope**
            - The project studies whether weather and road context can support severity classification in three levels `Slight` `Serious` `Fatal`.
            - Each row represents one collision context. The model learns repeated patterns across many observations.
            - The work is predictive rather than causal. Prediction estimates risk level while causality tests direct mechanism.
            - Results are designed for risk communication and decision support in early analysis.
            """,
            """
            **项目范围**
            - 本项目关注一个核心问题。天气与道路信息是否能够支持事故严重度三级分类 `Slight` `Serious` `Fatal`。
            - 每一行数据对应一次事故场景。模型从大量场景中提取可重复规律。
            - 研究定位为预测分析，不是因果识别。预测用于估计风险水平，因果用于验证直接作用机制。
            - 输出主要服务于风险沟通与早期决策支持。
            """,
        ),
        "data": t(
            """
            **Data preparation logic**
            - The first step checks whether key fields are complete and structurally valid, including date, time, and severity label.
            - The second step reviews potential explanatory signals such as precipitation, pressure, temperature, and vehicle related counts.
            - Variables with high missingness or leakage risk are marked for cautious use.
            - Leakage means training data contains hidden answer-like information and can lead to unrealistically high scores.
            """,
            """
            **数据准备逻辑**
            - 首先检查关键字段是否齐全且格式有效，重点包括日期、时间与严重度标签。
            - 然后筛查可解释风险的候选变量，例如降水、气压、温度与车辆相关计数。
            - 对高缺失字段与潜在泄漏字段进行标记，避免直接进入训练。
            - 泄漏指训练阶段混入近似答案的信息，常会造成不真实的高分表现。
            """,
        ),
        "quality": t(
            """
            **Data quality evidence**
            - Abnormal target codes such as `-10` are removed before modeling to reduce label noise.
            - Feature level missing rates are reported to reveal potential uncertainty in downstream estimates.
            - Cleaning actions are tracked with explicit counts so the process remains auditable and reproducible.
            - This module defines the reliability baseline for all later model conclusions.
            """,
            """
            **数据质量证据**
            - 建模前先剔除异常标签，例如 `-10`，以降低标签噪声对模型方向的干扰。
            - 按特征给出缺失率，帮助识别后续估计中的不确定来源。
            - 所有清洗动作均保留数量记录，流程可以追溯，也便于复现。
            - 该模块决定了后续模型结论的可信下限。
            """,
        ),
        "performance": t(
            """
            **Performance interpretation**
            - Multiple models are compared under the same setting to support transparent selection.
            - Accuracy reflects overall correctness. Macro F1 reflects balanced performance across classes and protects minority outcomes.
            - Confusion matrix and feature influence plots are used to interpret error structure and decision pattern.
            - Reported values are treated as sample level evidence with controlled claims.
            """,
            """
            **模型表现解读**
            - 在相同条件下比较多个模型，再进行选择，避免单模型偏见。
            - Accuracy反映整体正确率。Macro F1反映类别均衡表现，对少数类更敏感。
            - 结合混淆矩阵和特征影响图，可定位误差结构并解释模型判别路径。
            - 当前结果属于样本级证据，结论保持克制，不做外推夸大。
            """,
        ),
        "reliability": t(
            """
            **Reliability checks**
            - Single split performance can be unstable. K-fold validation measures repeatability under resampling.
            - Time based holdout simulates practical deployment by training on earlier data and testing on later periods.
            - Fatal class recall is reported separately because false negatives in high severity events carry larger risk.
            - This module focuses on stability and transferability rather than peak score only.
            """,
            """
            **可靠性检验**
            - 单次切分结果可能受随机性影响。K折验证用于评估重采样条件下的稳定程度。
            - 时间外推切分用于模拟真实部署，采用过去训练、未来测试的方式。
            - Fatal类召回率单独报告，因为高严重度漏检具有更高风险成本。
            - 本模块强调稳定性与可迁移性，而不仅是单点高分。
            """,
        ),
        "error": t(
            """
            **Error analysis purpose**
            - Misclassified samples are listed to locate concrete failure patterns.
            - Environmental and traffic context around errors is examined to identify recurring difficulty zones.
            - Findings are translated into actionable improvements in feature design and data enrichment.
            - This module connects model diagnosis with the next round of iteration.
            """,
            """
            **误差分析目的**
            - 列出误判样本，定位模型失败的具体模式。
            - 结合天气与交通上下文观察误差是否在特定场景中反复出现。
            - 将诊断结论转化为可执行改进，包括特征重构与数据补全。
            - 本模块用于把模型问题连接到下一轮迭代动作。
            """,
        ),
        "predict": t(
            """
            **Scenario prediction use**
            - This page serves as an interactive sandbox for single case inference.
            - Controlled one factor changes help reveal directional sensitivity of model outputs.
            - Full probability distribution should be read together with top class label.
            - The module is intended for demonstration, explanation, and preliminary assessment.
            """,
            """
            **情景预测用途**
            - 本页提供单样本推断沙盒，用于观察输入变化后的输出响应。
            - 采用单因素逐步调整，更容易识别变量对结果的方向性影响。
            - 除Top类别外，还应结合完整概率分布进行风险阅读。
            - 该模块用于演示解释与初步评估，不直接替代现场决策。
            """,
        ),
        "limits": t(
            """
            **Limitations and roadmap**
            - Boundaries of current evidence are stated explicitly to prevent over interpretation.
            - Known gaps include limited coverage, limited spatiotemporal granularity, and partial external data linkage.
            - Next steps are defined as executable tasks with clear validation targets.
            - Transparent reporting of limits improves credibility and practical value.
            """,
            """
            **局限与路线**
            - 先明确现有证据边界，避免结论超出数据支持范围。
            - 当前短板包括覆盖范围有限、时空粒度不足、外部数据接入不完整。
            - 下一步以可执行任务形式展开，并配套可检验的结果目标。
            - 透明呈现局限可提升项目可信度与应用价值。
            """,
        ),
    }
    st.info(explainers.get(section_key, ""))


def _render_figure_explainer(figure_key: str, t):
    explainers = {
        "model_comparison": t(
            """
            **Reading guide**
            - Horizontal axis shows model names. Vertical axis shows evaluation scores.
            - Accuracy indicates overall correctness. Macro F1 indicates balance across classes.
            - When Accuracy is close, higher Macro F1 usually means better treatment of minority classes.
            - The chart supports model selection with visible tradeoff evidence.
            """,
            """
            **读图说明**
            - 横轴为模型名称，纵轴为评估得分。
            - Accuracy代表整体正确率。Macro F1代表类别均衡表现。
            - 当Accuracy接近时，Macro F1更高通常意味着对少数类更稳健。
            - 该图用于展示选模过程中的权衡依据。
            """,
        ),
        "confusion_matrix": t(
            """
            **Reading guide**
            - Diagonal cells represent correct predictions. Off diagonal cells represent class confusion.
            - `Fatal` predicted as non-fatal is a key risk signal because it reflects high cost misses.
            - Rows with heavy off diagonal counts indicate classes that remain difficult for the model.
            - The matrix helps prioritize targeted data and feature improvement.
            """,
            """
            **读图说明**
            - 对角线表示预测正确，非对角线表示类别混淆。
            - `Fatal`被预测为非Fatal属于高风险漏判，需重点关注。
            - 某一行非对角线计数较高，说明该真实类别识别难度较大。
            - 该图可用于确定优先补强的数据与特征方向。
            """,
        ),
        "feature_importance": t(
            """
            **Reading guide**
            - Higher value means stronger contribution to model decision rules.
            - Feature importance is not causal proof. It reflects predictive usefulness under current model structure.
            - Top features should be checked against domain knowledge for plausibility.
            - The ranking supports prioritization in feature engineering and data collection.
            """,
            """
            **读图说明**
            - 数值越高表示该特征对模型判别规则贡献越大。
            - 特征重要性不等于因果结论。其含义是当前模型下的预测贡献度。
            - 需要结合交通领域常识检验结果是否合理。
            - 该排序可用于特征工程与数据补全的优先级安排。
            """,
        ),
    }
    st.caption(explainers.get(figure_key, ""))


def _render_story_transition(section_key: str, t):
    transitions = {
        "overview": t(
            "The next section turns from research question to available evidence in raw data.",
            "下一部分从研究问题进入原始数据证据。",
        ),
        "data": t(
            "After data structure is confirmed, quality checks become the necessary next step.",
            "确认数据结构后，下一步进入数据质量检验。",
        ),
        "quality": t(
            "With cleaned data in place, model comparison and evaluation can proceed.",
            "完成清洗后，进入模型比较与评估阶段。",
        ),
        "performance": t(
            "Performance scores are followed by reliability tests to examine repeatability.",
            "分数结果之后，需要通过可靠性检验评估可复现性。",
        ),
        "reliability": t(
            "Reliability findings lead naturally to error analysis and targeted diagnosis.",
            "可靠性结果之后，进入误差分析与定向诊断。",
        ),
        "error": t(
            "After diagnosis of errors, case based prediction is used for scenario demonstration.",
            "误差诊断完成后，进入案例化情景预测演示。",
        ),
        "predict": t(
            "The final section summarizes evidence limits and defines the next implementation targets.",
            "最后总结证据边界，并给出下一步执行目标。",
        ),
    }
    text = transitions.get(section_key)
    if text:
        st.markdown(f"> {text}")


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
            "从课程作业走向作品集，重点展示数据治理、模型比较与在线部署。",
        )
    )

    payload, model_load_error = load_model_payload()
    metrics = load_metrics()
    metrics_cv = load_metrics_cv()
    sample_df = load_sample_data()
    model_compare_df = load_model_comparison()
    error_cases_df = load_error_cases()
    feature_importance_df = load_feature_importance()
    data_quality_report = load_json_if_exists(DATA_QUALITY_REPORT_PATH)
    leakage_report = load_json_if_exists(LEAKAGE_REPORT_PATH)
    calibration_report = load_json_if_exists(CALIBRATION_REPORT_PATH)
    search_report = load_json_if_exists(SEARCH_REPORT_PATH)
    threshold_df = load_threshold_report()

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
        _render_story_transition("overview", t)
        st.markdown(
            t(
                """
                - **City context** Road-collision severity is a key public-safety signal.
                - **Data challenge** Real-world labels include invalid values such as `-10`, so cleaning decisions matter.
                - **Modeling decision** Baseline and tree models are compared, and **Macro F1** is used to preserve minority class visibility.
                - **Deployment outcome** The demo is online and reproducible with GitHub and Streamlit.
                """,
                """
                - **城市问题** 交通事故严重度是公共安全的重要信号。
                - **数据挑战** 原始标签中存在异常值，例如 `-10`，因此必须先清洗再建模。
                - **建模决策** 采用多模型比较，并以 **Macro F1** 作为关键指标，避免少数类被忽视。
                - **交付结果** 项目可通过 GitHub 与 Streamlit 在线复现和展示。
                """,
            )
        )
        if metrics:
            st.info(
                t(
                    f"Performance note: **{metrics.get('performance_note', 'sample demo performance')}**",
                    f"结果说明 **{metrics.get('performance_note', 'sample demo performance')}**",
                )
            )

    with tabs[1]:
        st.subheader(t("Sample Dataset Preview", "样本数据预览"))
        _render_module_explainer("data", t)
        _render_story_transition("data", t)
        if sample_df.empty:
            st.warning(t("Sample data not found.", "未找到样本数据。"))
        else:
            st.markdown(
                t(
                    "**Step evidence** This preview is used for schema and measurement sanity-check before feature engineering.",
                    "**步骤证据** 此预览用于特征工程前的结构与测量合理性校验。",
                )
            )
            st.dataframe(sample_df.head(20), use_container_width=True)
            st.write(t(f"Rows {len(sample_df)}", f"样本行数 {len(sample_df)}"))
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
        _render_story_transition("quality", t)
        if metrics is None:
            st.info(t("No metrics file found. Run training first.", "未找到指标文件，请先运行训练。"))
        else:
            st.metric(t("Removed invalid target labels", "剔除异常目标标签"), int(metrics.get("rows_removed_invalid_target", 0)))
            st.caption(
                t(
                    "Interpretation This count is a direct audit trail of label-governance strictness.",
                    "解读 该计数可直接反映标签治理的执行强度。",
                )
            )
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
                        "Missing-rate is computed before imputation; high-missing features imply larger uncertainty propagation risk.",
                        "缺失率在插补前计算；高缺失特征意味着更高的不确定性传播风险。",
                    )
                )
            else:
                st.write(t("Missing-rate details unavailable.", "缺失率明细不可用。"))
            if data_quality_report:
                st.markdown(t("Data quality evidence file", "数据质量证据文件"))
                st.json(data_quality_report)

    with tabs[3]:
        st.subheader(t("Model Comparison & Evidence", "模型对比与证据"))
        _render_module_explainer("performance", t)
        _render_story_transition("performance", t)
        if metrics is None:
            st.info(t("No metrics file found. Run training first.", "未找到指标文件，请先运行训练。"))
        else:
            selected = metrics.get("selected_model", "N/A")
            st.write(t(f"Selected model **{selected}**", f"选中模型 **{selected}**"))
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
                _render_figure_explainer("model_comparison", t)
            if _safe_image(FIGURES_DIR / "model_comparison.png", t("Model Comparison Figure", "模型对比图")):
                _render_figure_explainer("model_comparison", t)
            if _safe_image(FIGURES_DIR / "confusion_matrix.png", t("Confusion Matrix (Selected Model)", "混淆矩阵（选中模型）")):
                _render_figure_explainer("confusion_matrix", t)
            if _safe_image(FIGURES_DIR / "feature_importance.png", t("Feature Influence", "特征影响图")):
                _render_figure_explainer("feature_importance", t)
            if not feature_importance_df.empty:
                st.markdown(t("Top feature influence table", "特征影响Top表"))
                st.dataframe(feature_importance_df.head(10), use_container_width=True)
            if search_report:
                st.markdown(t("Hyperparameter search summary", "超参数搜索摘要"))
                st.json(search_report)

    with tabs[4]:
        st.subheader(t("Reliability Checks", "可靠性检验"))
        _render_module_explainer("reliability", t)
        _render_story_transition("reliability", t)
        if metrics_cv is None:
            st.info(t("No reliability artifact found. Run training first.", "未找到可靠性结果，请先运行训练。"))
        else:
            cv_rows = metrics_cv.get("cv", {}).get("rows", [])
            if cv_rows:
                cv_df = pd.DataFrame(cv_rows)
                st.markdown(t("Stratified K-Fold results", "分层K折结果"))
                st.dataframe(cv_df, use_container_width=True)
                st.caption(
                    t(
                        "Interpretation Mean and dispersion should be read together. Lower variance indicates more stable resampling behavior.",
                        "解读 需要同时观察均值与离散程度。方差越低，重采样稳定性通常越好。",
                    )
                )
            time_holdout = metrics_cv.get("time_holdout", {})
            st.markdown(t("Time-based holdout (simulation of real deployment)", "时间外推切分（模拟真实部署）"))
            if time_holdout.get("available"):
                st.json(time_holdout)
                st.caption(
                    t(
                        "Interpretation This split approximates future deployment. Performance degradation here has direct operational meaning.",
                        "解读 该切分近似未来上线场景。因此这一部分的性能下降具有直接业务意义。",
                    )
                )
            else:
                st.warning(time_holdout.get("reason", t("Time holdout unavailable.", "时间外推不可用。")))
            if metrics:
                st.metric(t("Fatal recall (test split)", "Fatal召回率（测试集）"), f"{metrics.get('fatal_recall_on_test_split', 0):.3f}")
            if not threshold_df.empty:
                st.markdown(t("Fatal threshold sensitivity", "Fatal阈值敏感性"))
                st.dataframe(threshold_df, use_container_width=True)
            if calibration_report:
                st.markdown(t("Calibration report", "校准报告"))
                st.json(calibration_report)

    with tabs[5]:
        st.subheader(t("Error Analysis", "误差分析"))
        _render_module_explainer("error", t)
        _render_story_transition("error", t)
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
            st.caption(
                t(
                    "Interpretation Each row is a traceable failure case. Recurring error patterns are usually more important than isolated outliers.",
                    "解读 每一行都是可追踪的失败样本。与个别离群点相比，重复出现的误差模式更值得优先处理。",
                )
            )
        if metrics:
            observations = metrics.get("error_analysis_observations", [])
            if observations:
                st.markdown(t("Key observations", "关键观察"))
                for item in observations:
                    st.write(f"- {item}")

    with tabs[6]:
        st.subheader(t("Single Prediction", "单条预测"))
        _render_module_explainer("predict", t)
        _render_story_transition("predict", t)
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
                        f"预测类别 {class_labels[pred_idx]}（严重度={pred_idx + 1}）",
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
        if leakage_report:
            st.subheader(t("Leakage Check", "泄漏检查"))
            st.json(leakage_report)
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
