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
        return True
    return False


def _render_module_explainer(section_key: str, t):
    explainers = {
        "overview": t(
            """
            **Research framing**
            - **Research question**: Can routinely collected transport and weather signals support a tri-level severity classification (`Slight/Serious/Fatal`)?
            - **Analytical unit**: Each row is treated as one observed collision context, with severity as the supervised target.
            - **Causal caution**: This project is predictive, not causal inference; coefficients and feature importances are interpreted as association evidence.
            - **Decision boundary**: Outputs are for risk triage and communication support, not autonomous policy enforcement.
            """,
            """
            **研究框架**
            - **研究问题**：常规交通与天气信号能否支持事故严重度三级分类（`Slight/Serious/Fatal`）？
            - **分析单元**：每一行代表一次事故上下文观测，严重度为监督学习目标变量。
            - **解释边界**：本项目是预测研究而非因果识别，系数/重要性仅解释“相关性证据”。
            - **应用定位**：输出用于风险分层与沟通支持，不作为自动执法或单点决策依据。
            """,
        ),
        "data": t(
            """
            **Data module interpretation protocol**
            - **Step 1: Structural audit**: verify schema completeness, data types, and key identifiers (`date/time/severity`).
            - **Step 2: Measurement validity**: inspect whether variables are physically meaningful for traffic risk (e.g., precipitation, visibility proxies, vehicle count).
            - **Step 3: Modeling readiness**: define the feasible feature space and record variables excluded due to missingness or leakage risk.
            - **Output meaning**: This table is the empirical boundary of what the model is allowed to learn.
            """,
            """
            **数据模块解读协议**
            - **步骤1：结构审计**：确认字段结构、数据类型和关键标识（`date/time/severity`）是否完备。
            - **步骤2：测量有效性**：判断变量是否对交通风险具有物理或行为意义（如降水、能见度代理、车辆数）。
            - **步骤3：建模可用性**：明确可进入模型的特征边界，并记录因缺失或泄漏风险被排除的变量。
            - **产出含义**：本页定义了模型可学习信息的经验边界。
            """,
        ),
        "quality": t(
            """
            **Data quality interpretation protocol**
            - **Label governance**: abnormal target codes (e.g., `-10`) are treated as non-inferential noise and removed before training.
            - **Missingness diagnostics**: feature-wise missing rate quantifies potential information loss and imputation uncertainty.
            - **Reproducibility evidence**: each cleaning decision is represented by explicit counts (rows removed, rows retained).
            - **Scientific implication**: model quality cannot exceed data quality; this module establishes evidence credibility.
            """,
            """
            **数据质量解读协议**
            - **标签治理**：将异常目标编码（如 `-10`）视为不可推断噪声，训练前剔除。
            - **缺失诊断**：按特征给出缺失率，量化信息损失与插补不确定性来源。
            - **可复现证据**：每项清洗决策都以可追溯计数呈现（剔除数、保留数）。
            - **研究含义**：模型上限受数据质量约束，本模块是证据可信性的基础。
            """,
        ),
        "performance": t(
            """
            **Performance interpretation protocol**
            - **Metric rationale**: Accuracy captures global correctness; Macro F1 equal-weights classes and protects minority safety outcomes.
            - **Model selection logic**: choose model via multi-metric comparison, not single-score optimization.
            - **Mechanism inspection**: confusion matrix localizes class-level failure modes; feature influence supports behavioral interpretation.
            - **Claim boundary**: report as sample-level predictive evidence, not definitive citywide performance.
            """,
            """
            **模型表现解读协议**
            - **指标依据**：Accuracy反映整体正确率；Macro F1对各类别等权，避免少数类（尤其fatal）被淹没。
            - **选模逻辑**：基于多指标比较做模型取舍，而非单一分数最大化。
            - **机制解释**：混淆矩阵定位类别级失误模式，特征影响用于解释模型行为路径。
            - **结论边界**：结果应表述为样本级预测证据，而非城市级最终结论。
            """,
        ),
        "reliability": t(
            """
            **Reliability interpretation protocol**
            - **Internal stability**: stratified K-fold mean/std indicates estimator variance under resampling.
            - **Temporal generalization**: time-based holdout approximates deployment under distribution shift.
            - **Safety focus**: fatal-class recall is monitored separately because false negatives are high-cost errors.
            - **Practical meaning**: reliability metrics determine whether performance is repeatable, not accidental.
            """,
            """
            **可靠性解读协议**
            - **内部稳定性**：分层K折均值/方差用于评估估计量在重采样下的波动。
            - **时间泛化能力**：时间外推切分用于模拟部署后的分布漂移情境。
            - **安全优先指标**：fatal类召回率单独监控，因为漏检属于高代价错误。
            - **实践意义**：可靠性指标回答“结果是否可复现”，而不只是“这次是否跑得好”。
            """,
        ),
        "error": t(
            """
            **Error-analysis interpretation protocol**
            - **Failure localization**: inspect where predictions fail by class pair (`actual -> predicted`).
            - **Context patterning**: examine whether specific weather/traffic contexts are over-represented among errors.
            - **Actionability**: translate error signatures into concrete data or feature interventions.
            - **Research value**: this module supports falsification and refinement, not model promotion.
            """,
            """
            **误差分析解读协议**
            - **失误定位**：按类别对（`真实 -> 预测`）识别模型失败位置。
            - **情境模式识别**：观察误判样本是否集中在特定天气/交通上下文。
            - **可执行转化**：将误差特征转化为数据补全或特征改造任务。
            - **研究价值**：本模块用于证伪与改进，而非单纯证明模型“有效”。
            """,
        ),
        "predict": t(
            """
            **Prediction module interpretation protocol**
            - **Usage mode**: single-case inference is a scenario simulator for sensitivity analysis.
            - **Controlled perturbation**: change one factor at a time (e.g., precipitation, hour, vehicles) to observe directional response.
            - **Probabilistic reading**: interpret full class-probability vector, not only top-1 label.
            - **Operational caution**: outputs require human oversight and policy context before action.
            """,
            """
            **预测模块解读协议**
            - **使用方式**：单样本推断用于情景模拟与敏感性分析。
            - **受控扰动**：一次改变一个变量（如降水、时段、车辆数），观察方向性响应。
            - **概率化解读**：必须结合完整类别概率分布，而非只看Top-1类别。
            - **应用约束**：预测结果需结合人工判断与政策语境，不能直接自动执行。
            """,
        ),
        "limits": t(
            """
            **Limitations interpretation protocol**
            - **External validity**: sample/demo evidence may not generalize to full-city distribution.
            - **Uncertainty disclosure**: unresolved bias sources are explicitly documented.
            - **Roadmap discipline**: next steps are prioritized by impact on validity, not by algorithm novelty.
            - **Professional standard**: transparent limitation reporting increases credibility.
            """,
            """
            **局限模块解读协议**
            - **外部有效性**：演示样本证据不等同于城市全量数据上的可泛化表现。
            - **不确定性披露**：主动公开仍未解决的偏差来源与数据盲区。
            - **路线优先级**：下一步以“提升有效性”为先，而不是“追求算法新颖”。
            - **专业性标准**：透明呈现局限可显著提升研究可信度。
            """,
        ),
    }
    st.info(explainers.get(section_key, ""))


def _render_figure_explainer(figure_key: str, t):
    explainers = {
        "model_comparison": t(
            """
            **Figure interpretation**
            - X-axis: candidate models; Y-axis: score value.
            - `Accuracy` reflects global correctness; `Macro F1` reflects class-balanced performance.
            - If bars diverge, prioritize Macro F1 when safety minority classes are policy-critical.
            - Practical meaning: model choice is a trade-off between overall fit and class fairness.
            """,
            """
            **图像解读**
            - 横轴为候选模型，纵轴为指标得分。
            - `Accuracy` 表示整体正确率，`Macro F1` 表示类别均衡表现。
            - 若两者差异明显，在安全场景中应优先参考 Macro F1。
            - 实践意义：选模是“整体拟合”与“类别公平”之间的取舍。
            """,
        ),
        "confusion_matrix": t(
            """
            **Figure interpretation**
            - Diagonal cells are correct classifications; off-diagonals are error transfers.
            - Focus on `Fatal -> non-Fatal` errors (false negatives) due to high safety risk.
            - Row-wise concentration indicates which true class is difficult for the model.
            - Practical meaning: this figure identifies where targeted data enrichment is needed.
            """,
            """
            **图像解读**
            - 对角线表示预测正确，非对角线表示类别误判流向。
            - 重点关注 `Fatal -> 非Fatal` 的漏检，这类错误安全代价最高。
            - 按行观察可判断模型对哪类真实样本识别困难。
            - 实践意义：该图用于定位后续应优先补强的数据情境。
            """,
        ),
        "feature_importance": t(
            """
            **Figure interpretation**
            - Higher importance indicates larger contribution to split/decision patterns.
            - Importance is model-dependent and does not imply causal effect magnitude.
            - Compare top factors with domain knowledge to assess plausibility.
            - Practical meaning: prioritize high-impact features for quality control and enrichment.
            """,
            """
            **图像解读**
            - 重要性越高，表示该特征在模型决策路径中的贡献越大。
            - 重要性是模型内部量，不代表严格因果效应大小。
            - 需与交通领域知识对照判断解释是否合理。
            - 实践意义：可据此优先投入高影响特征的数据治理与扩展。
            """,
        ),
    }
    st.caption(explainers.get(figure_key, ""))


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
            st.markdown(
                t(
                    "**Step evidence**: preview is used for schema/measurement sanity-check before feature engineering.",
                    "**步骤证据**：此预览用于特征工程前的结构与测量合理性校验。",
                )
            )
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
            st.caption(
                t(
                    "Interpretation: this count is a direct audit trail of label governance strictness.",
                    "解读：该计数是标签治理严格性的直接审计证据。",
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
                st.caption(
                    t(
                        "Interpretation: report mean and dispersion jointly; low variance indicates stable resampling behavior.",
                        "解读：应联合报告均值与离散程度；方差越低，重采样稳定性越好。",
                    )
                )
            time_holdout = metrics_cv.get("time_holdout", {})
            st.markdown(t("Time-based holdout (simulation of real deployment)", "时间外推切分（模拟真实部署）"))
            if time_holdout.get("available"):
                st.json(time_holdout)
                st.caption(
                    t(
                        "Interpretation: this split approximates future deployment, so degradation here is operationally important.",
                        "解读：该切分近似未来上线场景，因此这里的性能下降具有直接运营意义。",
                    )
                )
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
            st.caption(
                t(
                    "Interpretation: each row is a failure case for targeted diagnostics; prioritize recurring error patterns over isolated outliers.",
                    "解读：每一行都是可定位的失败样本；应优先处理重复出现的模式性误差，而非个别离群点。",
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
