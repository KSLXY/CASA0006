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
            **What this project is about**
            - This project asks a simple question: given weather and road context, can we estimate accident severity (`Slight/Serious/Fatal`)?
            - Think of each row as one real accident snapshot. The model learns patterns from many snapshots.
            - This is a **prediction** project, not a **causality** project:
            causality means proving "X directly causes Y", while prediction means "X helps us estimate Y".
            - The result is a risk-support tool, not an automatic decision maker.
            """,
            """
            **这个项目在讲什么**
            - 这个项目回答一个直白的问题：给定天气和道路信息，能否估计事故严重度（`Slight/Serious/Fatal`）？
            - 你可以把每一行数据理解成“一次真实事故现场快照”，模型从大量快照中学习规律。
            - 这是**预测**项目，不是**因果**项目：
            因果是证明“X导致Y”，预测是“X能帮助估计Y”。
            - 结果用于风险提示和辅助判断，不是自动执法或自动决策系统。
            """,
        ),
        "data": t(
            """
            **What we do in this tab**
            - First, we check whether key columns exist and look reasonable (`date/time/severity`).
            - Then we identify useful signals, such as rain, pressure, temperature, and vehicle/casualty counts.
            - We also mark risky columns that may be missing too much or may cause data leakage.
            data leakage means the model accidentally sees "future/answer-like" information during training.
            - In short, this tab defines what information the model can honestly use.
            """,
            """
            **这个页面在做什么**
            - 先检查关键字段在不在、格式对不对（`date/time/severity`）。
            - 再看哪些字段真的有帮助，比如降水、气压、温度、车辆数、伤亡数。
            - 同时标记高风险字段：缺失太多，或可能造成数据泄漏。
            数据泄漏就是训练时“偷看答案”或“偷看未来信息”。
            - 简单说，这一页定义了模型可以“合法学习”的信息范围。
            """,
        ),
        "quality": t(
            """
            **Why this tab matters**
            - We clean abnormal labels first (for example `-10`), because noisy labels will directly mislead the model.
            - We calculate missing rate for each feature to see where uncertainty may come from.
            - We keep cleaning counts transparent: how many rows were removed and how many stayed.
            - Core message: if data quality is weak, model scores are not trustworthy.
            """,
            """
            **为什么这一页很重要**
            - 先清理异常标签（比如 `-10`），因为脏标签会直接把模型带偏。
            - 按特征计算缺失率，找出不确定性来源。
            - 所有清洗动作都给出数量证据：剔除了多少、保留了多少。
            - 核心结论：数据质量不过关，模型分数再好也不可信。
            """,
        ),
        "performance": t(
            """
            **How to read model results**
            - We compare multiple models instead of trusting one model by default.
            - `Accuracy` = overall hit rate; `Macro F1` = average class performance (gives minority class fair weight).
            - We then use confusion matrix and feature-importance charts to explain where the model works and where it fails.
            - This is sample-level evidence, so we avoid over-claiming.
            """,
            """
            **模型结果怎么读**
            - 我们先比较多个模型，而不是默认某一个最好。
            - `Accuracy` 是整体命中率；`Macro F1` 是各类别平均表现（会公平照顾少数类）。
            - 再结合混淆矩阵和特征影响图解释“哪里预测得好、哪里容易错”。
            - 这里是样本级证据，不做夸大结论。
            """,
        ),
        "reliability": t(
            """
            **Why we still need this tab after good scores**
            - A single test score may be luck, so we use K-fold validation to check stability.
            K-fold means splitting data into several folds and repeating training/testing.
            - We also test by time order to mimic real deployment (train on past, test on later data).
            - We track fatal recall separately because missing a high-risk case has higher cost.
            - This tab answers: "Can performance repeat in a more realistic setting?"
            """,
            """
            **为什么分数不错还要看这一页**
            - 单次测试高分可能是运气，所以要用K折验证看稳定性。
            K折就是把数据分成多份，轮流训练和测试。
            - 还要按时间切分，模拟真实上线（用过去训练、用未来测试）。
            - fatal类召回率单独看，因为高风险事故漏判代价更高。
            - 这一页回答的是：“结果能不能稳定复现？”
            """,
        ),
        "error": t(
            """
            **What we learn from mistakes**
            - We list wrong predictions to see exactly what kind of cases confuse the model.
            - We check whether errors cluster under specific conditions (e.g., rain, high traffic load).
            - Then we turn these findings into actions: collect better data or add better features.
            - This is the most practical tab: it tells us how to improve next.
            """,
            """
            **从错误里学到什么**
            - 把预测错误样本列出来，看模型到底被什么情况“绊住”。
            - 观察误差是否集中在某些情境（例如降雨、交通负荷高）。
            - 再把发现转成改进动作：补数据、改特征、调模型。
            - 这是最实用的一页：直接告诉我们下一步该怎么优化。
            """,
        ),
        "predict": t(
            """
            **How to use this tab**
            - This is a sandbox: change inputs and observe how prediction changes.
            - Best practice is one-factor-at-a-time testing, so you can see each factor's effect clearly.
            - Do not look only at the top class; the probability distribution gives richer risk information.
            - It is for demonstration and understanding, not direct field decision.
            """,
            """
            **这一页怎么用**
            - 这是一个沙盒：改输入，看预测怎么变化。
            - 建议一次只改一个变量，这样更容易看清单个因素的影响。
            - 不只看第一名类别，要看完整概率分布，风险信息更充分。
            - 主要用于演示和理解，不用于直接现场决策。
            """,
        ),
        "limits": t(
            """
            **How to close the story responsibly**
            - We clearly state what this demo can prove and what it cannot prove.
            - We list current weak points (data coverage, time/space features, external sources not fully integrated).
            - We provide concrete next steps so the project can continue growing.
            - Honest limitation reporting makes the project more credible in interviews.
            """,
            """
            **如何把故事收好**
            - 明确说清这个Demo能证明什么、不能证明什么。
            - 公开当前不足（数据覆盖、时空特征有限、外部数据还未完全接入）。
            - 给出可执行的下一步，让项目能继续迭代成长。
            - 在面试里，坦诚局限反而更显专业和可信。
            """,
        ),
    }
    st.info(explainers.get(section_key, ""))


def _render_figure_explainer(figure_key: str, t):
    explainers = {
        "model_comparison": t(
            """
            **How to read this chart**
            - X-axis is model name; Y-axis is score.
            - `Accuracy` = overall correctness; `Macro F1` = balanced class performance.
            - If one model has similar Accuracy but higher Macro F1, it is usually safer for minority classes.
            - So this chart helps explain why we selected one model over others.
            """,
            """
            **这张图怎么读**
            - 横轴是模型名，纵轴是得分。
            - `Accuracy` 看整体正确率；`Macro F1` 看各类别是否被公平对待。
            - 如果 Accuracy 差不多但 Macro F1 更高，通常说明对少数类更友好。
            - 所以这张图用于解释“为什么最后选这个模型”。
            """,
        ),
        "confusion_matrix": t(
            """
            **How to read this chart**
            - Diagonal cells are correct predictions; off-diagonal cells are mistakes.
            - We pay special attention to `Fatal -> non-Fatal` mistakes because they are high-risk misses.
            - If one row has many off-diagonal counts, that true class is hard for the model.
            - This tells us which class needs more data or better features.
            """,
            """
            **这张图怎么读**
            - 对角线是预测正确，非对角线是预测错误。
            - 重点看 `Fatal -> 非Fatal`，这是高风险漏判。
            - 某一行非对角线很多，说明模型对该真实类别识别困难。
            - 这张图直接告诉我们该优先补哪类数据和特征。
            """,
        ),
        "feature_importance": t(
            """
            **How to read this chart**
            - Higher value means this feature influences model decisions more.
            - Important reminder: importance is not causality.
            causality means "direct cause", while importance means "useful for prediction".
            - Compare top features with traffic common sense to judge whether results are reasonable.
            - This helps us prioritize feature engineering and data collection.
            """,
            """
            **这张图怎么读**
            - 数值越高，说明该特征对模型决策影响越大。
            - 重要提醒：重要性不等于因果。
            因果是“直接导致”，重要性是“对预测有帮助”。
            - 需要和交通常识对照，看结果是否合理。
            - 这张图可用于确定特征工程和数据补全优先级。
            """,
        ),
    }
    st.caption(explainers.get(figure_key, ""))


def _render_story_transition(section_key: str, t):
    transitions = {
        "overview": t(
            "Next, we move from problem statement to the raw data used to answer it.",
            "下一步我们从“问题定义”进入“原始数据”，看手里到底有什么证据。",
        ),
        "data": t(
            "After confirming available data, we check data quality before any modeling.",
            "确认数据后，下一步先做数据质量检查，再开始建模。",
        ),
        "quality": t(
            "With cleaned data, we can now compare models and evaluate performance.",
            "完成清洗后，进入模型对比，看看不同方法表现如何。",
        ),
        "performance": t(
            "Good scores are not enough, so we continue to reliability tests.",
            "有分数还不够，接下来要验证结果是否稳定可靠。",
        ),
        "reliability": t(
            "Then we inspect wrong predictions to find practical improvement directions.",
            "然后进入误差分析，从错误里找出下一步优化方向。",
        ),
        "error": t(
            "After understanding errors, we use the prediction sandbox for case-based demonstration.",
            "理解误差后，可以在预测沙盒里做案例化演示。",
        ),
        "predict": t(
            "Finally, we summarize limitations and define concrete next steps.",
            "最后回到局限与下一步，形成完整闭环。",
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
        _render_story_transition("overview", t)
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
        _render_story_transition("data", t)
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
        _render_story_transition("quality", t)
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
        _render_story_transition("performance", t)
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
