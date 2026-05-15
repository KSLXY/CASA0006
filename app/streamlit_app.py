from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"


def artifact_path(name: str) -> Path:
    return ARTIFACTS_DIR / name


@st.cache_data
def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_model_payload() -> tuple[dict[str, Any] | None, str | None]:
    if not MODEL_PATH.exists():
        return None, None
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def fmt(value: Any, digits: int = 3, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def t(lang: str, en: str, zh: str) -> str:
    return zh if lang == "中文" else en


def render_metric_row(lang: str, metrics: dict[str, Any] | None) -> None:
    if not metrics:
        st.warning(t(lang, "No public metrics artifact found.", "未找到公开指标文件。"))
        return
    selected_metrics = metrics.get("overall_metrics_on_test_split", {})
    safety_metrics = metrics.get("safety_metrics_on_test_split", {})
    cols = st.columns(5)
    cols[0].metric(t(lang, "Rows", "样本数"), f"{int(metrics.get('rows_total', 0)):,}")
    cols[1].metric(t(lang, "Model", "模型"), metrics.get("selected_model", "N/A"))
    cols[2].metric(t(lang, "Accuracy", "准确率"), fmt(selected_metrics.get("accuracy")))
    cols[3].metric(t(lang, "Macro F1", "宏平均F1"), fmt(selected_metrics.get("f1_macro")))
    cols[4].metric(t(lang, "Fatal Recall", "Fatal召回率"), fmt(safety_metrics.get("fatal_recall")))


def render_explainer(lang: str, title_en: str, title_zh: str, body_en: str, body_zh: str) -> None:
    st.markdown(f"### {t(lang, title_en, title_zh)}")
    st.markdown(t(lang, body_en, body_zh))


def render_json_expander(lang: str, label_en: str, label_zh: str, data: Any) -> None:
    if data is None or data == {}:
        return
    with st.expander(t(lang, label_en, label_zh), expanded=False):
        st.json(data)


def safe_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def make_public_observations(metrics: dict[str, Any] | None, lang: str) -> list[str]:
    if not metrics:
        return []
    observations = metrics.get("error_analysis_observations", [])
    if lang == "English":
        return observations
    mapping = {
        "Error cases are concentrated in": "误差样本集中在",
        "Most variable features among errors are": "误差样本中波动较大的特征包括",
        "Fatal vs serious boundary remains the highest-risk confusion zone.": "Fatal 与 Serious 的边界仍是高风险混淆区域。",
    }
    translated: list[str] = []
    for item in observations:
        text = item
        for en, zh in mapping.items():
            if text.startswith(en):
                text = text.replace(en, zh)
        translated.append(text)
    return translated


def main() -> None:
    st.set_page_config(
        page_title="Road Collision Severity Risk Modeling",
        page_icon=":bar_chart:",
        layout="wide",
    )

    lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"], index=0)

    metrics = load_json(artifact_path("metrics.json"))
    metrics_cv = load_json(artifact_path("metrics_cv.json"))
    data_quality = load_json(artifact_path("data_quality_report.json"))
    leakage = load_json(artifact_path("leakage_check_report.json"))
    calibration = load_json(artifact_path("calibration_report.json"))
    model_compare = load_csv(artifact_path("model_compare.csv"))
    feature_importance = load_csv(artifact_path("feature_importance.csv"))
    permutation_importance = load_csv(artifact_path("permutation_importance.csv"))
    threshold_report = load_csv(artifact_path("threshold_report.csv"))
    missingness_by_time = load_csv(artifact_path("missingness_by_time.csv"))
    ablation = load_csv(artifact_path("ablation_leakage.csv"))
    payload, model_error = load_model_payload()

    st.title(t(lang, "Road Collision Severity Risk Modeling", "交通事故严重度风险建模"))
    st.caption(
        t(
            lang,
            "A recruiter-friendly applied ML case study with research-grade reliability checks.",
            "面向求职展示的应用型机器学习案例，同时保留研究级可靠性检验。",
        )
    )
    render_metric_row(lang, metrics if isinstance(metrics, dict) else None)

    tabs = st.tabs(
        [
            t(lang, "Brief", "项目概览"),
            t(lang, "Data & Features", "数据与特征"),
            t(lang, "Model Evidence", "模型证据"),
            t(lang, "Reliability & Risk", "可靠性与风险"),
            t(lang, "Research Notes", "研究说明"),
        ]
    )

    with tabs[0]:
        render_explainer(
            lang,
            "Project in 60 seconds",
            "一分钟读懂项目",
            """
            This project predicts whether a road collision is likely to be `Fatal`, `Serious`, or `Slight` using pre-event road,
            temporal, and environment signals from STATS19-style data. The goal is not to claim a deployable safety product, but
            to show an end-to-end workflow: data governance, feature policy, model comparison, reliability checks, and honest limitations.
            """,
            """
            本项目使用事故发生前可获得的道路、时间与环境信息，预测交通事故严重度等级：`Fatal`、`Serious`、`Slight`。
            它不是一个可直接部署的公共安全系统，而是一个完整工作流展示：数据治理、特征策略、模型比较、可靠性检验和诚实局限。
            """,
        )
        if isinstance(metrics, dict):
            selected = metrics.get("selected_model", "N/A")
            screening = metrics.get("fatal_screening_model", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(t(lang, "#### What I built", "#### 我构建了什么"))
                st.markdown(
                    t(
                        lang,
                        """
                        - reproducible data-build and training commands
                        - pre-event feature policy to reduce leakage
                        - model comparison across interpretable and tree-based models
                        - validation evidence across resampling, time, and geography
                        - public-release audit to avoid data/model leakage on GitHub
                        """,
                        """
                        - 可复现的数据构建与训练命令
                        - 默认使用事故发生前特征，降低信息泄漏风险
                        - 比较可解释基线模型与树模型
                        - 通过重采样、时间与空间切分验证稳定性
                        - 发布审计脚本，避免数据和模型泄漏到 GitHub
                        """,
                    )
                )
            with c2:
                st.markdown(t(lang, "#### Current evidence", "#### 当前证据"))
                st.markdown(
                    t(
                        lang,
                        f"""
                        - Balanced research model: `{selected}`
                        - Fatal screening model: `{screening.get('model_name', 'N/A')}`
                        - Public artifact tag: `{metrics.get('data_version_tag', 'unknown')}`
                        - Feature set: `{metrics.get('feature_set_mode', 'unknown')}`
                        """,
                        f"""
                        - 均衡研究模型：`{selected}`
                        - Fatal 筛查模型：`{screening.get('model_name', 'N/A')}`
                        - 公开产物标签：`{metrics.get('data_version_tag', 'unknown')}`
                        - 特征集合：`{metrics.get('feature_set_mode', 'unknown')}`
                        """,
                    )
                )
        st.info(
            t(
                lang,
                "Read this as baseline decision-support research, not as a finished traffic-safety deployment.",
                "请将本项目理解为基线级决策支持研究，而不是已经完成的交通安全部署系统。",
            )
        )

    with tabs[1]:
        render_explainer(
            lang,
            "Data scope and feature policy",
            "数据范围与特征策略",
            """
            The core modeling table comes from UK DfT STATS19 collision, vehicle, and casualty records. London weather is only a
            limited external enrichment source and does not define the study area. The default model uses pre-event features so the
            task remains closer to prospective risk screening.
            """,
            """
            核心建模表来自英国 DfT STATS19 的事故、车辆与伤亡记录。伦敦天气只是覆盖有限的外部补充来源，并不定义研究空间范围。
            默认模型使用事故发生前可获得的特征，使任务更接近前瞻性风险筛查。
            """,
        )
        st.markdown(t(lang, "#### Feature groups", "#### 特征分组"))
        feature_groups = pd.DataFrame(
            [
                {"group": "Temporal", "examples": "hour, day_of_week, month, is_weekend, season, hour_peak"},
                {"group": "Road context", "examples": "road_type, speed_limit, junction_detail, junction_control"},
                {"group": "Environment", "examples": "light_conditions, weather_conditions, road_surface_conditions"},
                {"group": "Weather enrichment", "examples": "precipitation, pressure, temperature, cloud_cover, sunshine"},
                {"group": "Derived interactions", "examples": "precipitation_peak_interaction, low_visibility_proxy"},
            ]
        )
        st.dataframe(feature_groups, use_container_width=True, hide_index=True)
        if isinstance(metrics, dict):
            missing_rate = metrics.get("missing_rate_by_feature", {})
            if missing_rate:
                missing_df = (
                    pd.DataFrame({"feature": list(missing_rate), "missing_rate": list(missing_rate.values())})
                    .sort_values("missing_rate", ascending=False)
                    .reset_index(drop=True)
                )
                st.markdown(t(lang, "#### Missingness snapshot", "#### 缺失情况快照"))
                st.dataframe(missing_df.head(12), use_container_width=True)
        if not missingness_by_time.empty:
            with st.expander(t(lang, "Monthly missingness details", "按月缺失率明细")):
                st.dataframe(missingness_by_time.head(120), use_container_width=True)
        render_json_expander(lang, "Data quality artifact", "数据质量产物", data_quality)

    with tabs[2]:
        render_explainer(
            lang,
            "Model evidence, not leaderboard chasing",
            "模型证据，而不是刷榜",
            """
            The project reports accuracy, Macro F1, balanced accuracy, Fatal recall, feature influence, and permutation importance.
            Macro F1 matters because severity classes are imbalanced. Fatal recall is discussed separately because high-severity misses
            have a larger practical cost.
            """,
            """
            项目报告准确率、宏平均 F1、平衡准确率、Fatal 召回率、特征影响和置换重要性。宏平均 F1 用于避免多数类掩盖少数类。
            Fatal 召回率单独讨论，因为高严重度漏判具有更高现实成本。
            """,
        )
        if not model_compare.empty:
            st.markdown(t(lang, "#### Model comparison", "#### 模型对比"))
            st.dataframe(model_compare, use_container_width=True)
            chart_cols = [c for c in ["accuracy", "f1_macro", "balanced_accuracy", "fatal_recall"] if c in model_compare.columns]
            if chart_cols:
                st.bar_chart(model_compare.set_index("model_name")[chart_cols], use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            safe_image(FIGURES_DIR / "model_comparison.png", t(lang, "Model comparison", "模型对比"))
            safe_image(FIGURES_DIR / "confusion_matrix.png", t(lang, "Confusion matrix", "混淆矩阵"))
        with col2:
            safe_image(FIGURES_DIR / "feature_importance.png", t(lang, "Feature influence", "特征影响"))
            if not feature_importance.empty:
                st.markdown(t(lang, "#### Top feature influence", "#### 主要特征影响"))
                st.dataframe(feature_importance.head(10), use_container_width=True)
        if not permutation_importance.empty:
            with st.expander(t(lang, "Permutation importance evidence", "置换重要性证据")):
                st.dataframe(permutation_importance.head(15), use_container_width=True)
        if not ablation.empty:
            with st.expander(t(lang, "Leakage ablation comparison", "泄漏对照实验")):
                st.dataframe(ablation, use_container_width=True)

    with tabs[3]:
        render_explainer(
            lang,
            "Reliability and risk checks",
            "可靠性与风险检验",
            """
            A model that performs well on one split can still fail under future time periods, different places, or different thresholds.
            This tab keeps those risks visible instead of hiding them behind a single headline score.
            """,
            """
            一个模型在单次切分上表现不错，并不代表它能稳定适应未来时间、不同空间或不同阈值。本页将这些风险显式展示，
            而不是只给一个总分。
            """,
        )
        if isinstance(metrics_cv, dict):
            cv_rows = metrics_cv.get("cv", {}).get("rows", [])
            if cv_rows:
                st.markdown(t(lang, "#### Stratified K-fold", "#### 分层 K 折"))
                st.dataframe(pd.DataFrame(cv_rows), use_container_width=True)
            time_holdout = metrics_cv.get("time_holdout", {})
            spatial_holdout = metrics_cv.get("spatial_holdout", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(t(lang, "#### Time holdout", "#### 时间外推"))
                st.json(time_holdout)
            with c2:
                st.markdown(t(lang, "#### Spatial holdout", "#### 空间外推"))
                st.json(spatial_holdout)
        if not threshold_report.empty:
            st.markdown(t(lang, "#### Fatal threshold sensitivity", "#### Fatal 阈值敏感性"))
            st.dataframe(threshold_report, use_container_width=True)
        render_json_expander(lang, "Calibration report", "校准报告", calibration)
        render_json_expander(lang, "Leakage check", "泄漏检查", leakage)

    with tabs[4]:
        render_explainer(
            lang,
            "Research notes and future direction",
            "研究说明与后续方向",
            """
            The strongest version of this project is not a claim that the current model is finished. It is a transparent research
            foundation for road-safety risk modeling: reproducible data, explicit leakage control, validation evidence, and a clear
            roadmap for stronger spatial, exposure, and causal analysis.
            """,
            """
            本项目最有价值的地方不是宣称当前模型已经完成，而是提供一个透明的交通安全风险建模研究基础：可复现数据、
            明确泄漏控制、可靠性证据，以及面向空间、暴露量和因果分析的后续路线。
            """,
        )
        st.markdown(t(lang, "#### What this is not", "#### 本项目不是什么"))
        st.markdown(
            t(
                lang,
                """
                - not causal inference
                - not a deployment-ready public-safety system
                - not a complete weather-fusion or road-network exposure model
                - not a repository containing all generated data and model binaries
                """,
                """
                - 不是因果推断研究
                - 不是可直接部署的公共安全系统
                - 不是完整天气融合或路网暴露量模型
                - 不是包含全部生成数据和模型二进制文件的仓库
                """,
            )
        )
        st.markdown(t(lang, "#### Public error insights", "#### 公开误差观察"))
        observations = make_public_observations(metrics if isinstance(metrics, dict) else None, lang)
        if observations:
            for item in observations:
                st.write(f"- {item}")
        else:
            st.write(t(lang, "No public aggregate error observations available.", "暂无公开聚合误差观察。"))
        st.markdown(t(lang, "#### Research extensions", "#### 研究扩展方向"))
        st.markdown(
            t(
                lang,
                """
                - exposure-adjusted severity risk
                - stronger spatial validation across road-network groups
                - richer official road-infrastructure features
                - geography and equity analysis of model errors
                - causal designs for policy interventions where data permits
                - uncertainty-aware threshold selection for high-severity screening
                """,
                """
                - 加入暴露量校正后的严重度风险建模
                - 按道路网络或地区做更强的空间验证
                - 引入更丰富的官方道路基础设施特征
                - 分析模型误差的地理与公平性差异
                - 在数据允许时设计政策干预相关的因果研究
                - 面向高严重度筛查的不确定性阈值选择
                """,
            )
        )
        if model_error:
            st.warning(t(lang, f"Local model exists but could not be loaded: {model_error}", f"本地模型存在但无法加载：{model_error}"))
        elif payload is None:
            st.info(
                t(
                    lang,
                    "Single-row prediction is local-only because the trained model is intentionally excluded from GitHub.",
                    "单条预测仅限本地运行，因为训练好的模型按公开发布策略不会上传到 GitHub。",
                )
            )
        else:
            with st.expander(t(lang, "Optional local single-row prediction", "可选：本地单条预测")):
                feature_columns = payload["feature_columns"]
                defaults = payload.get("feature_defaults", {})
                class_labels = payload.get("class_labels", ["Fatal", "Serious", "Slight"])
                values: dict[str, Any] = {}
                for col in feature_columns:
                    default = defaults.get(col, 0)
                    if isinstance(default, str):
                        values[col] = st.text_input(col, value=default)
                    else:
                        values[col] = st.number_input(col, value=float(default), format="%.4f")
                if st.button(t(lang, "Predict severity", "预测严重度")):
                    pipeline = payload["pipeline"]
                    X = pd.DataFrame([values], columns=feature_columns)
                    pred_idx = int(pipeline.predict(X)[0])
                    probs = pipeline.predict_proba(X)[0]
                    st.success(t(lang, f"Predicted class: {class_labels[pred_idx]}", f"预测类别：{class_labels[pred_idx]}"))
                    st.dataframe(pd.DataFrame({"class": class_labels, "probability": probs}), use_container_width=True)


if __name__ == "__main__":
    main()
