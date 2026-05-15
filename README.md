# Road Collision Severity Risk Modeling

Applied machine-learning case study using UK DfT STATS19 road-collision data to classify collision severity from **pre-event road, temporal, and environment signals**. The project is designed as a public GitHub portfolio project: easy to understand first, but still rigorous enough to show research judgment.

## Project In 60 Seconds

Road-safety teams care not only about how many collisions happen, but also about which conditions are associated with more severe outcomes. This project builds a reproducible severity-classification pipeline for three STATS19 labels:

- `Fatal`
- `Serious`
- `Slight`

What I built:

- a data pipeline for DfT collision, vehicle, casualty, weather, and holiday data,
- a strict feature policy that defaults to **pre-event** features to reduce leakage risk,
- a model-comparison workflow using Logistic Regression, Random Forest, and balanced HistGradientBoosting,
- reliability evidence through cross-validation, time holdout, spatial holdout, threshold analysis, calibration, and leakage checks,
- a Streamlit interface for explaining the evidence without exposing private/generated artifacts.

Current full-data run:

- Rows used: `503,475`
- Selected balanced research model: `random_forest`
- Test accuracy: `0.654`
- Test Macro F1: `0.381`
- Test balanced accuracy: `0.398`
- Fatal screening model: `hist_gradient_boosting_balanced`
- Fatal screening recall: `0.630`

These results should be read as **baseline decision-support evidence**, not as a deployment-ready safety system.

## Why This Matters

This project demonstrates the kind of ML work that is useful in real public-sector analytics:

- define a safety-relevant target,
- avoid answer-like post-event features,
- keep class imbalance visible with Macro F1 and Fatal recall,
- test whether results are stable across time and geography,
- explain where the model is weak before making claims.

The project is predictive rather than causal. It can support risk exploration and method development, but it does not prove that any feature directly causes collision severity.

## Data Scope

Core data comes from UK DfT STATS19 road-safety records. The modeling table is built locally from collision, vehicle, and casualty records, with optional weather and bank-holiday enrichment.

Important boundary:

- The core STATS19 data has Great Britain coverage.
- London weather is used only as a limited external enrichment source.
- London weather is **not** the study-area definition.
- OSM, air-quality, and IMD enrichments are currently placeholders, not formal model evidence.

The public repository does not include raw data, processed data, trained models, or row-level diagnostic outputs. These files are regenerated locally and excluded from Git.

## Method

The default modeling configuration uses pre-event signals:

- temporal features: hour, day of week, month, weekend, season, peak-hour marker,
- road/context features: road type, speed limit, junction detail, junction control, lighting, weather condition, road surface condition, urban/rural area,
- weather-derived features: precipitation, temperature, pressure, cloud cover, sunshine, low-visibility proxy,
- interaction features: precipitation during peak-hour periods.

The model-selection metric is Macro F1 because severity classes are imbalanced and minority outcomes must remain visible. Accuracy is still reported, but it is not enough on its own.

## Evidence

The project keeps lightweight public evidence in `artifacts/` and `reports/figures/`:

- `artifacts/metrics.json`
- `artifacts/metrics_cv.json`
- `artifacts/model_compare.csv`
- `artifacts/feature_importance.csv`
- `artifacts/permutation_importance.csv`
- `artifacts/data_quality_report.json`
- `artifacts/leakage_check_report.json`
- `artifacts/threshold_report.csv`
- `artifacts/calibration_report.json`
- `artifacts/ablation_leakage.csv`
- `artifacts/missingness_by_time.csv`
- `reports/figures/model_comparison.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`

Local-only artifacts:

- `artifacts/model.joblib`
- `artifacts/error_cases.csv`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/sample/`

## What This Is Not

This project is not:

- a causal study of collision mechanisms,
- a deployment-ready public-safety decision system,
- a complete road-network exposure model,
- a complete weather-fusion study,
- a repository containing all raw or processed data.

The value is in the reproducible workflow, modeling discipline, reliability checks, and transparent communication of limitations.

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/fetch_datasets.py --config configs/data.yaml --from 2015-01-01 --to 2024-12-31
python3 scripts/build_master_table.py --config configs/data.yaml
python3 -m src.train --config configs/default.yaml
python3 -m src.evaluate --config configs/default.yaml
python3 -m src.predict --config configs/default.yaml --input-file examples/sample_input.json
```

Launch the interface:

```bash
python3 -m streamlit run app/streamlit_app.py
```

Public-release check:

```bash
python3 scripts/publication_audit.py
python3 -m pytest
git status --short --branch
```

## Research Extensions

This project can become stronger as research work by adding:

- exposure-adjusted severity risk instead of collision-only classification,
- stronger spatial validation across local authorities or road-network groups,
- richer road-network features from OSM or official road infrastructure data,
- equity/geography analysis of model errors,
- causal designs for policy interventions where data permits,
- uncertainty-aware threshold selection for high-severity screening.

## 中文简介

本项目是一个面向公开展示的交通事故严重度建模案例。核心目标是基于英国 DfT STATS19 数据，使用事故发生前可获得的道路、时间与环境信息，预测事故严重度等级，并用交叉验证、时间外推、空间外推、阈值分析、校准和泄漏检查来说明结果可信度。

项目优先服务于求职作品集展示：读者能快速理解问题、数据、方法、结果和局限。同时保留研究严谨性，为未来博士申请或进一步研究扩展打基础。

本仓库不上传原始数据、处理后主表、训练模型或逐行误差文件。公开仓库只保留代码、配置、测试、聚合指标、图表和说明文档。

## License

MIT License.
