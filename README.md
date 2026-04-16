# London Road Collision Severity Classification Research

This repository focuses on a London-centered accident severity classification study with a reproducible pipeline and evidence-oriented outputs.

## 1) Problem Definition
Road-collision severity is treated as a structured safety signal rather than a simple accident count.  
The project predicts three severity levels:
- `1 = Fatal`
- `2 = Serious`
- `3 = Slight`

## 2) Research Objective
The workflow is designed to:
- govern and validate multi-source data,
- compare models under a consistent protocol,
- inspect model errors with traceable evidence,
- and maintain reproducible outputs for review and iteration.

## 3) Data Sources
- UK DfT road safety records (collision, vehicle, casualty)
- London weather dataset (Kaggle public data)
- UK bank holidays (GOV.UK API)
- Reproducible sample dataset: `data/sample/merged_sample.csv`
- Planned enrichments: OSM road attributes, London air quality, IMD

Dataset build commands:
```bash
python scripts/fetch_datasets.py --config configs/data.yaml --from 2015-01-01 --to 2024-12-31
python scripts/build_master_table.py --config configs/data.yaml
```

## 4) Data Quality Governance
The target label is strictly constrained to `{1, 2, 3}`.  
Invalid labels such as `-10` are removed before training, and removal counts are written into artifacts for auditability.

## 5) Method and Modeling Decisions
- Three-class formulation is kept to preserve safety-relevant granularity.
- Macro F1 is used for model selection to avoid majority-class dominance.
- Pre-event features are the default training set to control leakage risk.
- Models compared under the same interface:
  - Logistic Regression
  - Random Forest
  - HistGradientBoosting
- Reliability is evaluated with Stratified K-Fold and time-based holdout.

## 6) Outputs and Evidence
If `data/processed/processed_master.parquet` exists, it is used as the default training source.  
Otherwise the pipeline falls back to `data/sample/merged_sample.csv`.

Core artifacts:
- `artifacts/metrics.json`
- `artifacts/metrics_cv.json`
- `artifacts/model_compare.csv`
- `artifacts/error_cases.csv`
- `artifacts/feature_importance.csv`
- `artifacts/data_quality_report.json`
- `artifacts/leakage_check_report.json`
- `artifacts/threshold_report.csv`
- `artifacts/calibration_report.json`
- `artifacts/hyperparameter_search.json`
- `artifacts/ablation_leakage.csv`
- `artifacts/missingness_by_time.csv`
- `reports/figures/model_comparison.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`

## 7) Boundaries and Next Iteration
Current boundary is a London-focused, publicly available-data severity classification pipeline for risk estimation support and research analysis.

Priority next steps:
- train on larger processed master data,
- improve external data fusion quality,
- strengthen calibration and class-imbalance handling.

## 8) Reproducible Command Chain
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_datasets.py --config configs/data.yaml --from 2015-01-01 --to 2024-12-31
python scripts/build_master_table.py --config configs/data.yaml
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.predict --config configs/default.yaml --input-file examples/sample_input.json
```

## 9) Public Interfaces
- Train: `python -m src.train --config configs/default.yaml`
- Evaluate: `python -m src.evaluate --config configs/default.yaml`
- Predict: `python -m src.predict --config configs/default.yaml --input-file examples/sample_input.json`
- Fetch data: `python scripts/fetch_datasets.py --config configs/data.yaml --from 2015-01-01 --to 2024-12-31`
- Build master: `python scripts/build_master_table.py --config configs/data.yaml`

---

# 中文版

## 1) 问题定义
本项目聚焦伦敦道路交通安全场景，将事故严重度作为结构化风险信号进行预测，而非只做事故数量统计。  
标签定义为三分类：
- `1 = 致命`
- `2 = 严重`
- `3 = 轻微`

## 2) 研究目标
项目主流程用于实现以下目标：
- 多源数据治理与标签合法性校验，
- 统一评估口径下的模型对比，
- 可追溯误差分析与证据输出，
- 可复现训练评估链路。

## 3) 数据来源
- 英国 DfT 道路安全数据（collision, vehicle, casualty）
- 伦敦天气数据（Kaggle 公开数据）
- 英国法定节假日（GOV.UK API）
- 可复现样本：`data/sample/merged_sample.csv`
- 计划补全：OSM 路网属性、伦敦空气质量、IMD

数据构建命令：
```bash
python scripts/fetch_datasets.py --config configs/data.yaml --from 2015-01-01 --to 2024-12-31
python scripts/build_master_table.py --config configs/data.yaml
```

## 4) 数据质量治理
目标标签严格限制在 `{1,2,3}`。  
`-10` 等异常编码在训练前剔除，剔除数量写入 artifact，确保过程可审计。

## 5) 方法与建模决策
- 保留三分类以保持安全语义粒度。
- 采用 Macro F1 做模型选型，降低多数类掩盖风险。
- 默认使用 pre-event 特征集合，控制后验信息泄漏。
- 对比模型为 Logistic Regression、Random Forest、HistGradientBoosting。
- 可靠性验证采用 Stratified K-Fold 与时间外推切分。

## 6) 结果产物与证据
若存在 `data/processed/processed_master.parquet`，训练默认使用主表。  
若主表不可用，流程自动回退至 `data/sample/merged_sample.csv`。

核心产物：
- `artifacts/metrics.json`
- `artifacts/metrics_cv.json`
- `artifacts/model_compare.csv`
- `artifacts/error_cases.csv`
- `artifacts/feature_importance.csv`
- `artifacts/data_quality_report.json`
- `artifacts/leakage_check_report.json`
- `artifacts/threshold_report.csv`
- `artifacts/calibration_report.json`
- `artifacts/hyperparameter_search.json`
- `artifacts/ablation_leakage.csv`
- `artifacts/missingness_by_time.csv`
- `reports/figures/model_comparison.png`
- `reports/figures/confusion_matrix.png`
- `reports/figures/feature_importance.png`

## 7) 边界与下一步
当前边界为伦敦范围内、基于公开数据的事故严重度分类研究流程，定位是风险估计支持与研究分析。

下一步优先方向：
- 在更大规模主表上训练，
- 提升外部数据融合质量，
- 强化校准与类别不平衡处理。

## 8) 复现命令链
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_datasets.py --config configs/data.yaml --from 2015-01-01 --to 2024-12-31
python scripts/build_master_table.py --config configs/data.yaml
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.predict --config configs/default.yaml --input-file examples/sample_input.json
```
