# DfT/STATS19 Road Collision Severity Classification Research

This repository focuses on a full-table 2020-2024 STATS19 accident severity classification study with a reproducible pipeline and evidence-oriented outputs.

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
- Complete processed master table: `data/processed/processed_master.parquet`
- Planned enrichments: OSM road attributes, air quality, IMD

The public repository does not include raw data, processed data, trained models, or row-level diagnostic outputs. These files are regenerated locally from the commands below and are excluded from Git to keep the repository small and privacy-safe.

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
  - Balanced HistGradientBoosting
- STATS19 road and environment fields are modeled as categorical predictors with `ColumnTransformer` and `OneHotEncoder`.
- Two model roles are reported:
  - `balanced_research_model` for Macro F1-oriented model comparison
  - `fatal_screening_model` for Fatal recall-oriented safety screening
- Reliability is evaluated with Stratified K-Fold and time-based holdout.
- Spatial group holdout, safety-threshold analysis, probability calibration, and permutation importance are generated as reliability evidence.

## 6) Outputs and Evidence
`data/processed/processed_master.parquet` is the only supported research dataset. Training fails if the master table is missing or has fewer than 100,000 rows, preventing accidental use of old demonstration data.

Current processed-table caveats:
- External weather join coverage is about 17.75%, so STATS19 native weather and road-condition fields remain important evidence.
- OSM, air-quality, and IMD columns are placeholders until reliable enrichment is implemented.

Core artifacts:
- `artifacts/metrics.json`
- `artifacts/metrics_cv.json`
- `artifacts/model_compare.csv`
- `artifacts/feature_importance.csv`
- `artifacts/permutation_importance.csv`
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

Local-only artifacts:
- `artifacts/model.joblib`
- `artifacts/error_cases.csv`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/sample/`

## 7) Boundaries and Next Iteration
Current boundary is a publicly available STATS19 full-master-table severity classification pipeline for risk estimation support and research analysis. London weather is retained only as a partially matched external enrichment source, not as the study-area definition.

Priority next steps:
- improve external data fusion quality,
- add richer spatial exposure and road-network features,
- compare safety-screening thresholds with domain-specific false-positive constraints.

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

## 10) Public Release Hygiene
Before any GitHub sync, run:
```bash
python3 scripts/publication_audit.py
pytest
git status --short --branch
```

The project-local release skill is stored at `.codex/skills/casa0006-public-release/SKILL.md`. It must be followed before staging, committing, pushing, or publishing this repository.

This repository uses the MIT License. If a trained model needs to be distributed later, publish it through GitHub Releases or external storage rather than regular Git.

---

# 中文版

## 1) 问题定义
本项目基于 2020-2024 年完整 STATS19 主表，将事故严重度作为结构化风险信号进行预测，而非只做事故数量统计。  
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
- 完整处理主表：`data/processed/processed_master.parquet`
- 计划补全：OSM 路网属性、空气质量、IMD

公开仓库不包含原始数据、处理后主表、训练模型或逐行误差诊断文件。这些内容通过下方命令在本地重新生成，并通过 `.gitignore` 排除，避免仓库体积过大和隐私泄漏风险。

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
- 使用 `ColumnTransformer` 和 `OneHotEncoder` 将 STATS19 道路与环境字段作为类别特征纳入模型。
- 同时报告两类模型角色：
  - `balanced_research_model` 面向 Macro F1 选模，
  - `fatal_screening_model` 面向 Fatal 召回率筛查。
- 对比模型为 Logistic Regression、Random Forest、Balanced HistGradientBoosting。
- 可靠性验证采用 Stratified K-Fold 与时间外推切分。
- 额外生成空间分组外推、安全阈值、概率校准与置换重要性证据。

## 6) 结果产物与证据
`data/processed/processed_master.parquet` 是唯一支持的研究数据集。若主表缺失或低于 100,000 行，训练会直接失败，避免误用旧演示数据。

当前主表注意事项：
- 外部天气合并命中率约为 17.75%，因此 STATS19 原生天气与道路条件字段仍是重要证据。
- OSM、空气质量与 IMD 字段仍为占位列，可靠补全前不作为正式模型证据。

核心产物：
- `artifacts/metrics.json`
- `artifacts/metrics_cv.json`
- `artifacts/model_compare.csv`
- `artifacts/feature_importance.csv`
- `artifacts/permutation_importance.csv`
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

仅本地保留的产物：
- `artifacts/model.joblib`
- `artifacts/error_cases.csv`
- `data/raw/`
- `data/interim/`
- `data/processed/`
- `data/sample/`

## 7) 边界与下一步
当前边界为基于公开 STATS19 完整主表的事故严重度分类研究流程，定位是风险估计支持与研究分析。伦敦天气数据仅作为部分匹配的外部补充来源保留，不再定义研究空间范围。

下一步优先方向：
- 提升外部数据融合质量，
- 增加更强的空间暴露与路网特征，
- 结合领域约束比较安全筛查阈值。

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

## 9) 公开发布检查
同步到 GitHub 前必须运行：
```bash
python3 scripts/publication_audit.py
pytest
git status --short --branch
```

本项目的发布 skill 位于 `.codex/skills/casa0006-public-release/SKILL.md`。任何暂存、提交、推送或公开发布前都应遵循该流程。

本仓库采用 MIT License。若后续需要发布训练好的模型，应使用 GitHub Releases 或外部存储，不应直接放入普通 Git 历史。
