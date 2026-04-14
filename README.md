# London Road Collision Severity Project (CASA0006 Upgrade)

## EN
This repository upgrades `casa0006_individual_work.ipynb` into a portfolio-ready data science project.

### Problem
Predict collision severity (`1=slight`, `2=serious`, `3=fatal`) from weather and collision context.

### What was upgraded
- Converted one coursework notebook into a reproducible project structure.
- Added data pipeline with explicit invalid-label handling (`-10` and other non-1/2/3 targets are filtered).
- Added model comparison (`LogisticRegression`, `RandomForest`, `HistGradientBoosting`) with automatic selection by Macro F1.
- Added Streamlit app for interactive demo and portfolio sharing.
- Added tests and documented run commands.

### Project structure
```text
app/                 # Streamlit application
configs/             # YAML config
data/sample/         # Small runnable sample dataset
examples/            # Prediction input examples
scripts/             # Full data download / merge scripts
src/                 # Training, evaluation, prediction, pipeline
tests/               # Unit + integration-style tests
artifacts/           # Model and metrics outputs
reports/figures/     # Visual outputs
```

### Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.predict --config configs/default.yaml --input-file examples/sample_input.json
streamlit run app/streamlit_app.py
```

### Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create app:
   - Repository: your repo
   - Branch: `main`
   - Main file: `app/streamlit_app.py`
4. Deploy and share app URL in CV.

### Data notes
- `data/sample/merged_sample.csv` is for reproducible demo.
- For full data, run:
```bash
python scripts/download_full_data.py --output data/raw/merged_full.csv --start-year 2010
```
Requires Kaggle authentication for weather dataset download.

---

## 中文
这个仓库把你的课程作业 `casa0006_individual_work.ipynb` 升级成了一个可放进简历的数据科学项目。

### 项目目标
基于天气与交通碰撞特征，预测事故严重程度（`1=轻微`，`2=严重`，`3=致命`）。

### 核心升级点
- 从单 notebook 升级为可复现的项目结构。
- 明确处理脏标签：目标列中 `-10` 等非法值会被过滤并计数。
- 引入三模型对比并按 `Macro F1` 自动选择最佳模型。
- 增加 Streamlit 在线交互页面，便于招聘方直接打开查看。
- 增加测试与标准化运行命令。

### 快速运行
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.predict --config configs/default.yaml --input-file examples/sample_input.json
streamlit run app/streamlit_app.py
```

### 上线部署
1. 将仓库推送到 GitHub。
2. 使用 GitHub 登录 Streamlit Community Cloud。
3. 选择入口文件 `app/streamlit_app.py` 并部署。
4. 把生成的公开链接放进简历项目描述。

