# Interview Story Pack (Classification Mainline)

## 30-Second Elevator Version
I built a London road-collision severity classifier that predicts slight/serious/fatal outcomes using weather and collision context.  
The key contribution is not only model training, but data quality governance: I detect and remove invalid target codes (e.g. `-10`), compare multiple models by Macro F1, and deploy the whole evidence chain on Streamlit.

## 2-Minute Project Version
This project started from coursework and was rebuilt into a reproducible ML pipeline.  
I merged weather and collision data, handled invalid labels explicitly, and engineered time/interaction features such as peak-hour and precipitation-peak interaction.  
I compared Logistic Regression, Random Forest, and HistGradientBoosting, selected by Macro F1 to avoid minority-class blindness, and exported model comparison, confusion matrix, feature influence, and error-case artifacts.  
For reliability, I added stratified K-fold and time-based holdout checks and reported fatal recall separately.  
The app is online in Streamlit and supports both narrative review and single-case prediction.

## 5-Minute Deep-Dive Version

## 0:00 - 0:40 | Problem and value
I worked on a London road-safety classification task: predict accident severity (`slight`, `serious`, `fatal`) from weather and collision context.  
I wanted to turn a coursework notebook into a deployable, reproducible project that can be reviewed by recruiters directly.

## 0:40 - 1:40 | Data and cleaning decisions
I combined weather and road-collision data and found a key data quality issue: invalid labels such as `-10` in severity fields.  
I enforced a strict target space `{1,2,3}`, filtered invalid labels, and logged how many rows were removed.  
This was important because model quality is not trustworthy if label quality is not controlled.

## 1:40 - 3:00 | Model comparison and selection logic
I trained three models: Logistic Regression, Random Forest, and HistGradientBoosting.  
I selected by **Macro F1** instead of only Accuracy, because macro averaging protects minority classes like `fatal`.  
The project exports model comparison tables and figures so the decision is auditable.

## 3:00 - 4:00 | Error analysis and limitations
I generated an error-cases table with actual vs predicted labels and key weather features.  
When error rows are very few on a small sample, I explicitly mark results as `sample demo performance`, not final policy evidence.  
Main limitation: dataset scale and feature richness are still limited; this can inflate apparent performance.

## 4:00 - 5:00 | Reproducibility and deployment
The pipeline is modular (`train/evaluate/predict`) and deploys in Streamlit Cloud.  
Recruiters can open the app, inspect data quality metrics, model metrics, confusion matrix, feature influence, reliability pages, and run single-case predictions.  
So the project demonstrates not only modeling, but also engineering hygiene and communication.

---

# 8 High-Frequency Q&A Templates

## Q1: Why not deep learning?
For this tabular dataset size, tree-based and linear models are more data-efficient and easier to interpret. I prioritized reliable baselines first.

## Q2: Why Macro F1 instead of only Accuracy?
Accuracy can hide minority-class failure. Macro F1 gives each class equal weight, which is better for safety-oriented severity classes.

## Q3: How did you avoid data leakage?
I split train/test first, then fit model pipeline only on training data and evaluate on held-out data.

## Q4: Why keep 3 classes instead of binary?
Three classes preserve decision value. Safety interventions differ for serious vs fatal risk patterns.

## Q5: What did you do with invalid labels like `-10`?
I treated them as invalid targets, removed them before training, and recorded removed counts in metrics artifacts.

## Q6: Why trust model choice?
I compared multiple models under the same split and feature set, and selected by a defined metric (Macro F1), not by intuition.

## Q7: Why are metrics very high in demo?
This repository uses a small sample for reproducibility. I explicitly label outputs as sample demo performance and avoid over-claiming.

## Q8: If you had more time, what next?
I would run full-data experiments, add temporal/spatial features, and validate stability with cross-validation and imbalance handling.
