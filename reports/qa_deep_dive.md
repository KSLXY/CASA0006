# QA Deep Dive (Technical Interview)

## Q1: How do you prove data quality is under control?
I enforce valid target labels (`1/2/3`) and log the removed invalid count in artifacts.  
I also export feature-level missing rates before imputation, so data quality is measurable and reproducible.

## Q2: Why Macro F1 and not only Accuracy?
Severity classes are imbalanced and safety-sensitive. Macro F1 prevents dominant classes from hiding minority failure.

## Q3: How do you evaluate reliability beyond one train/test split?
I run both stratified K-fold and time-based holdout. The first checks variance robustness, the second simulates real deployment drift.

## Q4: Why include fatal recall separately?
In road safety, missing high-severity events is costly. Fatal recall is a task-relevant risk metric.

## Q5: What is your biggest current limitation?
Current public demo uses a small sample for reproducibility, so performance can appear optimistic.  
I explicitly label results as sample demo performance and avoid policy-level claims.

## Q6: What failure did you face and how did you fix it?
I faced invalid target values causing training instability, and cloud/runtime compatibility issues in model loading/plot rendering.  
Fixes: label-space governance, cloud version alignment, and non-interactive plotting backend.

## Q7: How do you avoid leakage?
The pipeline isolates target and feature lists, splits data before model fitting for evaluation, and reports all metrics from held-out data.

## Q8: What would you do next if given two more weeks?
Integrate OSM road attributes, air quality and deprivation data into spatial_key joins, then repeat reliability tests on larger processed master tables.

