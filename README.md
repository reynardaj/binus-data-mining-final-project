# Lettuce Growth Weight Prediction Pipeline (KDD Data Mining)

This repository contains an end-to-end Python data mining pipeline executing the full **Knowledge Discovery in Databases (KDD)** process (Selection, Preprocessing, Transformation, Data Mining / Modeling, and Interpretation / Evaluation) to predict daily lettuce leaf weight (`predicted_weight_g`) from hourly environmental sensor telemetry across 28-day growth cycles.

---

## 📌 Knowledge Discovery in Databases (KDD) Workflow

### 1. Data Selection & Preprocessing (`feature/preprocess.py`)
* **Data Integration**: Integrates 28 growth cases (`CASE_01` to `CASE_28`) containing hourly environmental telemetry (temperature, humidity, CO2, EC, water spray, LED lighting spectrums).
* **Data Cleaning**: Solves missing values via forward/backward fill and clips sensor values within valid physical bounds.
* **Chronological Alignment**: Maps 24 hourly climate observations from day $t$ (DAT $0 \dots 27$) to target daily lettuce weight measured at day $t+1$ (DAT $1 \dots 28$).

### 2. Feature Engineering & Transformation (`data/transformed_train_data.csv`)
* **Daily Summary Statistics**: Mean, std, min, max per DAT for temperature, humidity, CO2, EC, water spray, and lighting spectrums.
* **Cumulative Features**: Running totals over 28 days for water spray, total LED light, red light, white light, and CO2.
* **Interaction Features**: Multiplicative terms capturing dissolved nutrient delivery (`EC * water spray`), environmental stress (`temp * humidity`), light-CO2 potential (`total_light * CO2`), and thermal-light integral (`temp * total_light`).
* **Noise Reduction**: Advanced smoothing applied on daily telemetry averages:
  * **Exponential Moving Average (EMA)** (span = 3)
  * **1D Kalman Filter** state estimation

### 3. Data Mining & Regression Models (`predict_model/train.py`)
Implements 5 distinct regression models evaluated using **5-Fold `GroupKFold` Cross-Validation** (grouped by `case_id` to prevent temporal data leakage):
1. **Linear Regression** (Standardized Baseline)
2. **Random Forest Regressor** (150 Trees)
3. **Support Vector Regression (SVR)** (RBF Kernel, $C=50.0$)
4. **Gradient Boosting Regressor** (LightGBM)
5. **Multi-Layer Perceptron (MLP)** (Neural Network with 128-64 architecture)

---

## 📊 Evaluation Summary (`evaluation_summary.csv`)

| Rank | Model | Out-of-Fold RMSE (g) | Out-of-Fold MAE (g) | $R^2$ Score |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Multi-Layer Perceptron (MLP)** | **19.38** | **12.31** | **0.7827** |
| **2** | Support Vector Regression (SVR) | 21.14 | 14.46 | 0.7414 |
| **3** | Gradient Boosting (LightGBM) | 21.44 | 13.86 | 0.7341 |
| **4** | Random Forest Regressor | 21.57 | 13.70 | 0.7308 |
| **5** | Linear Regression (Baseline) | 25.60 | 18.36 | 0.6209 |

---

## 📈 Visualizations
- `model_comparison.png`: Comparison bar chart of RMSE & MAE across all 5 models.
- `feature_importance.png`: Top 15 environmental features driving lettuce biomass growth.
- `predicted_vs_actual.png`: Out-of-fold predicted vs actual weight scatter plot for best model (MLP).

---

## 🚀 How to Run

### Requirements
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn scipy joblib
```

### 1. Execute Preprocessing & Feature Engineering
```bash
python feature/preprocess.py
```

### 2. Execute Model Training & Evaluation
```bash
python predict_model/train.py
```
