# Gold Price Prediction Using Machine Learning
## 📌 Overview

Gold has long been considered a “safe haven” asset, especially during times of economic uncertainty. This project applies **machine learning regression models** to forecast future gold prices using historical financial data. The models are trained on key economic indicators including:

- **GLD** (Gold ETF prices)
- **SPX** (S&P 500 Index)
- **USO** (Oil prices)
- **SLV** (Silver prices)
- **EUR/USD** exchange rate

The goal is to identify the most accurate and reliable model for gold price prediction, enabling better decision-making for investors, analysts, and policymakers.



## 🎯 Objectives

- Understand key financial indicators influencing gold prices.
- Clean, preprocess, and transform raw financial data.
- Perform exploratory data analysis (EDA) to uncover trends and patterns.
- Engineer relevant features (lag values, rolling statistics, percentage changes).
- Build and compare multiple regression models:
  - Linear Regression (baseline)
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
  - XGBoost Regressor
- Evaluate models using **MAE**, **RMSE**, and **R² Score**.
- Provide actionable insights for gold market behavior.



## 📊 Dataset

- **Source**: Public financial datasets (Kaggle / Yahoo Finance / GitHub)
- **Time Period**: 2008 – 2018 (over a decade)
- **Records**: ~2,290 daily observations
- **Features**:
  - `SPX` – S&P 500 Index
  - `GLD` – Gold ETF price (target variable)
  - `USO` – Oil price ETF
  - `SLV` – Silver ETF price
  - `EUR/USD` – Exchange rate

> The dataset captures multiple market phases: pre/post-2008 crisis, oil price crashes, geopolitical events, and economic recovery.



## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Visualization
  - `scikit-learn` – Model building & evaluation
  - `xgboost` – Advanced boosting model



## 🔧 Data Preprocessing & Feature Engineering

### Preprocessing Steps:
- Handled missing values using forward fill.
- Converted `Date` column to datetime format.
- Removed irrelevant/non-numeric columns.
- Applied **Min-Max Scaling** & **Standardization**.
- Split data chronologically (80% train / 20% test) to avoid data leakage.

### Engineered Features:
- **Lag features** (t-1, t-7, t-30)
- **Rolling statistics** (7-day & 30-day mean, standard deviation)
- **Price change indicators** (daily difference, percentage change)
- **Volatility ratios**
- **Temporal features** (month, weekday)

---

## 📈 Exploratory Data Analysis (EDA)

Key insights from visualizations:

- **Upward trend** in gold prices over time, with volatility spikes during crises.
- **Right-skewed distribution** – occasional sharp price surges.
- **High correlation** between `GLD` and `SLV` (silver).
- **Weak correlation** between `GLD` and `SPX` – confirming gold’s safe-haven property.
- **Seasonality** – prices tend to rise around September–October (festive demand).



---

## 🤖 Model Building & Evaluation

### Models Implemented:
| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| Linear Regression | ~0.77 | ~3.12 | ~4.95 |
| Decision Tree | ~0.91 | ~1.42 | ~2.33 |
| **Random Forest** | **~0.989** | **~0.89** | **~1.68** |

### Key Findings:
- **Random Forest Regressor** outperformed all other models, achieving near-perfect R² and lowest error rates.
- Ensemble methods handle non-linear, volatile financial data far better than linear models.
- Feature importance analysis showed **previous day’s price**, **rolling means**, and **daily % change** as top predictors.

### Visual Validation:
- Actual vs. Predicted plots showed close alignment for Random Forest.
- Residual plots were randomly scattered around zero → no systematic bias.
- Feature importance charts improved model interpretability.


