# 📈 AAPL Stock Price Prediction

> End-to-end ML pipeline for forecasting Apple Inc. stock prices and classifying directional movements using LSTM, Gradient Boosting, and Random Forest.

---

## 🗂 Project Structure

```
.
├── AAPL.csv                    # Raw Apple Inc. historical stock data (OHLCV)
├── Untitled4.ipynb             # EDA & preprocessing pipeline
├── LSTM__1_.ipynb              # LSTM baseline regression model
└── LSTM___XGboost_2.ipynb      # LSTM + Gradient Boosting + Random Forest hybrid
```

---

## 📓 Notebooks

### 1. `Untitled4.ipynb` — EDA & Preprocessing Pipeline

Comprehensive data analysis and feature engineering workflow on `AAPL.csv`.

- Multi-class target creation: **Hausse / Stable / Baisse** (±0.5% daily change threshold)
- Correlation heatmaps, IQR-based outlier detection, multicollinearity checks
- Missing value imputation via forward-fill / backward-fill (time-series safe)
- Class imbalance handling with **SMOTE** oversampling
- Feature engineering: lag features, SMA, rolling volatility
- Temporal features: Year, Month, Day, DayOfWeek
- MinMax scaling for model-ready output

---

### 2. `LSTM__1_.ipynb` — LSTM Baseline

Baseline LSTM model for next-day closing price regression.

- 60-day sliding window sequence construction
- Stacked LSTM (2 × 50 units) + Dense output layers
- Adam optimizer — lr=0.0001, gradient clipping (clipvalue=0.5)
- 80/20 temporal train/test split
- Metrics: **RMSE, MAE, MSE, R²**

---

### 3. `LSTM___XGboost_2.ipynb` — LSTM + Ensemble Hybrid

Extended pipeline with ensemble models and full evaluation.

- LSTM regression (same architecture as baseline)
- **Gradient Boosting** & **Random Forest** regressors (100 estimators)
- 70/30 temporal split for forward-looking evaluation
- Binary classification: Hausse vs. Baisse
- Feature importance ranking (top-5 per model)
- Metrics: Accuracy, Precision, Recall, **F1, ROC-AUC**, Confusion Matrix, PR Curve

---

## 🛠 Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Preprocessing | `scikit-learn` — MinMaxScaler, train_test_split |
| Imbalance | `imbalanced-learn` — SMOTE |
| Deep Learning | `TensorFlow / Keras` — LSTM, Dense, Sequential |
| Ensemble | `scikit-learn` — GradientBoostingRegressor, RandomForestRegressor |
| Evaluation | `scikit-learn` metrics — R², RMSE, MAE, F1, ROC-AUC |
| Environment | Google Colab (Python 3) |

---

## 🚀 Getting Started

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

### Run in order

```
1. Untitled4.ipynb          → EDA + feature engineering
2. LSTM__1_.ipynb           → LSTM baseline
3. LSTM___XGboost_2.ipynb   → Ensemble + full evaluation
```

> All notebooks were developed on **Google Colab**. Upload `AAPL.csv` to your Colab session before running.

---

## 📊 Results

Models are evaluated on a held-out **temporal** test set (no data leakage).

| Task | Models | Key Metrics |
|---|---|---|
| Regression | LSTM, Gradient Boosting, Random Forest | RMSE, MAE, R² |
| Classification | Gradient Boosting, Random Forest | F1, ROC-AUC, Accuracy |

**Key finding:** Price history features (Close, OHLC, lag features) and technical indicators (SMA, rolling volatility) consistently rank as the most influential predictors across all ensemble models.

---

## 📁 Data

The dataset `AAPL.csv` contains Apple Inc. historical daily stock prices with the following columns:

| Column | Description |
|---|---|
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Daily high |
| `Low` | Daily low |
| `Close` | Closing price |
| `Adj Close` | Adjusted closing price |
| `Volume` | Number of shares traded |
