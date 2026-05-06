# XGBoost Analysis — NVDA 30-Day Price Direction

## Objective

Predict whether NVDA's price will be higher 30 days from any given trading day — framed as a binary classification:

| Label | Meaning |
|---|---|
| 1 | Positive 30-day forward ROC (price rises) |
| 0 | Non-positive 30-day forward ROC (price flat or falls) |

Predicting direction rather than exact price makes the problem more tractable and the output more actionable for trading decisions.

---

## Data

| | |
|---|---|
| Source | `model_data.csv`, produced by `DataPipeline.ipynb` |
| Period | April 1999 – December 2024 (6,479 trading days) |
| Split | 80% train / 20% test, chronological order |
| Class balance | 60% positive in train, 67% positive in test |

All cleaning, feature engineering, and multicollinearity reduction are handled upstream in the data pipeline. This notebook focuses entirely on model training and evaluation.

### Features

| Feature | Description |
|---|---|
| `MACD_Signal` | MACD signal line |
| `BB_Width` | Bollinger Band width — proxy for volatility |
| `RSI_lag1` | Relative Strength Index from the prior day |
| `ATR_Pct` | Average True Range as a percentage of price |
| `SMA20_Dist` | Distance from the 20-day simple moving average |
| `Stoch_K` | Stochastic oscillator %K |
| `ROC_roll5_mean` | 5-day rolling mean of Rate of Change |
| `Volume_Ratio` | Volume relative to its moving average |
| `CCI` | Commodity Channel Index |

---

## Approach

Algorithm: XGBoost Classifier, heavily regularised to reduce overfitting on financial time series.

Hyperparameter tuning: `RandomizedSearchCV` with 60 iterations, evaluated via `TimeSeriesSplit` (5 folds) scored on ROC-AUC. Parameters searched include `max_depth`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma`, `subsample`, and `colsample_bytree`.

Models trained:

| | Feature set |
|---|---|
| Model 1 | All 9 features |
| Model 2 | Smallest subset reaching 95% cumulative importance (derived from Model 1) |

---

## Results

### Model 1 — All 9 Features

| Metric | Value |
|---|---|
| CV ROC-AUC (best) | 0.4901 |
| Train accuracy | 61.78% |
| Test accuracy | 66.05% |
| Test precision (macro) | 0.5683 |
| Test recall (macro) | 0.5289 |
| Test F1 (macro) | 0.5002 |
| Test ROC-AUC | 0.5834 |
| Overfit gap | −0.04 |

### Model 2 — 95% Cumulative Importance Subset

| Metric | Value |
|---|---|
| CV ROC-AUC (best) | 0.4912 |
| Train accuracy | 62.32% |
| Test accuracy | 65.74% |
| Test precision (macro) | 0.5574 |
| Test recall (macro) | 0.5230 |
| Test F1 (macro) | 0.4903 |
| Test ROC-AUC | 0.5821 |
| Overfit gap | −0.03 |

### Winner: Model 1

Model 1 edges out Model 2 on test ROC-AUC (0.5834 vs 0.5821). Feature elimination offered no benefit — all 9 features contributed meaningful signal.

### Feature Importance

The strongest single predictor was ATR_Pct (importance score: 0.1503), suggesting short-term price volatility relative to level is the most useful signal for 30-day direction. The remaining eight features contributed more evenly.

---

## Interpretation

### What Worked

No overfitting. The overfit gap is essentially zero (−0.04), confirming the regularisation strategy was effective. A test ROC-AUC of 0.58 sits modestly above random chance (0.50) — a weak but genuine predictive signal.

### What Didn't

CV ROC-AUC of ~0.49 on training folds indicates the model barely distinguished classes during cross-validation. The marginal improvement on the held-out test set may partly reflect the characteristics of that specific window (the 2021–2024 NVDA bull run).

The 66% test accuracy is misleading. With 67% of test samples labelled positive, a naive "always predict up" baseline would score similarly.

Only 1.6% of predictions were high-confidence (probability ≥ 0.70 or ≤ 0.30). The model clusters near 0.5–0.65 on almost every sample, which severely limits its practical utility.

### Bottom Line

The selected technical indicators carry some directional signal for NVDA over a 30-day horizon, but not enough to build a reliable standalone classifier. The model is well-regularised and the generalisation is clean, so the performance ceiling is a feature problem, not a modelling one.
