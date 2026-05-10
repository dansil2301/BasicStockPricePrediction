# NVIDIA Stock Direction Prediction
### Binary Classification of 30-Day Price Movement Using Technical Indicators and Supervised Machine Learning

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Context and Literature](#2-research-context-and-literature)
3. [Dataset and Feature Engineering](#3-dataset-and-feature-engineering)
4. [Methodology](#4-methodology)
5. [Models and Evaluation](#5-models-and-evaluation)
6. [Results and Discussion](#6-results-and-discussion)
7. [Conclusions and Future Work](#7-conclusions-and-future-work)
8. [Repository Structure](#8-repository-structure)
9. [Requirements and Setup](#9-requirements-and-setup)
10. [References](#10-references)

---

## 1. Project Overview

This project investigates whether supervised machine learning applied to technical indicators derived from historical price data can produce a statistically meaningful directional signal for NVIDIA (`NVDA`) stock over a 30-trading-day horizon.

**Task definition:** Binary classification — predict whether the closing price 30 trading days ahead will be higher (`1`) or lower/equal (`0`) than the current closing price.

**Core research questions:**
- Can technical indicators alone generate above-chance directional accuracy on out-of-sample data?
- How do linear, ensemble, and gradient-boosted models compare on a noisy financial classification task?
- Does feature reduction via importance-based selection improve generalisation?
- How does class imbalance affect model behaviour, and how can it be mitigated?

---

## 2. Research Context and Literature

Financial machine learning is an active and contested field. Several key findings from the literature motivate and contextualise our design choices.

**Fischer & Krauss (2018)** demonstrated that LSTM-based models can achieve roughly 60–65% directional accuracy on S&P 500 constituents, but noted substantial performance degradation over time as market regimes shift — underscoring the generalisation challenge that all ML-based financial forecasting faces.

**Patel et al. (2015)** conducted a comparative study of Random Forest, SVM, and ANN models for stock index prediction using technical indicators. Random Forest emerged as the strongest performer in their setting, which directly informed our choice to include it as a core model alongside a linear baseline.

**Bailey et al. (2014)** provide an important cautionary result: many reported ML trading strategies suffer from overfitting and inflated backtest performance due to improper validation. Their work motivates our strict use of time-series-aware cross-validation (no random shuffling, no look-ahead) rather than standard k-fold procedures.

**Gu, Kelly & Xiu (2020)** conducted a large-scale study comparing dozens of ML models in asset pricing and found that while non-linear models consistently outperform linear benchmarks, the gains are often modest and highly sensitive to the experimental setup — including the choice of features, validation horizon, and rebalancing frequency. This provides a realistic benchmark expectation for our results.

Taken together, this literature suggests that (1) any model achieving consistently above 55% ROC-AUC on out-of-sample data is meaningful, (2) validation methodology is as important as model choice, and (3) ensemble methods should be expected to outperform linear baselines on noisy technical data.

---

## 3. Dataset and Feature Engineering

### Raw Data

Historical NVIDIA OHLCV data from `NVDA.csv` with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.
Date range: 1999-01-22 to 2025-02-14

### Pipeline (`DataPipeline.ipynb`)

The raw data is transformed into a modelling-ready dataset through:

1. **Parsing and sorting** — dates parsed, records sorted chronologically
2. **Type casting** — price and volume fields cast to numeric
3. **Indicator computation** — rolling-window technical indicators computed:

| Category | Indicators |
|---|---|
| Moving averages | SMA20, EMA20 |
| Momentum | RSI, MACD, ROC, Stochastic Oscillator |
| Volatility | ATR (%), Bollinger Band Width |
| Volume | OBV, MFI, Volume Ratio |
| Derived / distance | SMA20 Distance, Intraday Range |

4. **Lag and rolling features** — lagged indicators (`RSI_lag1`, `ROC_lag1`) and rolling statistics (`ROC_roll5_mean`, `ROC_roll5_std`) added to capture short-term trends
5. **NaN removal** — rows with missing values from rolling calculations dropped. New date range: 1999-04-05 to 2024-12-31
6. **Target construction** — 30-day forward return computed; label `1` if positive, `0` otherwise
7. **Output** — processed dataset saved to `model_data.csv`

### Class Balance

The target variable exhibits a mild class imbalance: approximately **62% positive (upward)** and **38% negative/flat**. This is addressed differently per model (see Section 5).

---

## 4. Methodology

This project follows the **CRISP-DM** framework iteratively. Evaluation findings from Logistic Regression directly informed the decision to pursue ensemble methods, and feature selection was revised based on multicollinearity analysis and mutual information scores during the modelling phase. The phases map naturally to our workflow: Business Understanding covers the prediction task definition, Data Understanding covers EDA on NVDA OHLCV data, Data Preparation covers `DataPipeline.ipynb`, Modelling covers the three analysis notebooks, and Evaluation covers the cross-model comparison and metric interpretation.

### Validation Strategy

To prevent look-ahead bias — a critical concern raised by Bailey et al. (2014) — all models use **strictly chronological splits**:

- Data is **never shuffled**
- Train/test split preserves temporal order
- Cross-validation uses `TimeSeriesSplit` from scikit-learn, which only trains on past data and validates on future data
- Hyperparameter search is conducted with `RandomizedSearchCV` using the same time-series folds

### Primary Evaluation Metric: ROC-AUC

The primary metric throughout this project is **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve), chosen for the following reasons:

- **Threshold-independence:** ROC-AUC evaluates the model's ability to rank positive outcomes above negative ones across all possible classification thresholds, making it more informative than accuracy for assessing a directional signal.
- **Class imbalance robustness:** With a 60/40 split, a naive classifier that always predicts the majority class achieves 60% accuracy but only 0.5 ROC-AUC. ROC-AUC therefore provides a more honest measure of discriminative ability.
- **Trading signal interpretability:** A model used to generate trading signals is more useful if it reliably ranks strong "up" days above "down" days — a property directly measured by ROC-AUC — than if it merely achieves a high raw accuracy.

**Other metrics are reported and interpreted** as secondary diagnostics:

| Metric | When it matters most |
|---|---|
| **Accuracy** | Useful as a sanity check, but misleading under class imbalance |
| **Precision** | Important if the cost of false positives (bad trades) is high |
| **Recall** | Important if missing true positives (missed gains) is costly |
| **F1-Score** | Balanced summary of precision and recall; useful for comparing models on the same threshold |
| **Confusion Matrix** | Reveals specific failure modes (e.g., always predicting one class) |

---

## 5. Models and Evaluation

### Model 1: Logistic Regression (Baseline)

Logistic Regression is included as a linear baseline. Given the non-linear and noisy nature of financial time series, this model is expected to produce near-random results, consistent with the broader finding in Gu et al. (2020) that linear models underperform ensembles on technical indicator data. Its primary role is to establish a lower-bound benchmark against which ensemble gains can be measured.

**Pipeline:** `StandardScaler` → `LogisticRegression` → `TimeSeriesSplit` cross-validation

### Model 2: Random Forest

A Random Forest classifier is trained on the raw (unscaled) feature set. An important step in this notebook is **feature importance-based reduction**: after fitting the full 9-feature model, the `volume_ratio` feature was identified as low-importance and removed, producing an 8-feature reduced model.

**Pipeline:** `RandomForest` → `RandomizedSearchCV` with `TimeSeriesSplit`

### Model 3: XGBoost

XGBoost is trained using gradient-boosted trees with regularisation, time-series cross-validation, and early stopping to limit overfitting. Because the dataset exhibits a 60/40 class imbalance, **class weights were applied to the target variable** (`scale_pos_weight`) to prevent the model from over-predicting the majority class. No feature reduction was applied; XGBoost's internal regularisation handles feature selection implicitly.

**Pipeline:** `XGBClassifier` with `scale_pos_weight` → `TimeSeriesSplit` CV → early stopping

---

## 6. Results and Discussion

### Logistic Regression (Baseline)

The linear baseline used 9 features and was evaluated with both a train/test split and time-series cross-validation.

| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.5497 | 0.5432 |
| Precision | 0.6498 | 0.6420 |
| Recall | 0.5719 | 0.6624 |
| F1-Score | 0.6083 | 0.6520 |
| **ROC-AUC** | **0.5684** | **0.4914** |

A test ROC-AUC of 0.4914 is marginally below chance (0.50), confirming that a linear model cannot extract a reliable directional signal from these technical indicators. The near-identical train and test accuracy (0.5497 vs. 0.5432) indicates the model is not overfitting — it simply has no predictive power in a linear regime. The high recall (0.6624) relative to precision (0.6420) further suggests the model is biased toward predicting the majority (upward) class, inflating recall without genuine discriminative ability. This result is consistent with Gu et al. (2020), who found that linear models are systematically outperformed by non-linear alternatives on financial indicator data.

### Random Forest: Full vs. Reduced Model

A key experiment in the Random Forest notebook was **feature importance-based reduction**: after fitting the full 9-feature model, `Volume_Ratio` was identified as low-importance and removed. The resulting 8-feature model was evaluated against the full model:

| Metric | Full (9 features) | Reduced (8 features) | Change |
|---|---|---|---|
| Accuracy | 0.6033 | 0.5269 | −0.0764 |
| Precision | 0.6748 | 0.6975 | +0.0227 |
| Recall | 0.7520 | 0.4801 | −0.2719 |
| F1-Score | 0.7113 | 0.5687 | −0.1425 |
| **ROC-AUC** | **0.5254** | **0.5807** | **+0.0553** |

This result illustrates a key principle: **optimising for accuracy, recall, or F1 can be misleading when the goal is to rank directional signals**. The full model's high recall (0.7520) indicates it is predominantly predicting the majority class — producing superficially strong F1 scores without genuine discriminative improvement. Removing `Volume_Ratio` reduces this majority-class bias, and the ROC-AUC improves by 0.0553, representing a meaningful gain in ranking quality. For a trading signal application, a model that reliably distinguishes up days from down days is more valuable than one with high raw recall.

### XGBoost

The XGBoost model was trained on 9 features (identical feature set to Logistic Regression) with class-weight correction (`scale_pos_weight`) to address the 60/40 imbalance and early stopping to limit overfitting.

| Metric | Value |
|---|---|
| Features | 9 — `MACD_Signal`, `BB_Width`, `RSI_lag1`, `ATR_Pct`, `SMA20_Dist`, `Stoch_K`, `ROC_roll5_mean`, `Volume_Ratio`, `CCI` |
| CV ROC-AUC | 0.4913 |
| Train Accuracy | 0.6211 |
| Test Accuracy | 0.6582 |
| Test Precision (macro) | 0.5606 |
| Test Recall (macro) | 0.5247 |
| Test F1 (macro) | 0.4935 |
| **Test ROC-AUC** | **0.5954** |
| Overfit gap (train − test accuracy) | −0.0371 |

XGBoost achieves the highest test ROC-AUC of all three models at **0.5954**. The negative overfit gap (−0.0371) is notable: the model generalises *better* on the test set than the training set, which is consistent with early stopping and class-weight regularisation preventing the model from memorising the training distribution. The CV ROC-AUC of 0.4913 is lower than the final test ROC-AUC, reflecting the difficulty of the earlier time-series folds. This is a common pattern in financial data where more recent data may be more learnable. These results align with Fischer & Krauss (2018) and Gu et al. (2020), who found that well-regularised non-linear models produce the most robust out-of-sample performance.

### Overfitting Monitoring

All three models include an overfitting gap analysis comparing train and test performance, directly motivated by Bailey et al. (2014). Logistic Regression shows minimal gap (near-zero), confirming underfitting rather than overfitting. XGBoost shows a negative gap, consistent with its regularisation design. The Random Forest full model warrants closer inspection given its high recall, which is a sign of majority-class bias rather than true overfitting.

### Model Comparison Summary

| Model | Test Accuracy | Test ROC-AUC | Test Precision | Test Recall | Test F1
|---|---|---|---|---|---
| Logistic Regression | 0.5432 | 0.4914 | 0.6420 | 0.6624 | 0.6520
| Random Forest (full) | 0.6033 | 0.5254| 0.6748 | 0.7520 | 0.7113
| Random Forest (reduced) | 0.5269 | 0.5807 | 0.6975 | 0.4801 | 0.5687
| **XGBoost** | **0.6582** | **0.5954** | 0.5606 | 0.5247 | 0.4935

**XGBoost is selected as the best-performing model** based on test ROC-AUC. It achieves the highest ranking quality, the best generalisation behaviour, and the strongest test accuracy. The Logistic Regression baseline confirms that non-linear methods are necessary for this task, consistent with Patel et al. (2015) and Gu et al. (2020).
- The above table reveals a consistent pattern across all three models. F1 metric and ROC-AUC scores tell different stories. 
Random Forest full model had highest F1 but weak ROC-AUC score (0.525), this is due to its high recall which points to the majority
class bias. However, XGBoost had the lowest F1 (0.494) but the strongest AUC. This proves that for imbalanced classification F1 can mislead model selection.
- XGBoost has the lowest precision (0.56) among all models, yet the best AUC. This indicates that its not blindly predicting majority
class like Random Forest full model. XGBoost has better ranking quality despite being more conservative in its predictions.
- Random Forest (reduced) model sacrificed a lot of recall (down 0.27) to gain AUC (up 0.055), improving generalisation and reducing bias.
- Logistic Regression has the second highest recall (0.6624), but random AUC. This implies majority class bias. 


---

## 7. Conclusions and Future Work

### Key Findings

- Technical indicators alone can produce a modest but measurable directional signal on NVIDIA stock. 
- Logistic Regression confirmed that the relationship between technical indicators 30-day price direction is not linearly separable.
- Ensemble and gradient-boosted models outperform the linear baseline, consistent with the literature. Modest performance of ~0.59 AUC
  for the best model suggests that some structure exists in the data but its not enough for trading use.
- ATR_pct is consistently ranked as the most important feature across all three models despite being a volatility indicator rather than
  a directional one, this suggests that our models were partly learning volatility regions. High ATR regimes in 2021-2024 test window
  correlated with upward movement due to the AI Boom.
- Volume_Ratio was identified as the weakest feature across all three models. This is likely because relative volume loses discriminative 
  power over a 30-day horizon. 
- ROC-AUC is the most appropriate primary metric for this task; accuracy and F1 can be misleading under class imbalance.
- Feature reduction based on importance for tree-based models can improve model generalisation even when it lowers raw accuracy.
- Class imbalance requires explicit handling; ignoring it biases models toward the majority class. XGBoost addressed it through a dedicated weight correction parameter, reducing its positive class bias. 
- Strict time-series validation is essential. Any data shuffling or look-ahead produces inflated and non-reproducible results.
  The temporal nature of the data needs to be preserved to avoid any leakages, as this can impact real trading performance.
- From a societal perspective, our findings reinforce the importance of validation in Financial ML. Models that disregard the temporal
  order or exploit unique market regimes can mislead investors and can cause poor trading decisions. Responsible deployment of such 
  systems requires honest reporting of limitations and performance claims.

### Limitations

- The most significant limitation is single-asset scope. NVDA experienced an extraordinary AI-driven bull run during the test period 
  2021-2024. This creates a unique environment which limits generalisability to other equities and market conditions.
- Our features are only based on price-volume data and the model lacks awareness of the macroeconomics, company fundamentals and news.
- Despite 25 years of data, the 2021–2024 test period is unique due to the AI Boom. Our models trained on historical data have no 
  mechanism to predict such shifts. This limits the feasibility of test-AUC score as a forward looking estimate of performance.
- Predicting Price Up/Down treats a 1% gain same as a 30% gain, in practise these are completely different outcomes. A model which catches
  a lot of tiny ups but misses the big moves can be less reliable as a trading tool.
- A 30-day horizon is one of many possible choices; results may differ substantially at other horizons.

### Future Work

- Incorporating macroeconomic features (e.g., interest rates, sector rotation) or sentiment signals (e.g., news NLP) as additional inputs.
- Testing alternative forecast horizons (5-day, 60-day)
- A multi-class model seperating big gains, noise and significant drops could prove more useful for real trading decisions. This can be 
  revisited after incorporating data from more equities and market periods.
- Walk-forward validation is something our models would benefit from. Retraining models on rolling windows and testing on successive
  periods would give a much more reliable estimate of real world performance than the single fixed test window used here.
- Exploring LSTM or Transformer architectures as per Fischer & Krauss (2018)
- Extending the pipeline to a multi-asset universe following the approach of Gu et al. (2020)

---

## 8. Repository Structure

```
.
├── NVDA.csv                          # Raw historical NVIDIA OHLCV data
├── DataPipeline.ipynb                # Feature engineering and target construction
├── model_data.csv                    # Processed dataset (output of pipeline)
├── Analysis/
│   ├── LogisticRegression.ipynb      # Baseline linear model
│   ├── RandomForest.ipynb            # Ensemble model with feature reduction
│   └── XGBoost_analysis.ipynb        # Gradient boosting with imbalance correction
└── README.md
```

**Execution order:**

1. `DataPipeline.ipynb` — must be run first to generate `model_data.csv`
2. `Analysis/LogisticRegression.ipynb`
3. `Analysis/RandomForest.ipynb`
4. `Analysis/XGBoost_analysis.ipynb`

---

## 9. References

Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–69.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273.

Patel, J., Shah, S., Thakkar, P., & Kotecha, K. (2015). Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques. *Expert Systems with Applications*, 42(1), 259–268.