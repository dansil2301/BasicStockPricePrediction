# NVIDIA Stock Movement Prediction

## Project Overview
This project predicts the 30-day directional movement of NVIDIA (`NVDA`) stock price using technical indicator features and supervised machine learning. The workflow is implemented as a reproducible data pipeline plus three evaluation notebooks for Logistic Regression, Random Forest, and XGBoost.

## Part 1: Introduction — Motivation and Problem Statement

### Problem Definition
The objective is to forecast whether NVIDIA stock will close higher after 30 trading days, using historical price data and engineered technical indicators. This is a binary classification problem where:
- `1` means the 30-day future return is positive
- `0` means the 30-day future return is non-positive

### Motivation and Key Goals
- Build a machine learning pipeline that turns raw price history into predictive features
- Evaluate classification models for directional trading signals
- Compare baseline linear models with ensemble and gradient-boosted models
- Demonstrate feature importance and the value of technical indicator engineering

### Why Machine Learning?
Stock movement prediction is noisy and driven by many interacting signals. Machine learning helps by:
- combining multiple technical indicators into a single predictive model
- learning non-linear relationships in the data
- providing evaluation metrics to compare model performance

### Applicable ML Techniques
The project uses the following ML task categories:
- cleaning: remove missing values caused by rolling features and target alignment
- transformation: compute technical indicators from raw OHLCV data
- scaling / normalization: standardize features for models such as Logistic Regression
- feature engineering: build lagged series, ratios, volatility and momentum indicators
- classification: train models to predict upward vs non-upward 30-day movement
- model validation: use time-series-aware cross-validation and hyperparameter search

### Research Context and SOTA Techniques
This project follows common research themes in financial forecasting:
- technical-indicator-based classification for short-term direction prediction
- ensemble methods such as Random Forest and XGBoost for robust signal extraction
- time-series-aware validation to avoid look-ahead bias

Financial ML literature often emphasizes:
- the importance of momentum and volatility indicators such as RSI, MACD, ATR, and Bollinger Bands
- the use of ensemble learners to reduce overfitting on noisy market data
- ROC-AUC as a preferred metric for imbalanced directional prediction tasks

### High-Level Architecture and Workflow
The project is structured in two main stages:
1. Data pipeline and feature engineering
2. Model training and evaluation

```
NVDA.csv
   └─> DataPipeline.ipynb
         └─> model_data.csv
               ├─> Analysis/LogisticRegression.ipynb
               ├─> Analysis/RandomForest.ipynb
               └─> Analysis/XGBoost_analysis.ipynb
```

## Part 2: Suggested Methodology — Data Understanding and EDA

### Data Understanding
The dataset contains historical NVIDIA equity data with columns such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.
The pipeline adds technical indicators including:
- moving averages: SMA, EMA
- momentum: RSI, MACD, ROC, Stochastic oscillator
- volatility: ATR, Bollinger band width
- volume features: OBV, MFI, volume ratio
- distance from moving average and intraday range metrics

### Data Cleaning and Processing
The data pipeline performs the following preprocessing steps:
- parse dates and sort chronologically
- cast price and volume fields to numeric
- compute indicator features with rolling windows
- drop the NaN rows created by rolling indicator calculations
- create a 30-day forward target label
- save processed dataset to `model_data.csv`

### Exploratory Data Analysis (EDA)
The notebooks include the following EDA elements:
- feature distribution plots by class
- class balance inspection for the binary target
- descriptive statistics for key indicators
- visualization of indicators against price through time
- correlation and importance analysis for feature selection

### Challenges and Limitations
Common challenges for this dataset and problem include:
- noisy, non-stationary stock returns
- small signal-to-noise ratio in technical indicators
- time-series leakage if data is shuffled incorrectly
- class imbalance in upward vs non-upward target labels

Proposed solutions in the codebase:
- preserve chronological order in train/test splits
- use time-series cross-validation rather than random CV
- standardize features when needed to improve model stability
- evaluate using ROC-AUC and confusion matrices, not just accuracy

### Advanced Feature Engineering
The pipeline creates features such as:
- lagged indicators (e.g., `RSI_lag1`, `ROC_lag1`)
- rolling means and standard deviations (e.g., `ROC_roll5_mean`, `ROC_roll5_std`)
- volatility ratios (`ATR_Pct`, `BB_Width`)
- trend distance relative to moving average (`SMA20_Dist`)

### Model Processing Pipelines
The notebooks build model-specific processing steps:
- `LogisticRegression.ipynb`: StandardScaler + Logistic Regression with `TimeSeriesSplit`
- `RandomForest.ipynb`: raw tree-based features + Random Forest with `RandomizedSearchCV`
- `XGBoost_analysis.ipynb`: raw tree-based features + XGBoost with time-series CV and early stopping

## Part 3: Validation & Evaluation — Predictive Modeling

### Models Implemented
Three different models are built and compared:
- Logistic Regression (linear classification baseline)
- Random Forest (ensemble tree-based model)
- XGBoost (gradient boosting tree ensemble)

### Training and Validation Strategy
Each model is validated using time-series-aware methods:
- chronological train/test split (no shuffling)
- `TimeSeriesSplit` for cross-validation
- `RandomizedSearchCV` for hyperparameter tuning
- early stopping in XGBoost to limit overfitting

### Evaluation Metrics
The notebooks evaluate classification performance using:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- confusion matrix

In addition, the project adds practical model assessment steps such as:
- overfitting gap analysis between train and test accuracy
- prediction confidence breakdown
- feature importance ranking for ensemble models

### Trade-offs and Interpretation
Model trade-offs discussed in code include:
- choosing ROC-AUC over accuracy for imbalanced directional forecasting
- using ensemble complexity to improve ranking performance while monitoring overfit gap
- preferring a reduced feature set when it improves out-of-sample ROC-AUC

The notebooks perform error analysis by inspecting:
- confusion matrix entries
- class imbalance behavior
- prediction probability distributions for high- and low-confidence forecasts

### Final Model Selection
The project selects the best model based on test ROC-AUC and generalization behavior. The Random Forest notebook also compares a full 9-feature model to a reduced model selected by feature importance.

## Part 4: Conclusion

### Final Insights
The project demonstrates a structured workflow for stock direction prediction using technical features and time-series ML evaluation. It emphasizes that short-term financial forecasting benefits from:
- careful feature engineering
- time-series-aware validation
- ensemble methods with feature selection

Based on the analysis notebooks, the most likely best model is XGBoost. This is because XGBoost combines regularized gradient boosting with time-series cross-validation, early stopping, and feature importance analysis, which makes it well suited to noisy financial indicator data and improves generalization over simpler models.

### Business and Technical Value
From a business perspective, this work provides:
- an interpretable process for building directional trading signals
- a repeatable pipeline from raw market data to model evaluation
- insight into which technical indicators contribute most to predictive power

From a technical perspective, it validates:
- the value of time-series cross-validation
- the utility of tree-based methods in noisy markets
- the importance of proper preprocessing and target construction

### Future Work
Potential improvements include:
- adding additional asset data or macroeconomic features
- testing longer or shorter forecast horizons
- using SHAP or permutation importance for deeper feature explainability
- exploring deep learning or hybrid models for richer temporal patterns
- improving class balance with resampling or cost-sensitive learning

## How to Read the Project
1. Open `DataPipeline.ipynb` first to understand feature engineering, target creation, and dataset preparation.
2. Review `model_data.csv` as the processed dataset for modeling.
3. Open `Analysis/LogisticRegression.ipynb` for a baseline classification workflow.
4. Open `Analysis/RandomForest.ipynb` for ensemble training, feature importance, and model reduction analysis.
5. Open `Analysis/XGBoost_analysis.ipynb` for advanced gradient boosting, hyperparameter tuning, and model comparison.

## Requirements
Recommended Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pandas_ta

Run the notebooks in order, using the processed dataset from `DataPipeline.ipynb` as input to the analysis notebooks.
