# AI/ML Internship 

A comprehensive collection of machine learning and data science projects demonstrating core skills in exploratory data analysis, model building, evaluation, and interpretation.

---

## 📋 Table of Contents

1. [Task 1: Iris Dataset EDA](#task-1-iris-dataset-eda)
2. [Task 2: Stock Price Prediction](#task-2-stock-price-prediction)
3. [Task 3: Heart Disease Prediction](#task-3-heart-disease-prediction)
4. [Task 4: House Price Prediction](#task-4-house-price-prediction)
5. [Technologies & Tools](#technologies--tools)
6. [Setup & Installation](#setup--installation)

---

## Task 1: Iris Dataset EDA

**File:** `iris_eda.ipynb`

### Objective
Load, inspect, and visualize the Iris dataset to understand data trends and distributions.

### Dataset
- **Source:** Seaborn (built-in dataset)
- **Size:** 150 samples × 5 features
- **Features:** 
  - `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (numeric)
  - `species` (categorical: setosa, versicolor, virginica)

### Models Applied
- **Exploratory Data Analysis (EDA)** — distributions, correlations, outlier detection

### Key Results & Findings
- ✅ **No missing values** — dataset is clean and ready to use
- ✅ **Class Balance:** 50 samples per species
- 🔍 **Setosa** is clearly separable from other species using petal dimensions
- 📊 **Feature Correlations:**
  - `petal_length` ↔ `petal_width`: r ≈ 0.96 (strongest discriminator)
  - `sepal_width` has lowest discriminative power
- 📈 Visualizations include pairplots, histograms, box plots, and correlation heatmap

---

## Task 2: Stock Price Prediction

**File:** `stock_prediction.ipynb`

### Objective
Use historical stock data to predict the next day's closing price.

### Dataset
- **Source:** Yahoo Finance (yfinance library)
- **Stock:** AAPL (Apple Inc.)
- **Period:** 2022-01-01 to 2024-12-31
- **Size:** 752 trading days
- **Features:** Open, High, Low, Close, Volume, Moving Averages, Price Change

### Models Applied
1. **Linear Regression** — baseline model
2. **Random Forest Regressor** — capture non-linear patterns

### Key Results & Findings
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | $2.69 | $3.40 | 0.9321 |
| Random Forest | $30.73 | $33.30 | -5.4907 |

- ✅ **Linear Regression** surprisingly outperforms due to strong auto-correlation in price data
- ⚠️ **Challenge:** Next-day prices are heavily influenced by previous day's price
- 💡 **Future Improvements:** Add technical indicators (RSI, MACD) or use LSTM models

---

## Task 3: Heart Disease Prediction

**File:** `heart_disease.ipynb`

### Objective
Predict whether a patient is at risk of heart disease using health metrics.

### Dataset
- **Source:** UCI Heart Disease Dataset (from Kaggle)
- **Size:** 303 samples × 14 features
- **Target:** Binary classification (0 = No disease, 1 = Disease present)
- **Features:** age, sex, chest pain type, blood pressure, cholesterol, exercise-induced angina, etc.
- **Class Distribution:** 54.5% positive cases (balanced dataset)

### Models Applied
1. **Logistic Regression** — interpretable, probabilistic
2. **Decision Tree Classifier** — feature importance analysis

### Key Results & Findings
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 80.33% | 0.8690 |
| Decision Tree | 77.05% | 0.8310 |

- ✅ Both models perform reasonably well
- 📊 **Feature Importance:**
  - `cp` (chest pain type) — most important
  - `thalach` (max heart rate achieved)
  - `oldpeak` (ST depression)
- 💡 **Findings:** Binary classification approach is effective; ensemble methods could improve performance

---

## Task 4: House Price Prediction

**File:** `house_price.ipynb`

### Objective
Predict house sale prices based on property features.

### Dataset
- **Source:** Kaggle House Prices Competition / Synthetic (fallback)
- **Size:** 1,000 samples × 9 features
- **Target:** `SalePrice` (continuous, regression task)
- **Features:** GrLivArea, BedroomAbvGr, TotalBsmtSF, GarageCars, OverallQual, YearBuilt, FullBath, Neighborhood

### Models Applied
1. **Ridge Regression** — regularized linear regression
2. **Gradient Boosting Regressor** — advanced ensemble method

### Key Results & Findings
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Ridge Regression | $21,650 | $28,561 | 0.9294 |
| Gradient Boosting | $19,611 | $24,954 | 0.9461 |

- ✅ **Gradient Boosting** outperforms Ridge by 2%
- 📊 **Top Features (by importance):**
  1. `OverallQual` (overall material & finish quality)
  2. `GrLivArea` (living area in sqft)
  3. `YearBuilt` (construction year)
- 💡 **Technique:** Log-transformation of target improves model robustness

---

## Technologies & Tools

### Languages & Frameworks
- **Python 3** — primary language
- **Jupyter Notebooks** — interactive development

### Libraries
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Financial Data:** yfinance
- **Preprocessing:** StandardScaler, LabelEncoder, SimpleImputer

### Algorithms
- Classification: Logistic Regression, Decision Trees
- Regression: Linear Regression, Ridge, Random Forest, Gradient Boosting
- EDA: Correlation analysis, distributions, outlier detection

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/kr8457/ai-ml-internship.git
cd ai-ml-internship

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements File
Create `requirements.txt` with:
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
yfinance>=0.1.70
```

### Running the Notebooks
```bash
jupyter notebook
# Open the desired .ipynb file in your browser
```

---

## 📊 Summary of Key Skills Demonstrated

| Skill | Tasks |
|-------|-------|
| **EDA & Visualization** | All tasks (pairplots, distributions, heatmaps) |
| **Data Preprocessing** | Tasks 3, 4 (missing values, scaling, encoding) |
| **Classification** | Task 3 (binary classification, confusion matrix, ROC-AUC) |
| **Regression** | Tasks 2, 4 (MAE, RMSE, R² metrics) |
| **Feature Engineering** | Task 2 (moving averages, percentage change) |
| **Time Series** | Task 2 (temporal splits, trend analysis) |
| **Model Evaluation** | All tasks (train-test split, cross-validation, metrics) |
| **Interpretation** | All tasks (feature importance, insights) |

---

## 📈 Project Progression

```
Task 1 (EDA) → Task 2 (Time Series) → Task 3 (Classification) → Task 4 (Regression)
     ↓              ↓                       ↓                         ↓
  Exploratory   Feature Eng.         Binary Classification    Multi-feature Prediction
```

---

## 🎯 Learning Outcomes

Upon completion of these projects, you will have:
- ✅ Hands-on experience with end-to-end ML workflows
- ✅ Knowledge of multiple algorithms and when to apply them
- ✅ Ability to preprocess and visualize diverse datasets
- ✅ Proficiency in model evaluation and interpretation
- ✅ Experience with Jupyter notebooks for reproducible analysis

---

## 📝 Notes

- All notebooks include explanatory markdown cells and comments
- Results may vary slightly due to random seeds and data splits
- For production deployment, consider adding hyperparameter tuning and cross-validation
- Synthetic data is used as fallback where original datasets are unavailable

---

## 📧 Contact & Feedback

For questions or suggestions, please reach out or open an issue in the repository.

---

**Last Updated:** April 24, 2026  
**Repository:** [kr8457/ai-ml-internship](https://github.com/kr8457/ai-ml-internship)
