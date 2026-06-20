# Customer Churn Prediction — Kaggle Playground Series S6E3

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![AUC](https://img.shields.io/badge/CV%20AUC-0.9157-brightgreen?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E3-20BEFF?style=flat-square&logo=kaggle)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square)

---

## Problem Statement

Predict the **probability that a telecom customer will churn** (leave the service) based on their usage patterns, contract details, and billing information.

This is a **binary classification** problem evaluated using the **ROC-AUC score**.

> **Competition:** [Kaggle Playground Series — Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3)
> **Sponsored by:** Google LLC

---

## Dataset Overview

| Property | Details |
|---|---|
| Records | ~594,000 customer entries |
| Features | 21 (demographics, services, billing) |
| Target | `Churn` — Yes (1) / No (0) |
| Evaluation Metric | ROC-AUC Score |

### Key Features

| Feature | Description |
|---|---|
| `tenure` | Months the customer has been with the company |
| `Contract` | Month-to-month, One year, Two year |
| `MonthlyCharges` | Monthly billing amount |
| `TotalCharges` | Total amount billed |
| `InternetService` | DSL, Fiber optic, or None |
| `PaymentMethod` | Electronic check, Credit card, etc. |
| `TechSupport` | Whether customer has tech support |

---

## Project Structure

```
customer-churn-kaggle/
│
├── notebook.ipynb          # Main Colab notebook (EDA + Model)
├── submission.csv          # Final Kaggle submission file
├── train.csv               # Training data (from Kaggle)
├── test.csv                # Test data (from Kaggle)
├── sample_submission.csv   # Submission format reference
└── README.md               # Project documentation
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized churn distribution across all key features
- Identified strongest churn signals:
  - Month-to-month contracts → **42% churn rate**
  - Electronic check payment → **49% churn rate**
  - No tech support → **40% churn rate**
  - Fiber optic internet → **41.5% churn rate**

### 2. Data Preprocessing
- Dropped irrelevant `id` column
- Encoded binary Yes/No columns to 1/0
- Converted `TotalCharges` to numeric, imputed missing values with median
- Applied one-hot encoding to multi-class categorical columns
- Aligned train and test columns after encoding

### 3. Model Training
- Trained **XGBoost** and **LightGBM** classifiers
- Used **5-Fold Stratified Cross-Validation** for reliable evaluation
- Selected XGBoost as the final model based on CV performance

### 4. Evaluation

| Model | Mean CV AUC |
|---|---|
| XGBoost | **0.9157** |
| LightGBM | 0.9156 |

---

## Model Configuration

```python
XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)
```

---

## How to Reproduce

### 1. Clone this repository
```bash
git clone https://github.com/your-username/customer-churn-kaggle.git
cd customer-churn-kaggle
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

### 3. Download dataset from Kaggle
```bash
kaggle competitions download -c playground-series-s6e3
```

### 4. Run the notebook
Open `notebook.ipynb` in Google Colab or Jupyter and run all cells.

---

## Key Insights

- **Contract type** is the strongest predictor of churn
- Customers on **month-to-month contracts** are 42x more likely to churn than two-year contract holders
- **Electronic check** users have nearly 6x higher churn than other payment methods
- Customers with **lower tenure** (< 17 months) are at the highest risk
- Churned customers pay **~$20 more per month** on average

---

## Tech Stack

- **Language:** Python 3.10
- **Environment:** Google Colab
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost, LightGBM

---

## Author
**Aviraj Virape**
- GitHub: [@avidada35](https://github.com/avidada35)
- Kaggle: [@avirajvirape](https://www.kaggle.com/avirajvirape)
- LinkedIn: [Aviraj Virape](https://www.linkedin.com/in/aviraj-virape-667a31217/)

---

## 📄 License

This project is built for educational purposes.
Dataset sourced from [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) under **CC BY 4.0** license.
