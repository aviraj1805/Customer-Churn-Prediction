# Data

The dataset comes from the Kaggle competition
[Playground Series - Season 6, Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3)
and is licensed under CC BY 4.0. It is not committed to this repository because of its size (~115 MB).

## Download

Option A: Kaggle CLI (requires a Kaggle API token in `~/.kaggle/kaggle.json`):

```bash
kaggle competitions download -c playground-series-s6e3 -p data/raw
unzip data/raw/playground-series-s6e3.zip -d data/raw
```

Option B: download the zip from the competition's *Data* tab and extract it into `data/raw/`.

## Expected layout

```
data/raw/
├── train.csv               # 594,194 rows, 21 columns (id, 19 features, Churn)
├── test.csv                # 254,655 rows, 20 columns (no target)
└── sample_submission.csv   # Kaggle submission format: id, Churn
```

## Schema

| Column | Type | Values |
|---|---|---|
| `id` | int | Row identifier (dropped before modelling) |
| `gender` | category | Male, Female |
| `SeniorCitizen` | int | 0, 1 |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | category | Yes, No |
| `MultipleLines` | category | Yes, No, No phone service |
| `InternetService` | category | DSL, Fiber optic, No |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | category | Yes, No, No internet service |
| `Contract` | category | Month-to-month, One year, Two year |
| `PaymentMethod` | category | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `tenure` | int | Months with the company (1-72) |
| `MonthlyCharges` | float | Monthly bill |
| `TotalCharges` | float | Total billed to date |
| `Churn` | category | Yes, No (target, train only; 22.5% Yes) |
