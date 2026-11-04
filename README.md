# Bank Fraud Analysis Dashboard

An exploratory data analysis project that examines **1 million bank transactions** to understand fraud patterns, customer behavior, and transaction characteristics.

The project combines a Jupyter Notebook with an interactive **Streamlit dashboard** to present the analysis through charts, summary metrics, and filters.

## Project Objectives

- Compare fraudulent and legitimate transactions.
- Explore fraud patterns across customer ages, payment methods, and merchant categories.
- Examine transaction timing, including night and weekend activity.
- Present findings through an interactive dashboard.

## Dataset

Source: [Bank Transaction Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset)

| Attribute | Details |
|---|---|
| Transactions | 1,000,000 |
| Original columns | 26 |
| Fraud indicator | `is_fraud`: 0 = legitimate, 1 = fraud |
| Fraudulent transactions | 55,255 |
| Fraud rate | Approximately 5.53% |

## Data Preparation

- Examined data types, missing values, and summary statistics.
- Replaced missing `fraud_type` values with `Legitimate` after checking their relationship with the fraud indicator.
- Created customer identifiers, transaction counts per customer, and age groups.

## Dashboard Features

- **Overview:** Transaction counts, fraud rate, and average transaction amount.
- **Data Exploration:** Dataset preview, column types, and descriptive statistics.
- **Data Cleaning:** Missing-value summaries and created features.
- **Visual Analysis:** Transaction amounts, customer ages, fraud types, and correlations.
- **Interactive Filters:** Country, payment method, merchant category, and fraud status.

## Key Findings

- Legitimate transactions account for approximately **94.47%** of the dataset.
- Customers aged **36–45** have the highest number of fraudulent transactions in the notebook analysis.
- Credit cards have the highest fraud case count among payment methods.
- Phishing is the most frequent fraud type, although counts across fraud types are similar.

These findings describe patterns in this dataset. The project focuses on **exploratory analysis and visualization**.

## Technologies

Python · pandas · NumPy · Streamlit · Plotly · Matplotlib · Seaborn · KaggleHub

## Project Files

| File | Description |
|---|---|
| `EDA_bank_fraud.ipynb` | Exploratory analysis and findings |
| `app.py` | Interactive Streamlit dashboard |
| `requirements.txt` | Dashboard dependencies |

## Run Locally

```bash
git clone https://github.com/Abdulrahman1101/Bank-Fraud.git
cd Bank-Fraud
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app supports uploading `bank_fraud.csv` through the sidebar or loading it from `data/bank_fraud.csv`. If neither is provided, it attempts to download the dataset using KaggleHub.
