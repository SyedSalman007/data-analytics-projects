# data-analytics-projects

## Structure

```text
data-analytics-projects
├── datasets
└── notebooks
```

## Project 01: Customer Churn Analytics

Implemented in:
- `notebooks/customer_churn_analysis.py`
- `notebooks/01_customer_churn_analytics.ipynb`

This project includes:
1. Cohort analysis (customer retention by signup month)
2. Behavior patterns before churn
3. Churn risk segmentation
4. Churn prediction model

### Setup and Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 notebooks/customer_churn_analysis.py
```

Outputs are generated in `notebooks/output/`.
