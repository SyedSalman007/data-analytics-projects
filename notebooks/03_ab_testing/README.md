# Project 03: A/B Analysis Testing (Bluetooth Speaker Conversions)

## Dataset
Reads `datasets/A:B Analysis Testing/cleaned_speakers_data.csv` (30,000 sessions).

## Run
From repo root:

```bash
scripts/bootstrap_venv.sh
source .venv/bin/activate
python notebooks/03_ab_testing/ab_testing_analysis.py
```

Or open the notebook:

```bash
jupyter notebook notebooks/03_ab_testing/03_ab_testing_dashboard.ipynb
```

## Outputs
Generated in `notebooks/03_ab_testing/output/`:
- `data_description_report.md` (detailed dataset report)
- `variant_summary.csv`
- `ab_test_results.json`
- Dashboard charts (`*.png`)
