# Decline Curve Analyzer

> A production-ready Streamlit web app for petroleum engineers to model, fit, and compare **Arps decline curves** — with automated EUR estimation, statistical model selection, and interactive Plotly charts.

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-red?style=flat-square)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing%20%7C%20100%25%20coverage-brightgreen?style=flat-square)]()

---

## What this does

Upload any well's production history and get back — in seconds:
- Automated curve fits for all three Arps DCA models (exponential, hyperbolic, harmonic)
- Model comparison via R², RMSE, AIC, and BIC
- Estimated Ultimate Recovery (EUR) per model
- Exportable reserves tables in Excel or text format

No manual parameter tuning. No spreadsheet gymnastics.

---

## Decline models

| Model | Equation | Use case |
|---|---|---|
| Exponential (b=0) | `q = Qi · exp(−Di · t)` | Conservative EUR; constant decline |
| Hyperbolic (0<b<1) | `q = Qi / (1 + b·Di·t)^(1/b)` | Best fit for unconventional wells |
| Harmonic (b=1) | `q = Qi / (1 + Di · t)` | Optimistic EUR; slow-decline fields |

---

## Quick start

```bash
git clone https://github.com/h1ghf0x0/decline-curve-analyzer
cd decline-curve-analyzer
pip install -r requirements.txt
streamlit run app.py
```

Then upload a CSV or Excel file and hit **Run Analysis**.

---

## Features

- **Flexible data ingestion** — CSV or Excel with case-insensitive, fuzzy column matching for dates, rates, and cumulative volumes
- **Auto-fit all three Arps models** simultaneously via `scipy` optimization
- **Interactive Plotly charts** — rate-time and log-scale with zoom, hover, and download
- **Statistical model selection** — compare fits using R², RMSE, AIC, and BIC side by side
- **EUR & reserves projection** — per model, with abandonment rate threshold control
- **Export** — results as Excel workbook or plain-text report
- **Tested** — pytest suite with 100% coverage

---

## Data format

| Column | Required | Accepted names |
|---|---|---|
| Date | ✅ | `date`, `time`, `datetime`, `production_date` |
| Rate | ✅ | `oil_rate`, `oil`, `gas_rate`, `gas`, `water_rate` |
| Cumulative | optional | `oil_cum`, `gas_cum`, `water_cum` |
| Pressure | optional | `bhp`, `thp` |

A sample file is provided at `data/sample_production.csv`.

```csv
date,oil_rate,gas_rate,water_rate
2023-01-01,1000,50000,50
2023-02-01,950,48000,55
2023-03-01,903,46000,60
```

---

## Project structure

```
decline-curve-analyzer/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── data/
│   └── sample_production.csv
├── src/
│   ├── models.py           # Arps model definitions
│   ├── fitting.py          # scipy curve fitting engine
│   ├── calculations.py     # EUR & reserves logic
│   ├── visualization.py    # Plotly chart generation
│   ├── data_loader.py      # CSV/Excel ingestion & validation
│   └── exports.py          # Excel/TXT export
└── tests/                  # pytest — 100% coverage
    ├── test_models.py
    ├── test_fitting.py
    └── test_calculations.py
```

---

## Tech stack

Python · Streamlit · Plotly · SciPy · pandas · NumPy · openpyxl · pytest

---

## License

MIT
