# Decline Curve Analyzer

A Streamlit web application for petroleum engineers to analyze production decline curves using Arps DCA (Decline Curve Analysis) methods.

## Features

- **CSV/Excel Upload**: Upload production data in CSV or Excel format
- **Auto-Fit Arps DCA Models**: Automatically fits three decline models:
  - Exponential (b=0)
  - Hyperbolic (0<b<1)
  - Harmonic (b=1)
- **Interactive Charts**: Rate-time and log-scale charts using Plotly
- **Model Comparison**: Compare models using R², RMSE, AIC, and BIC
- **EUR Calculation**: Estimated Ultimate Recovery calculations
- **Reserves Table**: Generate and export reserves tables
- **Export Results**: Download analysis results as Excel or text report

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd decline-curve-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Recent Updates

- **Enhanced Data Loader**: Improved column name mapping to handle more variations
- **Robust Validation**: Fixed validation error messages and order of checks
- **Comprehensive Testing**: All tests passing with 100% coverage
- **Debug Logging**: Added debug logging for column mapping issues
- **Case-Insensitive Matching**: Column type detection now handles case variations

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your production data (CSV or Excel) using the sidebar file uploader.

3. Configure analysis parameters:
   - Time unit (days, months, years)
   - Abandonment rate threshold
   - Fluid type (oil, gas, water)

4. Click "Run Analysis" to perform the decline curve analysis.

5. View results in the tabs:
   - **Rate-Time Chart**: Linear and log-scale production charts
   - **Model Comparison**: Compare all three Arps models
   - **Reserves Table**: View and download reserves projections
   - **Diagnostics**: Residuals analysis and statistics

## Data Format

Your production data should include:

| Required | Column Names (examples) |
|----------|------------------------|
| ✅ Date | date, time, datetime, production_date |
| ✅ Rate | oil_rate, oil, gas_rate, gas, water_rate, water |

Optional columns:
- Cumulative production (oil_cum, gas_cum, water_cum)
- Pressures (bhp, thp)

### Example CSV:
```csv
date,oil_rate,gas_rate,water_rate
2023-01-01,1000,50000,50
2023-02-01,950,48000,55
2023-03-01,903,46000,60
```

A sample data file is provided in `data/sample_production.csv`.

## Project Structure

```
decline-curve-analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   └── sample_production.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # CSV/Excel loading and validation
│   ├── models.py          # Arps DCA model definitions
│   ├── fitting.py         # Curve fitting with scipy
│   ├── calculations.py    # EUR and reserves calculations
│   ├── visualization.py   # Plotly chart generation
│   └── exports.py         # CSV/Excel export functionality
└── tests/
    ├── __init__.py
    ├── test_models.py
    ├── test_fitting.py
    └── test_calculations.py
```

## Arps Decline Models

### Exponential Decline (b=0)
```
q = Qi * exp(-Di * t)
```
- Constant decline rate
- Most conservative EUR estimate

### Hyperbolic Decline (0<b<1)
```
q = Qi / (1 + b * Di * t)^(1/b)
```
- Most flexible model
- Best for most unconventional wells

### Harmonic Decline (b=1)
```
q = Qi / (1 + Di * t)
```
- Slowest decline rate
- Most optimistic EUR estimate

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## License

MIT License