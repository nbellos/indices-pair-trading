# Pair Trading Strategy

Cointegration-based pairs trading analysis for financial indices using the Engle-Granger methodology.

## Overview

This project identifies cointegrated pairs of financial instruments and computes trading signals based on mean-reverting spreads. It uses:
- **ADF tests** to filter non-stationary (I(1)) series
- **Engle-Granger cointegration** to find stationary spreads
- **Z-score normalization** for signal generation

## Project Structure

```
pair-trading-strategy/
├── data/                           # Input data files
│   ├── indices final.xlsx         # Excel source data
│   └── indices_eur.csv            # CSV version (auto-generated)
├── scripts/                        # Analysis scripts
│   └── check_cointegration.py     # Main cointegration pipeline
├── outputs/                        # Generated results
│   ├── cointegration_results/     # CSV with all pair test results
│   ├── price_series/              # Price plots
│   ├── unnormalized_spreads/      # Spread plots
│   ├── z_scores/                  # Z-score plots
│   └── z_scores_data/             # Z-score CSV exports
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pair-trading-strategy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the cointegration analysis:

```bash
cd scripts
python check_cointegration.py
```

### What it does:

1. Loads price data from `data/indices_eur.csv` (or converts from Excel if needed)
2. Converts prices to log scale
3. Runs ADF tests on all series to filter I(1) (non-stationary) indices
4. Tests all I(1) pairs for cointegration using Engle-Granger:
   - OLS regression: `y = α + βx`
   - Spread = `y - (α + βx)`
   - ADF test on spread (p < 0.05 → cointegrated)
5. Computes z-scores for cointegrated spreads
6. Generates plots and exports results

### Outputs:

All outputs are timestamped and saved to `outputs/`:

- **cointegration_results_YYYYMMDD_HHMMSS.csv**: All pair test results (α, β, ADF stat, p-value)
- **price_series_YYYYMMDD_HHMMSS.png**: Raw price plots for all indices
- **unnormalized_spreads_YYYYMMDD_HHMMSS.png**: Spread time series with mean line
- **z_scores_YYYYMMDD_HHMMSS.png**: Normalized spreads with ±2σ bands
- **z_scores_YYYYMMDD_HHMMSS.csv**: Z-score data for backtesting

## Data Format

The script expects a CSV file with:
- First column: dates (any frequency: daily, weekly, monthly)
- Remaining columns: price series for each index/instrument
- Dates should be parseable by `pd.to_datetime`

Example:
```
Date,Index1,Index2,Index3
2020-01-01,100.5,200.3,150.2
2020-01-02,101.2,199.8,151.0
...
```

## Requirements

- Python 3.7+
- numpy
- pandas
- statsmodels
- matplotlib
- scipy
- openpyxl (for Excel support)

See `requirements.txt` for version details.

## Methodology

**Engle-Granger Cointegration:**
1. Pre-filter to I(1) series using ADF test (p ≥ 0.05)
2. For each pair (X, Y):
   - Estimate cointegrating relationship: `Y = α + βX + ε`
   - Test if residuals (spread) are I(0) using ADF
3. If spread is stationary (p < 0.05), the pair is cointegrated
4. Z-score the spread for trading signals

**Trading Signals (conceptual):**
- Z-score > +2: spread is too high → short Y, long X
- Z-score < -2: spread is too low → long Y, short X
- Z-score → 0: spread reverts to mean → exit position

## Next Steps

- [ ] Implement backtesting framework
- [ ] Add rolling z-scores for out-of-sample signals
- [ ] Risk management and position sizing
- [ ] Johansen test for multiple cointegrating vectors
- [ ] Real-time monitoring

## License

MIT License (or specify your license)
