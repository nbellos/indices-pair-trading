# Pairs Trading Strategy Using Cointegration

Statistical arbitrage implementation based on the Engle-Granger cointegration framework for identifying and trading mean-reverting pairs of financial indices.

## Overview

This project implements a pairs trading strategy that:
1. Identifies cointegrated pairs using the Engle-Granger two-step procedure
2. Computes dynamic z-scores using rolling and expanding OLS regression
3. Generates trading signals based on mean-reversion properties of cointegrated spreads

The implementation avoids look-ahead bias by using only historical data for parameter estimation, making it suitable for backtesting and live trading applications.

## Project Structure

```
pair-trading-strategy/
├── data/                           # Input data files
│   ├── indices final.xlsx         # Excel source data
│   └── indices_eur.csv            # CSV version (auto-generated)
├── scripts/                        # Analysis scripts
│   └── check_cointegration.py     # Main cointegration pipeline
├── outputs/                        # Generated results
│   ├── cointegration_results/     # Pair test statistics
│   ├── price_series/              # Price series plots
│   ├── z_scores/                  # Z-score comparison plots
│   └── z_scores_data/             # Z-score time series (CSV)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd pair-trading-strategy
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Execute the cointegration analysis pipeline:

```bash
cd scripts
python check_cointegration.py
```

### Workflow

The script performs the following operations:

1. **Data Loading**: Reads price data from `data/indices_eur.csv` (auto-converts from Excel if needed)
2. **Log Transformation**: Converts price series to log-prices for stationarity analysis
3. **Unit Root Testing**: Applies Augmented Dickey-Fuller (ADF) test to identify I(1) series
4. **Cointegration Testing**: Tests all I(1) pairs using Engle-Granger methodology
5. **Z-Score Calculation**: Computes normalized spreads using rolling and expanding windows
6. **Output Generation**: Saves results, plots, and data files with timestamps

### Output Files

All outputs are timestamped (`YYYYMMDD_HHMMSS`) and saved to `outputs/`:

| File | Description |
|------|-------------|
| `cointegration_results_*.csv` | Full test results for all pairs (α, β, ADF statistics, p-values) |
| `price_series_*.png` | Raw price plots for all indices |
| `z_scores_*.png` | Rolling vs expanding z-scores with ±2σ trading bands |
| `z_scores_all_pairs_*.csv` | Combined z-score time series for all cointegrated pairs |

## Mathematical Methodology

### 1. Data Preparation

Convert raw prices to log-prices:

$$P_t^{\text{log}} = \ln(P_t)$$

This transformation:
- Stabilizes variance
- Makes returns additive rather than multiplicative
- Simplifies cointegration analysis

### 2. Stationarity Testing (ADF Test)

For each series $X_t$, test the null hypothesis $H_0: X_t \sim I(1)$ using:

$$\Delta X_t = \alpha + \beta t + \gamma X_{t-1} + \sum_{i=1}^{p} \delta_i \Delta X_{t-i} + \epsilon_t$$

Where:
- $\Delta X_t = X_t - X_{t-1}$ is the first difference
- Test statistic: $\tau = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}$
- Reject $H_0$ if $p < 0.05$ (series is stationary, I(0))
- Fail to reject if $p \geq 0.05$ (series is non-stationary, I(1))

**Pre-filter**: Only I(1) series are candidates for cointegration.

### 3. Engle-Granger Cointegration Test

For each pair of I(1) series $(X_t, Y_t)$:

**Step 1: Estimate cointegrating regression** (Static OLS on full sample)

$$Y_t = \alpha + \beta X_t + \epsilon_t$$

Using OLS:

$$\hat{\beta} = \frac{\sum_{t=1}^{T} (X_t - \bar{X})(Y_t - \bar{Y})}{\sum_{t=1}^{T} (X_t - \bar{X})^2}$$

$$\hat{\alpha} = \bar{Y} - \hat{\beta} \bar{X}$$

**Step 2: Compute spread (residuals)**

$$S_t = Y_t - (\hat{\alpha} + \hat{\beta} X_t)$$

**Step 3: Test spread stationarity**

Apply ADF test to $S_t$:
- If $p < 0.05$: Spread is I(0) → **Pair is cointegrated**
- If $p \geq 0.05$: Spread is I(1) → Pair is not cointegrated

### 4. Dynamic Z-Score Calculation

For trading signals, we avoid look-ahead bias by using **dynamic window estimation**:

#### 4.1 Rolling Window (5-year window, $w = 1260$ trading days)

At each time $t$, estimate parameters using only data $[t-w, t)$:

$$\hat{\alpha}_t^{\text{roll}}, \hat{\beta}_t^{\text{roll}} = \text{OLS}(Y_{t-w:t}, X_{t-w:t})$$

$$S_t^{\text{roll}} = Y_t - (\hat{\alpha}_t^{\text{roll}} + \hat{\beta}_t^{\text{roll}} X_t)$$

$$Z_t^{\text{roll}} = \frac{S_t^{\text{roll}} - \mu_{t-w:t}^{\text{roll}}}{\sigma_{t-w:t}^{\text{roll}}}$$

Where:
- $\mu_{t-w:t}^{\text{roll}} = \frac{1}{w} \sum_{i=t-w}^{t} S_i^{\text{roll}}$
- $\sigma_{t-w:t}^{\text{roll}} = \sqrt{\frac{1}{w-1} \sum_{i=t-w}^{t} (S_i^{\text{roll}} - \mu_{t-w:t}^{\text{roll}})^2}$

**Advantages**: Adapts to regime changes, captures time-varying relationships

#### 4.2 Expanding Window (Recursive)

At each time $t$, estimate using all data from start to $t$:

$$\hat{\alpha}_t^{\text{exp}}, \hat{\beta}_t^{\text{exp}} = \text{OLS}(Y_{1:t}, X_{1:t})$$

$$S_t^{\text{exp}} = Y_t - (\hat{\alpha}_t^{\text{exp}} + \hat{\beta}_t^{\text{exp}} X_t)$$

$$Z_t^{\text{exp}} = \frac{S_t^{\text{exp}} - \mu_{1:t}^{\text{exp}}}{\sigma_{1:t}^{\text{exp}}}$$

Where:
- $\mu_{1:t}^{\text{exp}} = \frac{1}{t} \sum_{i=1}^{t} S_i^{\text{exp}}$
- $\sigma_{1:t}^{\text{exp}} = \sqrt{\frac{1}{t-1} \sum_{i=1}^{t} (S_i^{\text{exp}} - \mu_{1:t}^{\text{exp}})^2}$

**Advantages**: Uses maximum available data, more stable parameter estimates

### 5. Trading Signal Rules

Based on z-score thresholds:

| Condition | Action | Rationale |
|-----------|--------|-----------|
| $Z_t > +2$ | Short $Y$, Long $X$ | Spread too high, expect mean reversion downward |
| $Z_t < -2$ | Long $Y$, Short $X$ | Spread too low, expect mean reversion upward |
| $\|Z_t\| \to 0$ | Close position | Spread converged to mean |

The ±2σ thresholds ensure entry only when spread deviates significantly from equilibrium.

## Data Format

Input CSV must follow this structure:

| Date | Index1 | Index2 | Index3 | ... |
|------|--------|--------|--------|-----|
| 2020-01-01 | 100.5 | 200.3 | 150.2 | ... |
| 2020-01-02 | 101.2 | 199.8 | 151.0 | ... |
| ... | ... | ... | ... | ... |

**Requirements**:
- First column: Dates (parseable by `pd.to_datetime`)
- Subsequent columns: Price series for each instrument
- Frequency: Any (daily, weekly, monthly) - algorithm adapts automatically

## Dependencies

See `requirements.txt` for specific versions:

```
numpy>=1.21.0          # Numerical computing
pandas>=1.3.0          # Data manipulation
statsmodels>=0.13.0    # Statistical models (ADF, OLS)
matplotlib>=3.5.0      # Plotting
scipy>=1.7.0           # Scientific computing
openpyxl>=3.0.0        # Excel file support
```

## Theoretical Background

### Cointegration

Two I(1) time series $X_t$ and $Y_t$ are **cointegrated** if there exists a linear combination:

$$S_t = Y_t - \beta X_t$$

such that $S_t \sim I(0)$ (stationary). This implies:
- Short-term deviations occur but are temporary
- Long-run equilibrium relationship exists: $E[Y_t - \beta X_t] = \alpha$
- Spread mean-reverts, enabling profitable arbitrage

### Mean Reversion Speed

The half-life of mean reversion can be estimated from the spread AR(1) process:

$$S_t = \phi S_{t-1} + \epsilon_t$$

$$\text{Half-life} = \frac{\ln(2)}{\ln(1/\phi)}$$

This indicates how quickly the spread returns to equilibrium after a shock.

## Limitations & Assumptions

1. **Stationarity assumption**: Cointegration relationship may break down during structural regime changes
2. **Transaction costs**: Not accounted for in current implementation
3. **Static testing**: Cointegration tested on full sample; consider rolling cointegration tests for robustness
4. **Normality**: Z-score thresholds assume Gaussian spread distribution (may not hold under fat tails)

## Future Enhancements

- [x] Dynamic z-score calculation (rolling/expanding OLS)
- [ ] Backtesting framework with transaction costs
- [ ] Walk-forward analysis for out-of-sample validation
- [ ] Kalman filter for time-varying beta estimation
- [ ] Johansen test for multi-asset cointegration
- [ ] Risk management (position sizing, stop-loss)
- [ ] Live trading integration with broker API

## References

1. Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing". *Econometrica*, 55(2), 251-276.
2. Alexander, C. (1999). "Optimal Hedging Using Cointegration". *Philosophical Transactions of the Royal Society A*, 357(1758), 2039-2058.
3. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. John Wiley & Sons.

## License

MIT License
