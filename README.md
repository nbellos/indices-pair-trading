# Pair Trading Strategy (Cointegration + Grid + Bollinger)

Robust pair trading pipeline using Engle–Granger cointegration, dynamic z-scores (rolling/expanding), grid search to optimize z-entry, and Bollinger Band confirmation. Includes configurable timeframe (Daily/Weekly/Monthly), commissions, and comprehensive outputs (CSV + plots + aggregate summary).

## Project Structure

```
pair-trading-strategy/
├── data/
│   ├── indices final.xlsx            # Excel source (auto-converted if CSV missing)
│   └── indices_eur.csv               # CSV data (Date + columns of indices)
├── scripts/
│   ├── check_cointegration.py        # ADF, Engle–Granger, z-scores (rolling/expanding)
│   ├── backtest_grid.py              # Grid search + trade sim + metrics + plotting utils
│   ├── run_strategy.py               # Unified orchestrator (cointegration → grid+BB → backtest → outputs)
│   └── run_grid_bb_thresholds.py     # Threshold selection + OOS BB evaluation only
├── outputs/
│   ├── cointegration_results/
│   ├── price_series/
│   ├── z_scores/
│   ├── z_scores_data/
│   └── grid_backtest/
├── requirements.txt
└── README.md
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

### Quick Start (Unified Strategy)

Run cointegration, select best z-entry per pair via grid search, backtest with BB confirmation, and export CSV + plots + aggregate summary.

```bash
cd scripts
# Daily + rolling z-scores
python3 run_strategy.py --z rolling --tf D

# Weekly + expanding z-scores
python3 run_strategy.py --z expanding --tf W

# Monthly + rolling z-scores
python3 run_strategy.py --z rolling --tf M
```

### Thresholds Only (no plots nor aggregate)
```bash
# Finds best z-entry (IS) and evaluates OOS with BB confirmation
python3 run_grid_bb_thresholds.py --z rolling --tf W
```

### Workflow

The complete pipeline performs the following operations:

#### Phase 1: Cointegration Analysis (`check_cointegration.py`)
1. **Data Loading**: Reads price data from `data/indices_eur.csv` (auto-converts from Excel if needed)
2. **Log Transformation**: Converts price series to log-prices for stationarity analysis
3. **Unit Root Testing**: Applies Augmented Dickey-Fuller (ADF) test to identify I(1) series
4. **Correlation Prefilter**: Only test pairs with |Pearson corr| ≥ 0.8 on aligned log-prices
5. **Cointegration Testing**: Tests filtered I(1) pairs using Engle–Granger methodology
6. **Z-Score Calculation**: Computes normalized spreads using rolling and/or expanding windows (controlled via `--z`)
7. **Output Generation**: Saves cointegration results, plots, and z-score data

#### Phase 2: Grid + Backtest (`run_strategy.py` / `backtest_grid.py`)
1. **Grid Search (IS)**: z-entry ∈ [1.00, 2.00] step 0.05, maximize net PnL (with commissions)
2. **OOS Backtest**: Apply chosen z-entry to second half, with BB confirmation and BB-based stops
3. **Position Sizing**: Trade 1 unit of Y vs β units of X (β from cointegration)
4. **Costs**: Commission per leg = 0.01, charged on both entry and exit for both legs
5. **Exits**: z crosses 0 (profit) or BB-based stop (outer band ± 0.5σ)
6. **Plots**: Save z-score and spread plots with entry/exit markers (no GUI)
7. **Summary**: Save per-pair CSV and an aggregate summary (see below)

### Output Files

All outputs are timestamped (`YYYYMMDD_HHMMSS`) and saved to `outputs/`:

#### Cointegration Outputs
| File | Description |
|------|-------------|
| `cointegration_results_*.csv` | Full test results for all pairs (α, β, ADF statistics, p-values) |
| `price_series_*.png` | Raw price plots for all indices |
| `z_scores_*.png` | Z-scores (rolling and/or expanding per selection) with ±2σ bands |
| `z_scores_all_pairs_*.csv` | Combined z-score time series for all cointegrated pairs |

#### Backtest / Threshold Outputs
| File | Description |
|------|-------------|
| `grid_bb_thresholds_*.csv` | Best z-entry per pair (IS) + OOS BB metrics (from `run_grid_bb_thresholds.py`) |
| `strategy_grid_bb_*.csv` | Per-pair results from unified strategy (IS/OOS metrics) |
| `summary_*.json/csv` | Aggregate summary across pairs (see Metrics) |
| `z_with_trades_*_{ROLL/EXP}_{D/W/M}_*.png` | Z-score plot with entry/exit marks |
| `sample_trades_*_{ROLL/EXP}_{D/W/M}_*.png` | Spread plot with entry/exit marks |

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

### 5. Trading Signal Generation

#### 5.1 Entry Rules

| Condition | Signal | Position | Interpretation |
|-----------|--------|----------|----------------|
| $Z_t < -2$ | LONG SPREAD | Buy $Y$, Sell $X$ | Spread undervalued, expect upward reversion |
| $Z_t > +2$ | SHORT SPREAD | Sell $Y$, Buy $X$ | Spread overvalued, expect downward reversion |
| $-2 \leq Z_t \leq +2$ | NO ENTRY | Flat | Spread within normal range |

#### 5.2 Exit Rules

| Condition | Action | Reason |
|-----------|--------|--------|
| $Z_t \to 0$ (crosses threshold) | Close position | Mean reversion complete |
| $\|Z_t\| > 3$ | Force exit (stop-loss) | Risk management: spread diverging |
| Days held $\geq$ max holding | Force exit | Time-based stop |

#### 5.3 Risk Management Parameters

**Default Configuration:**
- Entry thresholds: $Z_t = \pm 2.0$ (±2 standard deviations)
- Exit threshold: $Z_t = 0.0$ (mean reversion)
- Stop loss: $\|Z_t\| = 3.0$ (±3 standard deviations)
- Max holding period: 60 trading days

These parameters can be adjusted based on:
- Historical signal performance
- Market volatility regime
- Risk tolerance
- Capital constraints

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

## CLI Options

`run_strategy.py`
```
--z {rolling,expanding}    # z-score source
--tf {D,W,M}               # timeframe: Daily, Weekly(Fri), Monthly(end)
```

`run_grid_bb_thresholds.py`
```
--z {rolling,expanding}
--tf {D,W,M}
```

Defaults: `--z rolling --tf D`.

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

## Theoretical Background (Brief)

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
2. **Static testing**: Cointegration tested on full sample; consider rolling cointegration for robustness
3. **Normality**: Z-score thresholds assume Gaussian spread distribution (tails can be fatter)

## Metrics & Aggregate Summary

The unified run saves an aggregate summary (`outputs/grid_backtest/summary_*.{csv,json}`) with:

- z_source, timeframe
- total_net_pnl, total_gross_pnl, total_commission
- num_trades, win_trades, loss_trades, win_rate
- avg_win, avg_loss, profit_factor, avg_days

Sharpe is reported per pair (proxy). Portfolio-level risk metrics (e.g., drawdown, portfolio Sharpe) can be added in future work.

## Development Roadmap

### Completed ✅
- [x] Engle–Granger cointegration testing (+ corr prefilter)
- [x] Dynamic z-score calculation (rolling/expanding OLS)
- [x] Grid search for z-entry + OOS backtest with BB confirmation
- [x] Commissions in PnL and per-trade metrics
- [x] Unified runner + plots + aggregate summary

### In Progress 🚧
- [ ] Portfolio-level risk (drawdown, volatility, portfolio Sharpe/Sortino)
- [ ] CLI flags for BB params, stop_extra, commission, max-hold

### Future Enhancements 🔮
- [ ] Walk-forward / rolling IS-OOS
- [ ] Portfolio optimization & capital allocation rules
- [ ] Kalman filter (time-varying β)
- [ ] Johansen test (multi-asset cointegration)
- [ ] Live trading integration

## References

1. Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing". *Econometrica*, 55(2), 251-276.
2. Alexander, C. (1999). "Optimal Hedging Using Cointegration". *Philosophical Transactions of the Royal Society A*, 357(1758), 2039-2058.
3. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. John Wiley & Sons.

## License

MIT License
