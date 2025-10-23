# Indices Pair Trading Strategy

A comprehensive quantitative trading system that identifies cointegrated index pairs and implements a sophisticated pair trading strategy using z-score signals with Bollinger Bands confirmation. The system performs rigorous statistical analysis, optimizes entry thresholds via grid search, and provides detailed backtesting results with comprehensive performance metrics.

## Project Structure

```
indices-pair-trading/
├── data/
│   ├── indices final.xlsx            # Original Excel data file
│   └── indices_eur.csv               # Processed CSV data (Date + European indices)
├── scripts/
│   ├── check_cointegration.py        # Cointegration analysis & z-score calculation
│   ├── backtest_grid.py              # Core backtesting engine with grid search
│   ├── run_strategy.py               # Main strategy runner (recommended)
│   ├── run_grid_bb_thresholds.py     # Simplified threshold optimizer
│   └── parametric_normal_stub.py     # Utility functions
├── outputs/
│   ├── cointegration_results/        # Statistical test results
│   ├── price_series/                 # Price visualization plots
│   ├── z_scores/                     # Z-score analysis plots
│   ├── z_scores_data/                # Z-score time series data
│   └── grid_backtest/                # Backtesting results & performance metrics
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

### Main Strategy (Recommended)

The primary strategy combines cointegration analysis, grid search optimization, and Bollinger Bands confirmation:

```bash
cd scripts
# Daily timeframe with rolling z-scores (default)
python3 run_strategy.py --z rolling --tf D

# Weekly timeframe with expanding z-scores
python3 run_strategy.py --z expanding --tf W

# Monthly timeframe with rolling z-scores
python3 run_strategy.py --z rolling --tf M
```

### Alternative: Threshold Optimization Only

For quick threshold optimization without comprehensive outputs:

```bash
# Find optimal z-score thresholds only
python3 run_grid_bb_thresholds.py --z rolling --tf D
```

**Note**: `run_strategy.py` is the recommended approach as it provides comprehensive analysis, plotting, and aggregate performance metrics.

## Strategy Overview

This system implements a sophisticated pair trading strategy that combines:

1. **Statistical Foundation**: Engle-Granger cointegration testing to identify mean-reverting pairs
2. **Dynamic Z-Scores**: Rolling or expanding window z-score calculation to avoid look-ahead bias
3. **Grid Search Optimization**: Systematic testing of z-score thresholds (1.00 to 2.00) to find optimal entry points
4. **Bollinger Bands Confirmation**: Additional signal filtering using spread-based Bollinger Bands
5. **Risk Management**: Stop-losses and profit targets based on both z-scores and Bollinger Bands

### Complete Workflow

#### Phase 1: Cointegration Analysis
1. **Data Processing**: Load European indices data and convert to log-prices
2. **Stationarity Testing**: Apply ADF tests to identify I(1) non-stationary series
3. **Correlation Filtering**: Pre-filter pairs with correlation ≥ 0.8
4. **Cointegration Testing**: Use Engle-Granger methodology to find cointegrated pairs
5. **Z-Score Generation**: Calculate rolling/expanding window z-scores for trading signals

#### Phase 2: Strategy Optimization & Backtesting
1. **Grid Search**: Test z-score thresholds from 1.00 to 2.00 (step 0.05) on in-sample data
2. **Optimization**: Select threshold that maximizes net P&L
3. **Out-of-Sample Testing**: Apply optimal threshold to second half of data
4. **Bollinger Bands Confirmation**: Require spread to exceed BB bands for entry
5. **Trade Simulation**: Execute trades with realistic position sizing and commissions
6. **Performance Analysis**: Calculate comprehensive metrics and generate visualizations

### Key Features

- **Dual Confirmation**: Z-score signals + Bollinger Bands confirmation
- **Dynamic Parameters**: Rolling/expanding window estimation prevents look-ahead bias
- **Realistic Trading**: Includes commissions, position sizing, and risk management
- **Comprehensive Outputs**: CSV results, performance plots, and aggregate summaries
- **Multiple Timeframes**: Daily, weekly, or monthly analysis
- **Statistical Rigor**: Proper cointegration testing and stationarity analysis

### Output Files

All outputs are timestamped and organized in the `outputs/` directory:

#### Analysis Results
- `cointegration_results_*.csv` - Statistical test results for all pairs
- `price_series_*.png` - Price evolution plots for all indices
- `z_scores_*.png` - Z-score analysis with statistical bands
- `z_scores_all_pairs_*.csv` - Complete z-score time series data

#### Backtesting Results
- `strategy_grid_bb_*.csv` - Detailed per-pair performance metrics
- `summary_*.json/csv` - Aggregate performance summary across all pairs
- `z_with_trades_*_*.png` - Z-score plots with trade entry/exit markers
- `sample_trades_*_*.png` - Spread plots showing actual trade executions

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

#### 5.1 Entry Rules (with Bollinger Bands Confirmation)

| Z-Score Condition | Bollinger Bands Condition | Signal | Position | Interpretation |
|-------------------|---------------------------|--------|----------|----------------|
| $Z_t \leq -z_{entry}$ | Spread $\leq$ Lower BB | LONG SPREAD | +1 unit Y, -β units X | Spread undervalued, expect upward reversion |
| $Z_t \geq +z_{entry}$ | Spread $\geq$ Upper BB | SHORT SPREAD | -1 unit Y, +β units X | Spread overvalued, expect downward reversion |
| Other cases | - | NO ENTRY | Flat | Insufficient signal strength |

**Where:**
- $z_{entry}$ is optimized via grid search (typically 1.00 to 2.00)
- Bollinger Bands: 20-period moving average ± 1.0 standard deviation
- Both conditions must be met for trade entry

#### 5.2 Exit Rules

| Condition | Action | Reason |
|-----------|--------|--------|
| $Z_t \to 0$ (crosses zero) | Close position | Mean reversion complete |
| BB Stop-loss hit | Force exit | Spread exceeds BB ± 0.5σ bands |
| Z-score stop-loss | Force exit | $|Z_t| \geq |Z_{entry}| + 0.3$ |

#### 5.3 Risk Management Parameters

**Default Configuration:**
- Entry thresholds: Optimized via grid search (1.00 to 2.00)
- Exit threshold: $Z_t = 0.0$ (mean reversion)
- BB Stop-loss: ±0.5σ beyond main Bollinger Bands
- Z-score Stop-loss: $|Z_{entry}| + 0.3$
- Commission: 0.01 per leg (entry and exit)
- Position sizing: Market-neutral (1 unit Y vs β units X)

**Optimization Process:**
- Grid search tests 21 different z-score thresholds
- Selects threshold that maximizes net P&L on in-sample data
- Evaluates performance on out-of-sample data

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

## Command Line Options

### Main Strategy Runner (`run_strategy.py`)
```bash
python3 run_strategy.py [options]

Options:
  --z {rolling,expanding}    Z-score calculation method
                             rolling: 5-year rolling window (adaptive)
                             expanding: recursive estimation (stable)
  --tf {D,W,M}              Timeframe for analysis
                             D: Daily (default)
                             W: Weekly (Friday close)
                             M: Monthly (end of month)

Default: --z rolling --tf D
```

### Threshold Optimizer (`run_grid_bb_thresholds.py`)
```bash
python3 run_grid_bb_thresholds.py [options]

Options:
  --z {rolling,expanding}    Same as above
  --tf {D,W,M}              Same as above

Note: This script provides threshold optimization only, without comprehensive outputs.
```

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

## Current Status & Roadmap

### Completed Features ✅
- [x] **Cointegration Analysis**: Engle-Granger testing with correlation pre-filtering
- [x] **Dynamic Z-Scores**: Rolling and expanding window calculation to prevent look-ahead bias
- [x] **Grid Search Optimization**: Systematic testing of z-score thresholds (1.00 to 2.00)
- [x] **Bollinger Bands Integration**: Dual confirmation system for entry signals
- [x] **Comprehensive Backtesting**: Realistic trading simulation with commissions and risk management
- [x] **Performance Analytics**: Detailed metrics, visualizations, and aggregate summaries
- [x] **Multiple Timeframes**: Daily, weekly, and monthly analysis capabilities
- [x] **Automated Workflow**: Single-command execution with comprehensive outputs

### Potential Enhancements 🔮
- [ ] **Portfolio-Level Risk Metrics**: Portfolio drawdown, volatility, and Sharpe ratios
- [ ] **Advanced Parameter Tuning**: Configurable BB parameters, stop-losses, and commissions
- [ ] **Walk-Forward Analysis**: Rolling in-sample/out-of-sample validation
- [ ] **Multi-Asset Cointegration**: Johansen test for multiple asset relationships
- [ ] **Dynamic Hedging**: Kalman filter for time-varying beta estimation
- [ ] **Capital Allocation**: Portfolio optimization and position sizing rules
- [ ] **Real-Time Integration**: Live data feeds and trading system connectivity

### System Capabilities

This system provides a complete quantitative trading framework suitable for:
- **Research & Development**: Testing new pair trading strategies
- **Backtesting**: Historical performance evaluation with realistic assumptions
- **Risk Analysis**: Comprehensive risk metrics and scenario analysis
- **Strategy Optimization**: Systematic parameter tuning and validation

## References

1. Engle, R.F. and Granger, C.W.J. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing". *Econometrica*, 55(2), 251-276.
2. Alexander, C. (1999). "Optimal Hedging Using Cointegration". *Philosophical Transactions of the Royal Society A*, 357(1758), 2039-2058.
3. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. John Wiley & Sons.

## License

MIT License
