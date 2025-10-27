# Indices Pair Trading Strategy

A quantitative trading system that identifies cointegrated index pairs and implements a pair trading strategy using z-score signals with Bollinger Bands confirmation.

## Overview

This project implements a sophisticated pair trading strategy that:

- Performs statistical cointegration analysis to identify mean-reverting pairs
- Uses dynamic z-score calculation with rolling or expanding windows
- Optimizes entry thresholds through grid search
- Implements Bollinger Bands confirmation for signal filtering
- Provides comprehensive backtesting with performance metrics

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd indices-pair-trading
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Main Strategy

Run the complete strategy with cointegration analysis and backtesting:

```bash
cd scripts
python run_strategy.py --z rolling --tf D
```

**Command line options:**
- `--z {rolling,expanding}`: Z-score calculation method
  - `rolling`: 5-year rolling window (adaptive)
  - `expanding`: recursive estimation (stable)
- `--tf {D,W,M}`: Timeframe for analysis
  - `D`: Daily (default)
  - `W`: Weekly (Friday close)
  - `M`: Monthly (end of month)

**Risk management options:**
- `--capital`: Initial capital amount in euros
- `--risk`: Fixed risk amount per trade in euros
- `--risk-pct`: Risk percentage of capital per trade (e.g., 1.0 for 1%)

### Examples

```bash
# Daily timeframe with rolling z-scores
python run_strategy.py --z rolling --tf D

# Weekly timeframe with expanding z-scores
python run_strategy.py --z expanding --tf W

# Risk-based position sizing
python run_strategy.py --capital 100000 --risk-pct 1.0 --z rolling --tf D
```

## Project Structure

```
indices-pair-trading/
├── data/
│   ├── example_indices.csv      # Example European indices data
│   └── indices_eur.csv          # Main dataset (if available)
├── scripts/
│   ├── check_cointegration.py   # Cointegration analysis
│   ├── backtest_grid.py         # Backtesting engine
│   ├── run_strategy.py          # Main strategy runner
│   └── parametric_normal_stub.py # Utility functions
├── outputs/                     # Generated results (ignored by git)
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Methodology

### 1. Cointegration Analysis

The system uses the Engle-Granger cointegration test:

1. **Stationarity Testing**: ADF tests identify I(1) non-stationary series
2. **Correlation Filtering**: Pre-filter pairs with correlation ≥ 0.8
3. **Cointegration Testing**: Engle-Granger methodology finds cointegrated pairs
4. **Z-Score Generation**: Calculate rolling/expanding window z-scores

### 2. Trading Strategy

**Entry Rules:**
- Z-score exceeds threshold (optimized via grid search: 1.00 to 2.00)
- Bollinger Bands confirmation required
- Long spread when z-score ≤ -threshold and spread ≤ lower BB
- Short spread when z-score ≥ +threshold and spread ≥ upper BB

**Exit Rules:**
- Z-score crosses zero (mean reversion complete)
- Stop-loss triggers (BB or z-score based)

**Risk Management:**
- Commission: 0.01 per leg
- Stop-loss: ±0.5σ beyond Bollinger Bands
- Position sizing: Market-neutral (1 unit Y vs β units X)

### 3. Optimization

- Grid search tests z-score thresholds from 1.00 to 2.00
- In-sample optimization on first half of data
- Out-of-sample evaluation on second half
- Selects threshold that maximizes net P&L

## Output Files

All outputs are timestamped and saved to the `outputs/` directory:

- `cointegration_results_*.csv`: Statistical test results
- `strategy_grid_bb_*.csv`: Detailed performance metrics per pair
- `summary_*.csv/json`: Aggregate performance summary
- `z_with_trades_*.png`: Z-score plots with trade markers
- `sample_trades_*.png`: Price spread plots showing trades
- `equity_curve_*.png`: Equity curve visualization

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0
- openpyxl >= 3.0.0

## Data Format

Input CSV should have:
- First column: Dates (parseable by pandas)
- Subsequent columns: Price series for each instrument
- Any frequency (daily, weekly, monthly)

## Limitations

- Cointegration relationships may break during structural changes
- Static cointegration testing (consider rolling cointegration for robustness)
- Z-score thresholds assume Gaussian spread distribution

## License

MIT License