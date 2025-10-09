# Pairs Trading Strategy

A comprehensive pairs trading implementation using cointegration analysis for identifying and trading cointegrated pairs of financial instruments.

## Project Structure

```
pair-trading-strategy/
├── data/                    # Data files (CSV, Excel)
├── scripts/                 # Main analysis scripts
│   ├── check_cointegration.py
│   └── extract_indices_csv.py
├── tests/                   # Unit tests
│   └── test_cointegration.py
├── outputs/                 # Generated plots and results
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Features

- **Cointegration Testing**: ADF tests for unit roots and cointegration
- **Spread Analysis**: Calculate and visualize spreads for all pairs
- **Z-Score Analysis**: Normalized spread analysis with trading signals
- **Automated Output**: Saves plots and data to outputs folder
- **Modular Design**: Clean separation of data, analysis, and outputs

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

### Basic Analysis

Run the main cointegration analysis:

```bash
cd scripts
python check_cointegration.py
```

This will:
- Load data from `../data/indices_eur.csv`
- Test for unit roots in all series
- Find cointegrated pairs
- Generate spread and z-score plots
- Save all outputs to `../outputs/` with timestamps

### Data Requirements

Place your data files in the `data/` folder. The expected format is:
- CSV file with dates in the first column
- Price data for different indices/instruments in subsequent columns
- Date column should be parseable by pandas

### Outputs

The analysis generates:
- `price_series_YYYYMMDD_HHMMSS.png`: Price series plots
- `unnormalized_spreads_YYYYMMDD_HHMMSS.png`: Raw spread plots
- `z_scores_YYYYMMDD_HHMMSS.png`: Z-score plots with trading signals
- `z_scores_YYYYMMDD_HHMMSS.csv`: Z-score data for further analysis

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Next Steps

- [ ] Implement signal generation and backtesting
- [ ] Add risk management and position sizing
- [ ] Create walk-forward analysis
- [ ] Add more robust cointegration tests (Johansen)
- [ ] Implement real-time monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license here]