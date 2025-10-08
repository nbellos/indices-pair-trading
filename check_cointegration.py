import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Load and prepare data
def load_data(csv_path="indices_eur.csv"):
    df = pd.read_csv(csv_path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)
    df_log = np.log(df)
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Indices: {list(df.columns)}")
    return df, df_log

# Plot indices
def plot_indices(df_to_plot):
    n_indices = len(df_to_plot.columns)
    fig, axes = plt.subplots(n_indices, 1, figsize=(15, 3*n_indices))

    for i, col in enumerate(df_to_plot.columns):
        axes[i].plot(df_to_plot.index, df_to_plot[col], linewidth=1.5)
        axes[i].set_title(f'{col} - Price Series')
        axes[i].set_ylabel('Price (EUR)')
        axes[i].grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

# Test unit roots
def test_unit_roots(log_data):
    results = []
    for col in log_data.columns:
        adf_stat, p_value, _, _, _, _ = adfuller(log_data[col].dropna())
        is_stationary = p_value < 0.05
        results.append({
            'Index': col,
            'p_value': round(p_value, 4),
            'Is_Stationary': is_stationary
        })
    adf_results = pd.DataFrame(results)
    stationary = adf_results[adf_results['Is_Stationary'] == True]['Index'].tolist()
    non_stationary = adf_results[adf_results['Is_Stationary'] == False]['Index'].tolist()
    
    print(f"Stationary (I(0)): {len(stationary)} - {stationary}")
    print(f"Non-stationary (I(1)): {len(non_stationary)} - {non_stationary}")
    return adf_results

# Test cointegration for pairs
def test_cointegration(log_data, adf_data):
    # Get non-stationary indices (I(1))
    non_stationary = adf_data[adf_data['Is_Stationary'] == False]['Index'].tolist()
    cointegrated_pairs = []
    
    print(f"\nTesting cointegration for {len(non_stationary)} I(1) series...")
    
    for i, index1 in enumerate(non_stationary):
        for j, index2 in enumerate(non_stationary):
            if i < j:  # Avoid duplicate pairs and self-pairs
                
                # OLS Regression: index2 = α + β * index1 + ε
                y = log_data[index2].dropna()
                x = log_data[index1].dropna()
                
                # Align data
                common_index = y.index.intersection(x.index)
                y = y[common_index]
                x = x[common_index]
                
                # Add constant for intercept
                X = sm.add_constant(x)
                
                # Run OLS regression
                model = sm.OLS(y, X).fit()
                beta = model.params[1]  # β coefficient
                alpha = model.params[0]  # α intercept
                
                # Calculate spread (residual): spread = y - (α + β * x)
                spread = y - (alpha + beta * x)
                
                # Test if spread is stationary (ADF test)
                adf_stat, p_value, _, _, _, _ = adfuller(spread.dropna())
                is_spread_stationary = p_value < 0.05
                
                if is_spread_stationary:
                    cointegrated_pairs.append({
                        'Pair': f"{index1} - {index2}",
                        'Beta': round(beta, 4),
                        'Alpha': round(alpha, 4),
                        'Spread_p_value': round(p_value, 4),
                        'Cointegrated': True
                    })
    
    if cointegrated_pairs:
        print(f"\nFound {len(cointegrated_pairs)} cointegrated pairs:")
        for pair in cointegrated_pairs:
            print(f"{pair['Pair']} - β={pair['Beta']}, p-value={pair['Spread_p_value']}")
    else:
        print("\nNo cointegrated pairs found.")
    
    return cointegrated_pairs

# Calculate z-scores and plot spreads for all pairs
def analyze_spreads(pairs_data, log_data):
    if not pairs_data:
        print("No cointegrated pairs to analyze.")
        return
    
    print(f"\nAnalyzing spreads for {len(pairs_data)} cointegrated pairs...")
    
    # Plot unnormalized spreads
    num_plots = len(pairs_data)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    for i, pair in enumerate(pairs_data):
        # Extract pair names
        pair_names = pair['Pair'].split(' - ')
        index1, index2 = pair_names[0], pair_names[1]
        
        # Get data
        y = log_data[index2].dropna()
        x = log_data[index1].dropna()
        
        # Align data
        common_index = y.index.intersection(x.index)
        y = y[common_index]
        x = x[common_index]
        
        # Calculate spread: spread = y - (α + β * x)
        spread = y - (pair['Alpha'] + pair['Beta'] * x)
        
        # Plot unnormalized spread
        axes[i].plot(spread.index, spread, linewidth=1, alpha=0.7, label='Spread')
        axes[i].axhline(y=spread.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {spread.mean():.4f}')
        axes[i].set_title(f'{pair["Pair"]} - Unnormalized Spread')
        axes[i].set_ylabel('Spread Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    
    # Calculate and plot normalized spreads (z-scores)
    fig2, axes2 = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
    if num_plots == 1:
        axes2 = [axes2]
    
    print(f"\nZ-score analysis for all {num_plots} pairs:")
    
    for i, pair in enumerate(pairs_data):
        # Extract pair names
        pair_names = pair['Pair'].split(' - ')
        index1, index2 = pair_names[0], pair_names[1]
        
        # Get data
        y = log_data[index2].dropna()
        x = log_data[index1].dropna()
        
        # Align data
        common_index = y.index.intersection(x.index)
        y = y[common_index]
        x = x[common_index]
        
        # Calculate spread
        spread = y - (pair['Alpha'] + pair['Beta'] * x)
        
        # Calculate z-score: (spread - mean) / std
        z_score = (spread - spread.mean()) / spread.std()
        
        # Plot normalized spread (z-score)
        axes2[i].plot(z_score.index, z_score, linewidth=1, alpha=0.7, label='Z-Score')
        axes2[i].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean (0)')
        axes2[i].axhline(y=2, color='green', linestyle=':', linewidth=1, label='±2σ')
        axes2[i].axhline(y=-2, color='green', linestyle=':', linewidth=1)
        axes2[i].set_title(f'{pair["Pair"]} - Normalized Spread (Z-Score)')
        axes2[i].set_ylabel('Z-Score')
        axes2[i].grid(True, alpha=0.3)
        axes2[i].legend()
        
        # Print statistics
        print(f"{pair['Pair']}: Mean={spread.mean():.4f}, Std={spread.std():.4f}, Z-range=[{z_score.min():.2f}, {z_score.max():.2f}]")
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def main(csv_path="indices_eur.csv"):
    df, df_log = load_data(csv_path)
    plot_indices(df)
    adf_results = test_unit_roots(df_log)
    pairs = test_cointegration(df_log, adf_results)
    analyze_spreads(pairs, df_log)


if __name__ == "__main__":
    main()
