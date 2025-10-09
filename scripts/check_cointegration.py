import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

# Convert Excel to CSV if needed
def convert_excel_to_csv(excel_path="../data/indices final.xlsx", csv_path="../data/indices_eur.csv"):
    """
    Convert Excel file to CSV format.
    
    Args:
        excel_path: Path to the Excel file
        csv_path: Path for the output CSV file
    """
    try:
        # Check if Excel file exists
        if not os.path.exists(excel_path):
            print(f"Excel file not found: {excel_path}")
            return False
            
        # Check if CSV already exists
        if os.path.exists(csv_path):
            print(f"CSV file already exists: {csv_path}")
            return True
            
        # Read Excel file and get sheet names
        excel_file = pd.ExcelFile(excel_path)
        sheet_names = excel_file.sheet_names
        
        print(f"Found {len(sheet_names)} sheets in the Excel file:")
        for i, sheet_name in enumerate(sheet_names, 1):
            print(f"  {i}. {sheet_name}")
        
        # Find the "Indices in EUR" sheet
        target_sheet = "Indices in EUR"
        if target_sheet in sheet_names:
            sheet_name = target_sheet
            sheet_index = sheet_names.index(target_sheet) + 1
            print(f"\nFound '{target_sheet}' sheet (sheet #{sheet_index})")
        else:
            print(f"\nError: Sheet '{target_sheet}' not found in the Excel file.")
            print(f"Available sheets: {sheet_names}")
            return False
        
        # Read the target sheet
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
        # Display basic info about the data
        print(f"\nData shape: {df.shape} (rows x columns)")
        print(f"Columns: {list(df.columns)}")
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"\nData successfully saved to: {csv_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return False

# Load and prepare data
def load_data(csv_path="../data/indices_eur.csv", excel_path="../data/indices final.xlsx"):
    # Convert Excel to CSV if CSV doesn't exist
    if not os.path.exists(csv_path):
        print("CSV file not found. Converting Excel to CSV...")
        convert_excel_to_csv(excel_path, csv_path)
    
    df = pd.read_csv(csv_path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df.set_index(df.columns[0], inplace=True)
    df_log = np.log(df)
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Indices: {list(df.columns)}")
    return df, df_log

# Plot indices
def plot_indices(df_to_plot, save_plots=True):
    n_indices = len(df_to_plot.columns)
    fig, axes = plt.subplots(n_indices, 1, figsize=(15, 3*n_indices))

    for i, col in enumerate(df_to_plot.columns):
        axes[i].plot(df_to_plot.index, df_to_plot[col], linewidth=1.5)
        axes[i].set_title(f'{col} - Price Series')
        axes[i].set_ylabel('Price (EUR)')
        axes[i].grid(True, alpha=0.3)

    plt.xlabel('Date')
    plt.tight_layout()
    
    if save_plots:
        os.makedirs("../outputs/price_series", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"../outputs/price_series/price_series_{timestamp}.png", dpi=300, bbox_inches='tight')
        print(f"Price series plot saved to outputs/price_series/price_series_{timestamp}.png")
    
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
                beta = model.params.iloc[1]  # β coefficient
                alpha = model.params.iloc[0]  # α intercept
                
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
def analyze_spreads(pairs_data, log_data, save_plots=True):
    if not pairs_data:
        print("No cointegrated pairs to analyze.")
        return
    
    print(f"\nAnalyzing spreads for {len(pairs_data)} cointegrated pairs...")
    
    # Create outputs directories
    if save_plots:
        os.makedirs("../outputs/unnormalized_spreads", exist_ok=True)
        os.makedirs("../outputs/z_scores", exist_ok=True)
        os.makedirs("../outputs/z_scores_data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    if save_plots:
        plt.savefig(f"../outputs/unnormalized_spreads/unnormalized_spreads_{timestamp}.png", dpi=300, bbox_inches='tight')
        print(f"Unnormalized spreads plot saved to outputs/unnormalized_spreads/unnormalized_spreads_{timestamp}.png")
    
    plt.show()
    
    # Calculate and plot normalized spreads (z-scores)
    fig2, axes2 = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
    if num_plots == 1:
        axes2 = [axes2]
    
    print(f"\nZ-score analysis for all {num_plots} pairs:")
    
    # Store z-scores for saving
    z_scores_data = {}
    
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
        z_scores_data[pair['Pair']] = z_score
        
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
    
    if save_plots:
        plt.savefig(f"../outputs/z_scores/z_scores_{timestamp}.png", dpi=300, bbox_inches='tight')
        print(f"Z-scores plot saved to outputs/z_scores/z_scores_{timestamp}.png")
        
        # Save z-scores data to CSV
        z_scores_df = pd.DataFrame(z_scores_data)
        z_scores_df.to_csv(f"../outputs/z_scores_data/z_scores_{timestamp}.csv")
        print(f"Z-scores data saved to outputs/z_scores_data/z_scores_{timestamp}.csv")
    
    plt.show()
    
    return z_scores_data

def main(csv_path="../data/indices_eur.csv", excel_path="../data/indices final.xlsx"):
    df, df_log = load_data(csv_path, excel_path)
    plot_indices(df)
    adf_results = test_unit_roots(df_log)
    pairs = test_cointegration(df_log, adf_results)
    z_scores = analyze_spreads(pairs, df_log)
    return df, df_log, adf_results, pairs, z_scores


if __name__ == "__main__":
    main()
