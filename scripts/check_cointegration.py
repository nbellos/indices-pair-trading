import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
import os
from datetime import datetime
from itertools import combinations

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

def compute_dynamic_ols_zscore(log_data, index1, index2, window=None, min_periods=252):
    """
    Compute z-score using rolling or expanding OLS to avoid look-ahead bias.
    
    Args:
        log_data: DataFrame of log-transformed prices
        index1: Name of X series (independent variable)
        index2: Name of Y series (dependent variable)
        window: If specified, uses rolling window. If None, uses expanding window
        min_periods: Minimum observations required to start calculation
    
    Returns:
        Series of z-scores indexed by date
    """
    # Extract and align the two price series
    y = log_data[index2].dropna()
    x = log_data[index1].dropna()
    common_index = y.index.intersection(x.index)
    y = y[common_index]
    x = x[common_index]
    
    z_scores = []
    start_idx = window if window else min_periods
    
    # For each time point, estimate parameters using only past data
    for i in range(start_idx, len(common_index)):
        # Determine window: rolling uses last 'window' obs, expanding uses all from start
        start = i - window if window else 0
        y_window = y.iloc[start:i]
        x_window = x.iloc[start:i]
        
        # OLS regression: Y = α + βX + ε
        X_window = sm.add_constant(x_window)
        model = sm.OLS(y_window, X_window).fit()
        alpha = model.params.iloc[0]  # Intercept
        beta = model.params.iloc[1]   # Slope (hedge ratio)
        
        # Compute current spread and its statistics from window
        spread_value = y.iloc[i] - (alpha + beta * x.iloc[i])
        spread_window = y_window - (alpha + beta * x_window)
        spread_mean = spread_window.mean()
        spread_std = spread_window.std()
        
        # Z-score normalization
        z_score = (spread_value - spread_mean) / spread_std if spread_std > 0 else 0.0
        z_scores.append(z_score)
    
    return pd.Series(z_scores, index=common_index[start_idx:])

def plot_z_scores(z_scores_rolling, z_scores_expanding):
    """
    Plot rolling vs expanding z-scores for all pairs and save to CSV.
    """
    if not z_scores_rolling:
        return
    
    num_plots = len(z_scores_rolling)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 4*num_plots))
    if num_plots == 1:
        axes = [axes]
    
    os.makedirs("../outputs/z_scores", exist_ok=True)
    os.makedirs("../outputs/z_scores_data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, pair_name in enumerate(z_scores_rolling.keys()):
        z_roll = z_scores_rolling[pair_name]
        z_exp = z_scores_expanding[pair_name]
        
        axes[i].plot(z_roll.index, z_roll, linewidth=1.5, alpha=0.8, label='Rolling (5yr)', color='blue')
        axes[i].plot(z_exp.index, z_exp, linewidth=1.5, alpha=0.8, label='Expanding (Recursive)', color='orange')
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1, label='Mean (0)')
        axes[i].axhline(y=2, color='red', linestyle=':', linewidth=1, label='±2σ')
        axes[i].axhline(y=-2, color='red', linestyle=':', linewidth=1)
        axes[i].set_title(f'{pair_name} - Z-Score (Rolling vs Expanding OLS)')
        axes[i].set_ylabel('Z-Score')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f"../outputs/z_scores/z_scores_{ts}.png", dpi=300, bbox_inches='tight')
    print(f"Z-scores plot saved to outputs/z_scores/z_scores_{ts}.png")
    
    # Combine all z-scores into a single CSV
    all_z_scores = {}
    for pair_name in z_scores_rolling.keys():
        # Sanitize column names
        safe_name = pair_name.replace(' - ', '_').replace(' ', '_').replace('&', 'and').replace('/', '_')
        all_z_scores[f'{safe_name}_Rolling'] = z_scores_rolling[pair_name]
        all_z_scores[f'{safe_name}_Expanding'] = z_scores_expanding[pair_name]
    
    combined_df = pd.DataFrame(all_z_scores)
    combined_df.to_csv(f"../outputs/z_scores_data/z_scores_all_pairs_{ts}.csv")
    print(f"Z-scores data saved to outputs/z_scores_data/z_scores_all_pairs_{ts}.csv")
    
    plt.show()

    
def compute_cointegration_and_spreads(log_data):
    """
    Identify cointegrated pairs using Engle-Granger methodology and compute dynamic z-scores.
    
    Args:
        log_data: DataFrame of log-transformed price series
    
    Returns:
        tuple: (cointegrated_pairs, z_scores_rolling, z_scores_expanding)
            - cointegrated_pairs: List of dicts with pair info and test statistics
            - z_scores_rolling: Dict of rolling z-score series by pair name
            - z_scores_expanding: Dict of expanding z-score series by pair name
    """
    # Step 0: Unit root testing (ADF) to identify I(1) series
    # H0: Series has unit root (non-stationary)
    # Reject H0 if p < 0.05 → series is I(0) (stationary)
    adf_rows = []
    for col in log_data.columns:
        adf_stat, p_value, _, _, _, _ = adfuller(log_data[col].dropna())
        is_stationary = p_value < 0.05
        adf_rows.append({'Index': col, 'p_value': round(p_value, 4), 'Is_Stationary': is_stationary})
    
    adf_data = pd.DataFrame(adf_rows)
    stationary = adf_data[adf_data['Is_Stationary'] == True]['Index'].tolist()
    non_stationary = adf_data[adf_data['Is_Stationary'] == False]['Index'].tolist()
    print(f"Stationary (I(0)): {len(stationary)} - {stationary}")
    print(f"Non-stationary (I(1)): {len(non_stationary)} - {non_stationary}")

    # Step 1: Engle-Granger cointegration test
    # Test all pairs of I(1) series for cointegration
    cointegrated_pairs = []
    all_pairs_results = []
    print(f"\nTesting cointegration for {len(non_stationary)} I(1) series...")
    
    for index1, index2 in combinations(non_stationary, 2):
        # Align data on common dates
        y = log_data[index2].dropna()
        x = log_data[index1].dropna()
        common_index = y.index.intersection(x.index)
        y = y[common_index]
        x = x[common_index]
        
        # Engle-Granger Step 1: Estimate cointegrating regression
        # Y_t = α + βX_t + ε_t (static OLS on full sample)
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1]   # Hedge ratio
        alpha = model.params.iloc[0]  # Constant term
        
        # Engle-Granger Step 2: Test residuals for stationarity
        spread = y - (alpha + beta * x)
        adf_stat, p_value, _, _, _, _ = adfuller(spread.dropna())
        result_row = {
            'Index_X': index1,
            'Index_Y': index2,
            'Alpha': alpha,
            'Beta': beta,
            'Spread_adf_stat': adf_stat,
            'Spread_p_value': p_value,
            'Cointegrated_5pct': bool(p_value < 0.05)
        }
        all_pairs_results.append(result_row)
        if p_value < 0.05:
            cointegrated_pairs.append({
                'Pair': f"{index1} - {index2}",
                'Beta': beta,
                'Alpha': alpha,
                'Spread_adf_stat': adf_stat,
                'Spread_p_value': p_value,
                'Cointegrated': True
            })
    if cointegrated_pairs:
        print(f"\nFound {len(cointegrated_pairs)} cointegrated pairs:")
        for pair in cointegrated_pairs:
            print(f"{pair['Pair']} - β={pair['Beta']:.4f}, p-value={pair['Spread_p_value']:.4f}")
    else:
        print("\nNo cointegrated pairs found.")
    
    # Step 2: compute spreads and z-scores for all pairs
    if not cointegrated_pairs:
        # Export full results even if nothing cointegrates
        try:
            os.makedirs("../outputs/cointegration_results", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pd.DataFrame(all_pairs_results).to_csv(f"../outputs/cointegration_results/cointegration_results_{ts}.csv", index=False)
            print(f"Saved cointegration results to outputs/cointegration_results/cointegration_results_{ts}.csv")
        except Exception as e:
            print(f"Warning: failed to save cointegration results CSV: {e}")
        return cointegrated_pairs, [], {}
    
    
    print(f"\nComputing rolling and expanding z-scores for {len(cointegrated_pairs)} cointegrated pairs...")
    z_scores_rolling = {}
    z_scores_expanding = {}
    
    for pair in cointegrated_pairs:
        pair_names = pair['Pair'].split(' - ')
        index1, index2 = pair_names[0], pair_names[1]
        pair_name = pair['Pair']
        
        # Compute z-scores using rolling OLS (5-year window = 1260 days)
        z_roll = compute_dynamic_ols_zscore(log_data, index1, index2, window=1260)
        z_scores_rolling[pair_name] = z_roll
        
        # Compute z-scores using expanding OLS (recursive, min 252 days)
        z_exp = compute_dynamic_ols_zscore(log_data, index1, index2, window=None, min_periods=252)
        z_scores_expanding[pair_name] = z_exp
        
        print(f"{pair_name}:")
        print(f"  Rolling   - Z-range: [{z_roll.min():.2f}, {z_roll.max():.2f}], {len(z_roll)} observations")
        print(f"  Expanding - Z-range: [{z_exp.min():.2f}, {z_exp.max():.2f}], {len(z_exp)} observations")

    # Save full pair-wise cointegration results
    try:
        os.makedirs("../outputs/cointegration_results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(all_pairs_results).to_csv(f"../outputs/cointegration_results/cointegration_results_{ts}.csv", index=False)
        print(f"\nSaved cointegration results to outputs/cointegration_results/cointegration_results_{ts}.csv")
    except Exception as e:
        print(f"Warning: failed to save cointegration results CSV: {e}")
    
    return cointegrated_pairs, z_scores_rolling, z_scores_expanding

def main(csv_path="../data/indices_eur.csv", excel_path="../data/indices final.xlsx"):
    df, df_log = load_data(csv_path, excel_path)
    # Compute cointegration with rolling and expanding z-scores
    pairs, z_scores_rolling, z_scores_expanding = compute_cointegration_and_spreads(df_log)
    
    if pairs:
        # Price series
        plot_indices(df)
        # Z-score plots (rolling vs expanding)
        plot_z_scores(z_scores_rolling, z_scores_expanding)

    return df, df_log, pairs, z_scores_rolling, z_scores_expanding


if __name__ == "__main__":
    main()
