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

def plot_unnormalized_spreads(precomputed_spreads):
    """
    Plot and save unnormalized spreads given list of (pair_str, spread_series).
    """
    if not precomputed_spreads:
        return
    num_plots = len(precomputed_spreads)
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
    if num_plots == 1:
        axes = [axes]
    os.makedirs("../outputs/unnormalized_spreads", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (pair_str, spread) in enumerate(precomputed_spreads):
        axes[i].plot(spread.index, spread, linewidth=1, alpha=0.7, label='Spread')
        axes[i].axhline(y=spread.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {spread.mean():.4f}')
        axes[i].set_title(f'{pair_str} - Unnormalized Spread')
        axes[i].set_ylabel('Spread Value')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f"../outputs/unnormalized_spreads/unnormalized_spreads_{ts}.png", dpi=300, bbox_inches='tight')
    print(f"Unnormalized spreads plot saved to outputs/unnormalized_spreads/unnormalized_spreads_{ts}.png")
    plt.show()

def plot_z_scores(precomputed_spreads, z_scores):
    """
    Plot and save z-score series for each pair and export the z-score data to CSV.
    """
    if not precomputed_spreads:
        return
    num_plots = len(precomputed_spreads)
    fig2, axes2 = plt.subplots(num_plots, 1, figsize=(15, 3*num_plots))
    if num_plots == 1:
        axes2 = [axes2]
    os.makedirs("../outputs/z_scores", exist_ok=True)
    os.makedirs("../outputs/z_scores_data", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (pair_str, _) in enumerate(precomputed_spreads):
        z = z_scores[pair_str]
        axes2[i].plot(z.index, z, linewidth=1, alpha=0.7, label='Z-Score')
        axes2[i].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean (0)')
        axes2[i].axhline(y=2, color='green', linestyle=':', linewidth=1, label='±2σ')
        axes2[i].axhline(y=-2, color='green', linestyle=':', linewidth=1)
        axes2[i].set_title(f'{pair_str} - Normalized Spread (Z-Score)')
        axes2[i].set_ylabel('Z-Score')
        axes2[i].grid(True, alpha=0.3)
        axes2[i].legend()
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(f"../outputs/z_scores/z_scores_{ts}.png", dpi=300, bbox_inches='tight')
    print(f"Z-scores plot saved to outputs/z_scores/z_scores_{ts}.png")
    pd.DataFrame(z_scores).to_csv(f"../outputs/z_scores_data/z_scores_{ts}.csv")
    print(f"Z-scores data saved to outputs/z_scores_data/z_scores_{ts}.csv")
    plt.show()

    
def compute_cointegration_and_spreads(log_data):
    """
    Find cointegrated pairs, compute spreads and z-scores.
    Returns (cointegrated_pairs, precomputed_spreads, z_scores_data).
    - precomputed_spreads: list of (pair_str, spread_series)
    - z_scores_data: dict of {pair_str: z_score_series}
    """
    # Step 0: Unit root tests (ADF) and prefilter to I(1)
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

    # Step 1: Find cointegrated pairs
    
    cointegrated_pairs = []
    all_pairs_results = []
    print(f"\nTesting cointegration for {len(non_stationary)} I(1) series...")
    for index1, index2 in combinations(non_stationary, 2):
        y = log_data[index2].dropna()
        x = log_data[index1].dropna()
        common_index = y.index.intersection(x.index)
        y = y[common_index]
        x = x[common_index]
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        beta = model.params.iloc[1]
        alpha = model.params.iloc[0]
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
    
    # Step 2: compute spreads and z-scores for all pairs (no plotting)
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
    print(f"\nAnalyzing spreads for {len(cointegrated_pairs)} cointegrated pairs...")
    precomputed = []
    z_scores_data = {}
    for pair in cointegrated_pairs:
        pair_names = pair['Pair'].split(' - ')
        index1, index2 = pair_names[0], pair_names[1]
        y = log_data[index2].dropna()
        x = log_data[index1].dropna()
        common_index = y.index.intersection(x.index)
        y = y[common_index]
        x = x[common_index]
        spread = y - (pair['Alpha'] + pair['Beta'] * x)
        precomputed.append((pair['Pair'], spread))
        spread_std = spread.std()
        if spread_std is None or np.isclose(spread_std, 0.0):
            print(f"{pair['Pair']}: Skipping z-score (zero or near-zero spread std)")
            continue
        z_score = (spread - spread.mean()) / spread_std
        z_scores_data[pair['Pair']] = z_score
        print(f"{pair['Pair']}: Mean={spread.mean():.4f}, Std={spread_std:.4f}, Z-range=[{z_score.min():.2f}, {z_score.max():.2f}]")

    # Save full pair-wise results including non-cointegrated
    try:
        os.makedirs("../outputs/cointegration_results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pd.DataFrame(all_pairs_results).to_csv(f"../outputs/cointegration_results/cointegration_results_{ts}.csv", index=False)
        print(f"Saved cointegration results to outputs/cointegration_results/cointegration_results_{ts}.csv")
    except Exception as e:
        print(f"Warning: failed to save cointegration results CSV: {e}")
    return cointegrated_pairs, precomputed, z_scores_data

def main(csv_path="../data/indices_eur.csv", excel_path="../data/indices final.xlsx"):
    df, df_log = load_data(csv_path, excel_path)
    # Compute cointegration, spreads and z-scores (includes internal ADF)
    pairs, precomputed_spreads, z_scores = compute_cointegration_and_spreads(df_log)
    
    # Optional plotting centralized here for clarity
    if pairs:
        # Price series
        plot_indices(df)
        # Unnormalized spreads
        plot_unnormalized_spreads(precomputed_spreads)

        # Z-score plots
        plot_z_scores(precomputed_spreads, z_scores)

    return df, df_log, pairs, z_scores


if __name__ == "__main__":
    main()
