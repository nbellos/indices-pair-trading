import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from check_cointegration import main as run_cointegration
from backtest_grid import grid_search_best_z, simulate_pair_trades, trades_to_metrics


def resample_to_timeframe(z_series: pd.Series, prices_df: pd.DataFrame, timeframe: str):
    if timeframe.upper() == 'D':
        common = z_series.dropna().index.intersection(prices_df.index)
        return z_series.loc[common], prices_df.loc[common]
    rule = 'W-FRI' if timeframe.upper() == 'W' else 'M'
    z_res = z_series.resample(rule).last().dropna()
    px_res = prices_df.resample(rule).last().dropna()
    common = z_res.index.intersection(px_res.index)
    return z_res.loc[common], px_res.loc[common]


def main(z_type: str = 'rolling', timeframe: str = 'D'):
    z_mode = 'rolling' if z_type.lower() == 'rolling' else ('expanding' if z_type.lower() == 'expanding' else 'both')
    df, df_log, pairs, z_roll, z_exp = run_cointegration(z_mode=z_mode)
    if not pairs:
        print("No cointegrated pairs found.")
        return

    prices = df.copy()
    pair_to_params = {p['Pair']: (p['Alpha'], p['Beta']) for p in pairs}

    os.makedirs("../outputs/grid_backtest", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"../outputs/grid_backtest/grid_bb_thresholds_{timestamp}.csv"

    z_grid = [round(x, 2) for x in np.arange(1.00, 2.001, 0.05)]

    rows = []
    z_dict = z_roll if z_type.lower() == 'rolling' else z_exp
    for pair_name, z_series in z_dict.items():
        x_name, y_name = pair_name.split(' - ')[0], pair_name.split(' - ')[1]
        alpha, beta = pair_to_params[pair_name]
        price_df = prices[[x_name, y_name]].dropna()

        # Resample timeframe
        z_series_tf, price_df_tf = resample_to_timeframe(z_series, price_df, timeframe)
        if len(z_series_tf) < 50:
            print(f"{pair_name}: insufficient data after resample ({len(z_series_tf)} points) — skipping")
            continue

        # Choose best z on IS (first half) and evaluate OOS (second half)
        z_best, metrics_is, metrics_oos = grid_search_best_z(
            pair_name, z_series_tf, price_df_tf, x_name, y_name, alpha, beta, z_grid,
            stop_extra=0.5, commission_per_leg=0.01, split_at_middle=True
        )

        # Evaluate with BB confirmation on OOS using chosen z
        mid = len(z_series_tf.dropna()) // 2
        z_oos = z_series_tf.dropna().iloc[mid:]
        price_oos = price_df_tf.iloc[mid:]
        trades_oos_bb = simulate_pair_trades(
            pair_name, z_oos, price_oos, x_name, y_name, alpha, beta, z_best,
            stop_extra=0.5, commission_per_leg=0.01, confirm='bb'
        )
        metrics_oos_bb = trades_to_metrics(trades_oos_bb)

        rows.append({
            'Pair': pair_name,
            'Alpha': alpha,
            'Beta': beta,
            'BestZ_IS': z_best,
            'IS_netPnL': metrics_is['net_pnl'],
            'IS_Sharpe': metrics_is['sharpe'],
            'IS_trades': metrics_is['num_trades'],
            'OOS_netPnL_BB': metrics_oos_bb['net_pnl'],
            'OOS_Sharpe_BB': metrics_oos_bb['sharpe'],
            'OOS_trades_BB': metrics_oos_bb['num_trades'],
        })
        print(f"{pair_name}: best z={z_best:.2f}, OOS PnL(BB)={metrics_oos_bb['net_pnl']:.1f}, Sharpe={metrics_oos_bb['sharpe']:.2f}")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved Grid+BB thresholds to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run grid threshold selection with BB evaluation.')
    parser.add_argument('--z', choices=['rolling', 'expanding'], default='rolling', help='Z-score source')
    parser.add_argument('--tf', choices=['D', 'W', 'M'], default='D', help='Timeframe: D/W/M')
    args = parser.parse_args()
    main(z_type=args.z, timeframe=args.tf)
