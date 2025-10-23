import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime
import argparse

from check_cointegration import main as run_cointegration
from backtest_grid_risk_sized import (
    grid_search_best_z_risk_sized,
    simulate_pair_trades_risk_sized,
    trades_to_metrics_risk_sized,
    plot_trades_z_series,
    plot_trades_price_series,
    plot_multiple_trades_price_series,
    plot_multiple_trades_z_series,
)


def resample_to_timeframe(z_series: pd.Series, prices_df: pd.DataFrame, timeframe: str):
    """
    Resample series to timeframe: 'D' (daily), 'W' (weekly W-FRI), 'M' (month-end).
    """
    if timeframe.upper() == 'D':
        # Align indices (no resample)
        common = z_series.dropna().index.intersection(prices_df.index)
        return z_series.loc[common], prices_df.loc[common]
    rule = 'W-FRI' if timeframe.upper() == 'W' else 'M'
    z_res = z_series.resample(rule).last().dropna()
    px_res = prices_df.resample(rule).last().dropna()
    common = z_res.index.intersection(px_res.index)
    return z_res.loc[common], px_res.loc[common]


def run_strategy_risk_sized(z_type: str = 'rolling', timeframe: str = 'D', initial_capital: float = 10000.0, risk_per_trade: float = 100.0):
    """
    Run pair trading strategy with risk-based position sizing.
    
    Args:
        z_type: 'rolling' or 'expanding' z-score calculation
        timeframe: 'D', 'W', or 'M' for daily, weekly, monthly
        initial_capital: Initial capital amount in euros (default €10,000)
        risk_per_trade: Fixed risk amount per trade in euros (default €100)
    """
    # Pass mode to cointegration to skip extra logs/computation
    z_mode = 'rolling' if z_type.lower() == 'rolling' else ('expanding' if z_type.lower() == 'expanding' else 'both')
    df, df_log, pairs, z_roll, z_exp = run_cointegration(z_mode=z_mode)
    if not pairs:
        print("No cointegrated pairs found.")
        return

    prices = df.copy()
    pair_to_params = {p['Pair']: (p['Alpha'], p['Beta']) for p in pairs}

    os.makedirs("../outputs/grid_backtest", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = f"../outputs/grid_backtest/strategy_grid_bb_risk_sized_{timestamp}.csv"

    z_grid = [round(x, 2) for x in np.arange(1.00, 2.001, 0.05)]

    rows = []
    # Limit plotting to first 6 pairs for readability in 2x3 grid
    plot_limit = 6
    plotted = 0
    # Aggregates
    agg = {
        'net_pnl': 0.0,
        'gross_pnl': 0.0,
        'commission': 0.0,
        'num_trades': 0,
        'win_trades': 0,
        'loss_trades': 0,
        'sum_win': 0.0,
        'sum_loss': 0.0,
        'sum_days': 0.0,
        'total_risk': 0.0,
        'max_loss': 0.0,
    }

    z_dict = z_roll if z_type.lower() == 'rolling' else z_exp
    merged_price_items = []
    merged_z_items = []
    
    print(f"Running strategy with FIXED risk-based position sizing:")
    print(f"- Initial capital: €{initial_capital:,.0f}")
    print(f"- Fixed risk per trade: €{risk_per_trade:.0f}")
    print(f"- Z-score method: {z_type}")
    print(f"- Timeframe: {timeframe}")
    print(f"- Number of cointegrated pairs: {len(pairs)}")
    print("-" * 50)
    
    for pair_name, z_series in z_dict.items():
        x_name, y_name = pair_name.split(' - ')[0], pair_name.split(' - ')[1]
        alpha, beta = pair_to_params[pair_name]
        price_df = prices[[x_name, y_name]].dropna()

        # Resample to timeframe
        z_series_tf, price_df_tf = resample_to_timeframe(z_series, price_df, timeframe)
        if len(z_series_tf) < 50:
            print(f"{pair_name}: insufficient data after resample ({len(z_series_tf)} points) — skipping")
            continue

        # Choose best z on IS; evaluate OOS with BB confirmation and risk sizing
        z_best, metrics_is, metrics_oos = grid_search_best_z_risk_sized(
            pair_name, z_series_tf, price_df_tf, x_name, y_name, alpha, beta, z_grid,
            stop_extra=0.3, commission_per_leg=0.01, initial_capital=initial_capital, 
            risk_per_trade=risk_per_trade, split_at_middle=True
        )

        mid = len(z_series_tf.dropna()) // 2
        z_oos = z_series_tf.dropna().iloc[mid:]
        price_oos = price_df_tf.iloc[mid:]
        trades_oos_bb = simulate_pair_trades_risk_sized(
            pair_name, z_oos, price_oos, x_name, y_name, alpha, beta, z_best,
            stop_extra=0.3, commission_per_leg=0.01, initial_capital=initial_capital, 
            risk_per_trade=risk_per_trade, confirm='bb'
        )
        metrics_oos_bb = trades_to_metrics_risk_sized(trades_oos_bb)

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
            'OOS_max_loss': metrics_oos_bb['max_loss'],
            'OOS_total_risk': metrics_oos_bb['total_risk'],
            'OOS_avg_position_size': metrics_oos_bb['avg_position_size'],
        })

        if len(trades_oos_bb) > 0 and plotted < plot_limit:
            merged_z_items.append({
                'pair_name': pair_name,
                'z': z_oos,
                'trades': trades_oos_bb,
                'z_entry': z_best,
            })
            merged_price_items.append({
                'pair_name': pair_name,
                'prices': price_oos,
                'x_name': x_name,
                'y_name': y_name,
                'alpha': alpha,
                'beta': beta,
                'trades': trades_oos_bb,
            })
            plotted += 1

        print(f"{pair_name}: best z={z_best:.2f}, OOS PnL(BB)={metrics_oos_bb['net_pnl']:.1f}, "
              f"Sharpe={metrics_oos_bb['sharpe']:.2f}, Max Loss={metrics_oos_bb['max_loss']:.1f}")

        # Update aggregates
        agg['net_pnl'] += metrics_oos_bb['net_pnl']
        agg['gross_pnl'] += metrics_oos_bb['gross_pnl']
        agg['commission'] += metrics_oos_bb['commission']
        agg['num_trades'] += metrics_oos_bb['num_trades']
        agg['win_trades'] += metrics_oos_bb['win_trades']
        agg['loss_trades'] += metrics_oos_bb['loss_trades']
        agg['total_risk'] += metrics_oos_bb['total_risk']
        agg['max_loss'] = min(agg['max_loss'], metrics_oos_bb['max_loss'])
        
        if metrics_oos_bb['avg_win'] > 0 and metrics_oos_bb['win_trades'] > 0:
            agg['sum_win'] += metrics_oos_bb['avg_win'] * metrics_oos_bb['win_trades']
        if metrics_oos_bb['avg_loss'] < 0 and metrics_oos_bb['loss_trades'] > 0:
            agg['sum_loss'] += metrics_oos_bb['avg_loss'] * metrics_oos_bb['loss_trades']
        agg['sum_days'] += metrics_oos_bb['avg_days'] * max(metrics_oos_bb['num_trades'], 0)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Save merged plots (first N pairs)
    if merged_z_items:
        tag = f"{z_type.upper()}_{timeframe.upper()}_RISK{risk_per_trade}"
        merged_z_path = f"../outputs/z_scores/z_with_trades_MERGED_{tag}_{timestamp}.png"
        plot_multiple_trades_z_series(merged_z_items, merged_z_path, ncols=2)
    if merged_price_items:
        tag = f"{z_type.upper()}_{timeframe.upper()}_RISK{risk_per_trade}"
        merged_px_path = f"../outputs/grid_backtest/sample_trades_MERGED_{tag}_{timestamp}.png"
        plot_multiple_trades_price_series(merged_price_items, merged_px_path, ncols=2)

    # Aggregate summary
    win_rate = (agg['win_trades'] / agg['num_trades']) if agg['num_trades'] > 0 else 0.0
    avg_win = (agg['sum_win'] / agg['win_trades']) if agg['win_trades'] > 0 else 0.0
    avg_loss = (agg['sum_loss'] / agg['loss_trades']) if agg['loss_trades'] > 0 else 0.0
    profit_factor = (agg['sum_win'] / abs(agg['sum_loss'])) if agg['sum_loss'] != 0 else float('inf')
    avg_days = (agg['sum_days'] / agg['num_trades']) if agg['num_trades'] > 0 else 0.0
    
    summary = {
        'z_source': z_type,
        'timeframe': timeframe,
        'initial_capital': initial_capital,
        'risk_per_trade': risk_per_trade,
        'total_net_pnl': round(agg['net_pnl'], 4),
        'total_gross_pnl': round(agg['gross_pnl'], 4),
        'total_commission': round(agg['commission'], 4),
        'num_trades': int(agg['num_trades']),
        'win_trades': int(agg['win_trades']),
        'loss_trades': int(agg['loss_trades']),
        'win_rate': round(win_rate, 4),
        'avg_win': round(avg_win, 4),
        'avg_loss': round(avg_loss, 4),
        'profit_factor': round(profit_factor, 4) if np.isfinite(profit_factor) else None,
        'avg_days': round(avg_days, 4),
        'total_risk': round(agg['total_risk'], 4),
        'max_loss': round(agg['max_loss'], 4),
    }
    
    # Save summary CSV and JSON
    import json
    summary_csv = f"../outputs/grid_backtest/summary_risk_sized_{timestamp}.csv"
    summary_json = f"../outputs/grid_backtest/summary_risk_sized_{timestamp}.json"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("RISK-SIZED STRATEGY SUMMARY")
    print("="*60)
    print(f"Risk per trade: €{risk_per_trade:.0f}")
    print(f"Total trades: {agg['num_trades']}")
    print(f"Win rate: {win_rate:.1%}")
    print(f"Total P&L: {agg['net_pnl']:.2f}")
    print(f"Total risk taken: {agg['total_risk']:.2f}")
    print(f"Maximum single loss: {agg['max_loss']:.2f}")
    print(f"Profit factor: {profit_factor:.2f}" if np.isfinite(profit_factor) else "Profit factor: N/A")
    print(f"Average days held: {avg_days:.1f}")
    print(f"Saved summary to {summary_csv} and {summary_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pair trading strategy with risk-based position sizing.')
    parser.add_argument('--z', choices=['rolling', 'expanding'], default='rolling', help='Z-score source')
    parser.add_argument('--tf', choices=['D', 'W', 'M'], default='D', help='Timeframe: Daily (D), Weekly (W), Monthly (M)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital amount in euros (default 10000)')
    parser.add_argument('--risk', type=float, default=100.0, help='Fixed risk amount per trade in euros (default 100)')
    args = parser.parse_args()
    run_strategy_risk_sized(z_type=args.z, timeframe=args.tf, initial_capital=args.capital, risk_per_trade=args.risk)
