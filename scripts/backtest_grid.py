import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class TradeResult:
    pair: str
    direction: int  # 1 long spread, -1 short spread
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_z: float
    exit_z: float
    pnl: float
    gross_pnl: float
    commission: float
    days_held: int


def simulate_pair_trades(
    pair_name: str,
    z: pd.Series,
    prices: pd.DataFrame,
    x_name: str,
    y_name: str,
    alpha: float,
    beta: float,
    z_entry: float,
    stop_extra: float = 0.5,
    commission_per_leg: float = 0.01,
    confirm: str = 'none',  # 'none' | 'bb'
    bb_window: int = 20,
    bb_num_std: float = 1.0,
) -> list:
    """
    Simulate trades on spread Y - (alpha + beta*X) using z-score series.
    Position sizing: 1 unit on Y leg, beta units on X leg (hedge ratio).
    Entry when |z| >= z_entry; Exit on z crossing 0; Stop-loss when |z| >= z_entry + stop_extra.
    Commissions: charged per leg on both entry and exit at fixed amount.
    """
    z = z.dropna()
    common_idx = z.index.intersection(prices.index)
    z = z.loc[common_idx]
    px_y = prices[y_name].loc[common_idx]
    px_x = prices[x_name].loc[common_idx]

    # Spread series for confirmations
    spread = px_y - (alpha + beta * px_x)
    # Bollinger Bands on spread
    if confirm == 'bb':
        mid = spread.rolling(bb_window, min_periods=bb_window).mean()
        vol = spread.rolling(bb_window, min_periods=bb_window).std(ddof=0)
        upper = mid + bb_num_std * vol
        lower = mid - bb_num_std * vol
        # extra stop thresholds
        upper_stop = upper + 0.5 * vol
        lower_stop = lower - 0.5 * vol
    else:
        mid = vol = upper = lower = upper_stop = lower_stop = None
    # No EMA variant retained

    trades: list[TradeResult] = []
    in_pos = 0
    entry_i = None
    entry_z = None

    for i, date in enumerate(z.index):
        zi = float(z.iloc[i])
        if in_pos == 0:
            if zi >= z_entry:
                # require confirmations if set
                ok = True
                if confirm == 'bb':
                    ok = (not pd.isna(upper.iloc[i])) and (spread.iloc[i] >= upper.iloc[i])
                if ok:
                    in_pos = -1  # short spread
                    entry_i = i
                    entry_z = zi
            elif zi <= -z_entry:
                ok = True
                if confirm == 'bb':
                    ok = (not pd.isna(lower.iloc[i])) and (spread.iloc[i] <= lower.iloc[i])
                if ok:
                    in_pos = 1   # long spread
                    entry_i = i
                    entry_z = zi
        else:
            # stop-loss
            stop_hit = False
            if confirm == 'bb':
                if in_pos == 1 and (not pd.isna(lower_stop.iloc[i])) and spread.iloc[i] <= lower_stop.iloc[i]:
                    stop_hit = True
                if in_pos == -1 and (not pd.isna(upper_stop.iloc[i])) and spread.iloc[i] >= upper_stop.iloc[i]:
                    stop_hit = True
            else:
                if abs(zi) >= abs(entry_z) + stop_extra:
                    stop_hit = True
            if stop_hit:
                exit_i = i
                trades.append(_close_trade(pair_name, in_pos, entry_i, exit_i, entry_z, zi, z, px_x, px_y, beta, commission_per_leg))
                in_pos = 0
                entry_i = None
                entry_z = None
                continue
            # profit exit at z crossing 0
            if (in_pos == 1 and zi >= 0) or (in_pos == -1 and zi <= 0):
                exit_i = i
                trades.append(_close_trade(pair_name, in_pos, entry_i, exit_i, entry_z, zi, z, px_x, px_y, beta, commission_per_leg))
                in_pos = 0
                entry_i = None
                entry_z = None

    return trades


def plot_trades_z_series(
    pair_name: str,
    z: pd.Series,
    trades: list,
    z_entry: float,
    output_path: str,
):
    """
    Save a plot of z-score time series with entry/exit markers and ±z_entry bands.
    """
    if z is None or z.empty:
        return
    z = z.dropna()
    plt.figure(figsize=(12, 5))
    plt.plot(z.index, z.values, label='Z-Score', color='tab:blue')
    plt.axhline(z_entry, color='tab:red', linestyle='--', linewidth=1, label=f'+{z_entry:.2f}')
    plt.axhline(-z_entry, color='tab:green', linestyle='--', linewidth=1, label=f'-{z_entry:.2f}')
    plt.axhline(0.0, color='black', linestyle='-', linewidth=0.8)
    # Mark entries/exits
    for t in trades:
        try:
            e = t.entry_date
            x = t.exit_date
            ze = float(z.loc[e]) if e in z.index else None
            zx = float(z.loc[x]) if x in z.index else None
            if ze is not None:
                plt.scatter(e, ze, color=('tab:red' if t.direction == -1 else 'tab:green'), marker='^', s=60)
            if zx is not None:
                plt.scatter(x, zx, color='black', marker='x', s=50)
        except Exception:
            continue
    plt.title(f'{pair_name} — Z-Score with Entries (triangles) and Exits (x)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trades_price_series(
    pair_name: str,
    prices: pd.DataFrame,
    x_name: str,
    y_name: str,
    alpha: float,
    beta: float,
    trades: list,
    output_path: str,
):
    """
    Save a plot of the spread price series with entry/exit markers based on closing prices.
    """
    common_idx = prices.index
    spread = prices[y_name].loc[common_idx] - (alpha + beta * prices[x_name].loc[common_idx])
    plt.figure(figsize=(12, 5))
    plt.plot(spread.index, spread.values, label='Spread (Y - (α + β·X))', color='tab:purple')
    for t in trades:
        try:
            e = t.entry_date
            x = t.exit_date
            if e in spread.index:
                plt.scatter(e, float(spread.loc[e]), color=('tab:red' if t.direction == -1 else 'tab:green'), marker='^', s=60)
            if x in spread.index:
                plt.scatter(x, float(spread.loc[x]), color='black', marker='x', s=50)
        except Exception:
            continue
    plt.title(f'{pair_name} — Spread with Entries (triangles) and Exits (x)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_multiple_trades_price_series(
    items: list,
    output_path: str,
    ncols: int = 2,
):
    """
    Save a single figure with multiple subplots of spread price series and entry/exit markers.
    Each item in `items` must be a dict with keys:
      - pair_name, prices (DataFrame with x_name/y_name columns), x_name, y_name, alpha, beta, trades
    """
    if not items:
        return
    n = len(items)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for idx, item in enumerate(items):
        ax = axes_flat[idx]
        pair_name = item['pair_name']
        prices = item['prices']
        x_name = item['x_name']
        y_name = item['y_name']
        alpha = item['alpha']
        beta = item['beta']
        trades = item['trades']
        common_idx = prices.index
        spread = prices[y_name].loc[common_idx] - (alpha + beta * prices[x_name].loc[common_idx])
        ax.plot(spread.index, spread.values, label='Spread', color='tab:purple')
        for t in trades:
            try:
                e = t.entry_date
                x = t.exit_date
                if e in spread.index:
                    ax.scatter(e, float(spread.loc[e]), color=('tab:red' if t.direction == -1 else 'tab:green'), marker='^', s=50)
                if x in spread.index:
                    ax.scatter(x, float(spread.loc[x]), color='black', marker='x', s=45)
            except Exception:
                continue
        ax.set_title(pair_name)
        ax.legend(loc='upper right')
    # Hide any unused subplots
    for j in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_multiple_trades_z_series(
    items: list,
    output_path: str,
    ncols: int = 2,
):
    """
    Save a single figure with multiple subplots of z-score series and entry/exit markers.
    Each item in `items` must be a dict with keys:
      - pair_name, z (Series), trades, z_entry (float)
    """
    if not items:
        return
    n = len(items)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for idx, item in enumerate(items):
        ax = axes_flat[idx]
        pair_name = item['pair_name']
        z = item['z'].dropna()
        trades = item['trades']
        z_entry = float(item['z_entry'])
        ax.plot(z.index, z.values, label='Z-Score', color='tab:blue')
        ax.axhline(z_entry, color='tab:red', linestyle='--', linewidth=1)
        ax.axhline(-z_entry, color='tab:green', linestyle='--', linewidth=1)
        ax.axhline(0.0, color='black', linestyle='-', linewidth=0.8)
        wins = 0
        for t in trades:
            try:
                e = t.entry_date
                x = t.exit_date
                ze = float(z.loc[e]) if e in z.index else None
                zx = float(z.loc[x]) if x in z.index else None
                # Entry marker (direction color)
                if ze is not None:
                    entry_color = 'tab:red' if t.direction == -1 else 'tab:green'
                    ax.scatter(e, ze, color=entry_color, marker='^', s=50)
                # Exit marker colored by PnL, with annotation
                if zx is not None:
                    exit_color = 'green' if t.pnl > 0 else 'red'
                    ax.scatter(x, zx, color=exit_color, marker='x', s=55)
                    ax.annotate(f"{t.pnl:.0f}", (x, zx), textcoords="offset points", xytext=(4, 4), fontsize=8, color=exit_color)
                    if t.pnl > 0:
                        wins += 1
            except Exception:
                continue
        total_pnl = sum(tr.pnl for tr in trades) if trades else 0.0
        win_rate = (wins / len(trades)) if trades else 0.0
        ax.set_title(f"{pair_name}  |  z={z_entry:.2f}  |  trades={len(trades)}  |  PnL={total_pnl:.0f}  |  win%={win_rate:.0%}")
        ax.legend(loc='upper right')
    for j in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

def _close_trade(
    pair_name: str,
    direction: int,
    entry_i: int,
    exit_i: int,
    entry_z: float,
    exit_z: float,
    z: pd.Series,
    px_x: pd.Series,
    px_y: pd.Series,
    beta: float,
    commission_per_leg: float,
) -> TradeResult:
    idx = z.index
    e_date = idx[entry_i]
    x_date = idx[exit_i]
    y_e = float(px_y.loc[e_date])
    y_x = float(px_y.loc[x_date])
    x_e = float(px_x.loc[e_date])
    x_x = float(px_x.loc[x_date])
    # Position legs: 1 on Y, beta on X
    if direction == 1:
        # Long spread: +Y, -beta*X
        leg_y = (y_x - y_e)
        leg_x = -beta * (x_x - x_e)
    else:
        # Short spread: -Y, +beta*X
        leg_y = -(y_x - y_e)
        leg_x = beta * (x_x - x_e)
    gross = leg_y + leg_x
    # Commissions: entry and exit, both legs
    commission = 2 * commission_per_leg * (1 + abs(beta))
    pnl = gross - commission
    days_held = (x_date - e_date).days
    return TradeResult(
        pair=pair_name,
        direction=direction,
        entry_date=e_date,
        exit_date=x_date,
        entry_z=entry_z,
        exit_z=exit_z,
        pnl=pnl,
        gross_pnl=gross,
        commission=commission,
        days_held=days_held,
    )


def trades_to_metrics(trades: list) -> dict:
    if not trades:
        return {
            'net_pnl': 0.0,
            'gross_pnl': 0.0,
            'commission': 0.0,
            'num_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'avg_days': 0.0,
            'sharpe': 0.0,
        }
    pnl_series = pd.Series([t.pnl for t in trades])
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    avg_days = np.mean([t.days_held for t in trades]) if trades else 0.0
    # Proxy daily returns from per-trade pnl normalized by |beta|+1 notionally ~ 2 legs; simple ratio
    ret_series = pnl_series
    sharpe = 0.0
    if ret_series.std(ddof=0) > 0:
        sharpe = (ret_series.mean() / ret_series.std(ddof=0)) * np.sqrt(252)
    return {
        'net_pnl': float(pnl_series.sum()),
        'gross_pnl': float(sum(t.gross_pnl for t in trades)),
        'commission': float(sum(t.commission for t in trades)),
        'num_trades': int(len(trades)),
        'win_trades': int(wins.shape[0]),
        'loss_trades': int(losses.shape[0]),
        'avg_win': float(wins.mean()) if not wins.empty else 0.0,
        'avg_loss': float(losses.mean()) if not losses.empty else 0.0,
        'avg_days': float(avg_days),
        'sharpe': float(sharpe),
    }


def grid_search_best_z(
    pair_name: str,
    z: pd.Series,
    prices: pd.DataFrame,
    x_name: str,
    y_name: str,
    alpha: float,
    beta: float,
    z_grid: list,
    stop_extra: float = 0.5,
    commission_per_leg: float = 0.01,
    split_at_middle: bool = False,
):
    """
    Run grid search on z_entry list. If split_at_middle=True, choose best on first half,
    evaluate on second half; otherwise choose on full sample.
    """
    z = z.dropna()
    common_idx = z.index.intersection(prices.index)
    z = z.loc[common_idx]
    prices = prices.loc[common_idx]

    if split_at_middle:
        mid = len(z) // 2
        z_is = z.iloc[:mid]
        z_oos = z.iloc[mid:]
        prices_is = prices.iloc[:mid]
        prices_oos = prices.iloc[mid:]
        best = None
        for ze in z_grid:
            trades = simulate_pair_trades(pair_name, z_is, prices_is, x_name, y_name, alpha, beta, ze, stop_extra, commission_per_leg)
            metrics = trades_to_metrics(trades)
            score = metrics['net_pnl']
            if best is None or score > best['score']:
                best = {'z_entry': ze, 'score': score, 'metrics': metrics}
        # Evaluate on OOS
        trades_oos = simulate_pair_trades(pair_name, z_oos, prices_oos, x_name, y_name, alpha, beta, best['z_entry'], stop_extra, commission_per_leg)
        metrics_oos = trades_to_metrics(trades_oos)
        return best['z_entry'], best['metrics'], metrics_oos
    else:
        best = None
        for ze in z_grid:
            trades = simulate_pair_trades(pair_name, z, prices, x_name, y_name, alpha, beta, ze, stop_extra, commission_per_leg)
            metrics = trades_to_metrics(trades)
            score = metrics['net_pnl']
            if best is None or score > best['score']:
                best = {'z_entry': ze, 'score': score, 'metrics': metrics}
        return best['z_entry'], best['metrics'], None


