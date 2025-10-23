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
    position_size: float  # New field for position size
    risk_amount: float    # New field for risk amount


def simulate_pair_trades_risk_sized(
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
    initial_capital: float = 10000.0,  # Initial capital amount in euros
    risk_per_trade: float = 100.0,  # Fixed risk amount per trade in euros
    confirm: str = 'none',  # 'none' | 'bb'
    bb_window: int = 20,
    bb_num_std: float = 1.0,
) -> list:
    """
    Simulate trades with risk-based position sizing.
    Position size is calculated so that maximum loss equals fixed risk_per_trade amount.
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

    trades: list[TradeResult] = []
    in_pos = 0
    entry_i = None
    entry_z = None
    position_size = 0.0

    for i, date in enumerate(z.index):
        zi = float(z.iloc[i])
        if in_pos == 0:
            if zi >= z_entry:
                # require confirmations if set
                ok = True
                if confirm == 'bb':
                    ok = (not pd.isna(upper.iloc[i])) and (spread.iloc[i] >= upper.iloc[i])
                if ok:
                    # Calculate position size based on fixed risk amount
                    position_size = calculate_position_size(
                        px_x.iloc[i], px_y.iloc[i], beta, zi, z_entry, stop_extra, 
                        risk_per_trade, confirm, spread.iloc[i], upper_stop.iloc[i] if confirm == 'bb' else None
                    )
                    in_pos = -1  # short spread
                    entry_i = i
                    entry_z = zi
            elif zi <= -z_entry:
                ok = True
                if confirm == 'bb':
                    ok = (not pd.isna(lower.iloc[i])) and (spread.iloc[i] <= lower.iloc[i])
                if ok:
                    # Calculate position size based on fixed risk amount
                    position_size = calculate_position_size(
                        px_x.iloc[i], px_y.iloc[i], beta, zi, z_entry, stop_extra, 
                        risk_per_trade, confirm, spread.iloc[i], lower_stop.iloc[i] if confirm == 'bb' else None
                    )
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
                trade_result = _close_trade_risk_sized(
                    pair_name, in_pos, entry_i, exit_i, entry_z, zi, z, px_x, px_y, 
                    beta, commission_per_leg, position_size, risk_per_trade
                )
                trades.append(trade_result)
                in_pos = 0
                entry_i = None
                entry_z = None
                position_size = 0.0
                continue
            # profit exit at z crossing 0
            if (in_pos == 1 and zi >= 0) or (in_pos == -1 and zi <= 0):
                exit_i = i
                trade_result = _close_trade_risk_sized(
                    pair_name, in_pos, entry_i, exit_i, entry_z, zi, z, px_x, px_y, 
                    beta, commission_per_leg, position_size, risk_per_trade
                )
                trades.append(trade_result)
                in_pos = 0
                entry_i = None
                entry_z = None
                position_size = 0.0

    return trades


def calculate_position_size(px_x, px_y, beta, entry_z, z_entry, stop_extra, risk_amount, confirm, spread, stop_level):
    """
    Calculate position size so that maximum loss equals fixed risk_amount (e.g., €100).
    
    For a spread trade: PnL = position_size * (spread_exit - spread_entry)
    We want: max_loss = position_size * max_spread_move = risk_amount
    
    So: position_size = risk_amount / max_spread_move
    """
    if confirm == 'bb':
        # BB-based stop: calculate spread move to stop level
        if entry_z > 0:  # Short spread
            max_spread_move = abs(stop_level - spread) if stop_level is not None else 0
        else:  # Long spread
            max_spread_move = abs(spread - stop_level) if stop_level is not None else 0
    else:
        # Z-score based stop: estimate spread move from z-score move
        z_stop = abs(entry_z) + stop_extra
        z_move = z_stop - abs(entry_z)
        # Estimate spread volatility from current prices (simplified)
        spread_vol = (px_y + abs(beta) * px_x) * 0.01  # Rough estimate: 1% of combined value
        max_spread_move = z_move * spread_vol
    
    if max_spread_move <= 0:
        return 0.0
    
    position_size = risk_amount / max_spread_move
    return position_size


def _close_trade_risk_sized(
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
    position_size: float,
    risk_amount: float,
) -> TradeResult:
    idx = z.index
    e_date = idx[entry_i]
    x_date = idx[exit_i]
    y_e = float(px_y.loc[e_date])
    y_x = float(px_y.loc[x_date])
    x_e = float(px_x.loc[e_date])
    x_x = float(px_x.loc[x_date])
    
    # Position legs: position_size on Y, position_size*beta on X
    if direction == 1:
        # Long spread: +Y, -beta*X
        leg_y = position_size * (y_x - y_e)
        leg_x = -position_size * beta * (x_x - x_e)
    else:
        # Short spread: -Y, +beta*X
        leg_y = -position_size * (y_x - y_e)
        leg_x = position_size * beta * (x_x - x_e)
    
    gross = leg_y + leg_x
    # Commissions: entry and exit, both legs, scaled by position size
    commission = 2 * commission_per_leg * position_size * (1 + abs(beta))
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
        position_size=position_size,
        risk_amount=risk_amount,
    )


def trades_to_metrics_risk_sized(trades: list) -> dict:
    """Enhanced metrics for risk-sized trades"""
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
            'max_loss': 0.0,
            'avg_position_size': 0.0,
            'total_risk': 0.0,
        }
    
    pnl_series = pd.Series([t.pnl for t in trades])
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    avg_days = np.mean([t.days_held for t in trades]) if trades else 0.0
    avg_position_size = np.mean([t.position_size for t in trades]) if trades else 0.0
    total_risk = sum([t.risk_amount for t in trades]) if trades else 0.0
    max_loss = min([t.pnl for t in trades]) if trades else 0.0
    
    # Sharpe ratio
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
        'max_loss': float(max_loss),
        'avg_position_size': float(avg_position_size),
        'total_risk': float(total_risk),
    }


# You would also need to modify the grid_search_best_z function to use the risk-sized version
def grid_search_best_z_risk_sized(
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
    initial_capital: float = 10000.0,
    risk_per_trade: float = 100.0,
    split_at_middle: bool = False,
):
    """
    Grid search with risk-based position sizing.
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
            trades = simulate_pair_trades_risk_sized(
                pair_name, z_is, prices_is, x_name, y_name, alpha, beta, ze, 
                stop_extra, commission_per_leg, initial_capital, risk_per_trade
            )
            metrics = trades_to_metrics_risk_sized(trades)
            score = metrics['net_pnl']
            if best is None or score > best['score']:
                best = {'z_entry': ze, 'score': score, 'metrics': metrics}
        # Evaluate on OOS
        trades_oos = simulate_pair_trades_risk_sized(
            pair_name, z_oos, prices_oos, x_name, y_name, alpha, beta, best['z_entry'], 
            stop_extra, commission_per_leg, initial_capital, risk_per_trade
        )
        metrics_oos = trades_to_metrics_risk_sized(trades_oos)
        return best['z_entry'], best['metrics'], metrics_oos
    else:
        best = None
        for ze in z_grid:
            trades = simulate_pair_trades_risk_sized(
                pair_name, z, prices, x_name, y_name, alpha, beta, ze, 
                stop_extra, commission_per_leg, initial_capital, risk_per_trade
            )
            metrics = trades_to_metrics_risk_sized(trades)
            score = metrics['net_pnl']
            if best is None or score > best['score']:
                best = {'z_entry': ze, 'score': score, 'metrics': metrics}
        return best['z_entry'], best['metrics'], None


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
