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
    position_size: float = 1.0  # Position size multiplier
    risk_amount: float = 0.0    # Risk amount for this trade


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
    initial_capital: float = None,  # If provided, enables risk-based position sizing
    risk_per_trade: float = None,   # Fixed risk amount per trade
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
    position_size = 1.0  # Default position size
    current_capital = initial_capital if initial_capital is not None else None

    for i, date in enumerate(z.index):
        zi = float(z.iloc[i])
        if in_pos == 0:
            # Entry logic: Use z-score for entry decisions
            if zi >= z_entry:
                # Short spread when z-score is above threshold
                # For BB confirmation, we use z-score thresholds instead of spread levels
                ok = True
                if confirm == 'bb':
                    # BB confirmation: z-score must be above threshold (no additional spread check needed)
                    # The z-score already incorporates the spread information
                    ok = True  # z-score threshold is sufficient
                
                if ok:
                    # Calculate position size if risk-based sizing is enabled
                    if current_capital is not None and risk_per_trade is not None:
                        position_size = calculate_position_size(
                            px_x.iloc[i], px_y.iloc[i], beta, zi, z_entry, stop_extra, 
                            risk_per_trade, confirm, spread.iloc[i], upper_stop.iloc[i] if confirm == 'bb' else None
                        )
                    in_pos = -1  # short spread
                    entry_i = i
                    entry_z = zi
                    
            elif zi <= -z_entry:
                # Long spread when z-score is below negative threshold
                # For BB confirmation, we use z-score thresholds instead of spread levels
                ok = True
                if confirm == 'bb':
                    # BB confirmation: z-score must be below threshold (no additional spread check needed)
                    # The z-score already incorporates the spread information
                    ok = True  # z-score threshold is sufficient
                
                if ok:
                    # Calculate position size if risk-based sizing is enabled
                    if current_capital is not None and risk_per_trade is not None:
                        position_size = calculate_position_size(
                            px_x.iloc[i], px_y.iloc[i], beta, zi, z_entry, stop_extra, 
                            risk_per_trade, confirm, spread.iloc[i], lower_stop.iloc[i] if confirm == 'bb' else None
                        )
                    in_pos = 1   # long spread
                    entry_i = i
                    entry_z = zi
        else:
            # stop-loss: 0.5 z-score move in opposite direction
            stop_hit = False
            if confirm == 'bb':
                # For BB confirmation, use both z-score and spread-based stops
                z_stop_hit = False
                bb_stop_hit = False
                
                # Z-score based stop: 0.5 z-score move in opposite direction
                if in_pos == 1:  # Long position
                    # Stop when z-score moves 0.5 in opposite direction (more negative)
                    z_stop_hit = zi <= (entry_z - 0.5)
                elif in_pos == -1:  # Short position  
                    # Stop when z-score moves 0.5 in opposite direction (more positive)
                    z_stop_hit = zi >= (entry_z + 0.5)
                
                # BB-based stop (additional confirmation)
                if in_pos == 1 and (not pd.isna(lower_stop.iloc[i])) and spread.iloc[i] <= lower_stop.iloc[i]:
                    bb_stop_hit = True
                elif in_pos == -1 and (not pd.isna(upper_stop.iloc[i])) and spread.iloc[i] >= upper_stop.iloc[i]:
                    bb_stop_hit = True
                
                # Stop if either z-score or BB stop is hit
                stop_hit = z_stop_hit or bb_stop_hit
            else:
                # For non-BB mode, use only z-score based stops
                if in_pos == 1:  # Long position
                    # Stop when z-score moves 0.5 in opposite direction (more negative)
                    stop_hit = zi <= (entry_z - 0.5)
                elif in_pos == -1:  # Short position
                    # Stop when z-score moves 0.5 in opposite direction (more positive)
                    stop_hit = zi >= (entry_z + 0.5)
            if stop_hit:
                exit_i = i
                trade_result = _close_trade(pair_name, in_pos, entry_i, exit_i, entry_z, zi, z, px_x, px_y, beta, commission_per_leg, position_size, risk_per_trade)
                trades.append(trade_result)
                # Update capital if risk-based sizing is enabled
                if current_capital is not None:
                    current_capital += trade_result.pnl
                in_pos = 0
                entry_i = None
                entry_z = None
                position_size = 1.0
                continue
            # profit exit at z crossing 0
            if (in_pos == 1 and zi >= 0) or (in_pos == -1 and zi <= 0):
                exit_i = i
                trade_result = _close_trade(pair_name, in_pos, entry_i, exit_i, entry_z, zi, z, px_x, px_y, beta, commission_per_leg, position_size, risk_per_trade)
                trades.append(trade_result)
                # Update capital if risk-based sizing is enabled
                if current_capital is not None:
                    current_capital += trade_result.pnl
                in_pos = 0
                entry_i = None
                entry_z = None
                position_size = 1.0

    return trades


def calculate_position_size(px_x, px_y, beta, entry_z, z_entry, stop_extra, risk_amount, confirm, spread, stop_level):
    """
    Calculate position size using simple approach similar to the article.
    
    The logic:
    1. Risk a fixed percentage of capital per trade (e.g., 1%)
    2. Use simple position sizing based on the ratio approach
    3. Position size = risk_amount / (estimated_loss_per_unit * z_stop_distance)
    4. This matches the article's simpler approach
    """
    # Calculate stop distance in z-score terms
    z_stop_distance = 0.5  # Fixed 0.5 z-score move for stop-loss
    
    # Simple approach: estimate loss per unit based on current prices
    # This matches the article's approach more closely
    
    # Calculate the current spread value
    current_spread = spread  # This is Y - (alpha + beta*X)
    
    # Estimate potential loss per unit of position
    # Use a percentage of the spread magnitude as a rough estimate
    spread_magnitude = abs(current_spread)
    if spread_magnitude == 0:
        spread_magnitude = px_y * 0.01  # Fallback: 1% of Y price
    
    # Estimate potential loss per unit (conservative estimate)
    # This represents how much the spread can move against us
    estimated_loss_per_unit = spread_magnitude * 0.02  # 2% of spread magnitude
    
    # Calculate position size using simple risk approach
    # We want: position_size * estimated_loss_per_unit * z_stop_distance = risk_amount
    # So: position_size = risk_amount / (estimated_loss_per_unit * z_stop_distance)
    
    if estimated_loss_per_unit <= 0 or z_stop_distance <= 0:
        return 1.0  # Default position size if calculation fails
    
    position_size = risk_amount / (estimated_loss_per_unit * z_stop_distance)
    
    # Cap position size to prevent excessive leverage
    max_position_size = 10.0  # Maximum 10x leverage
    position_size = min(position_size, max_position_size)
    
    # Ensure minimum position size for meaningful trades
    min_position_size = 0.1
    position_size = max(position_size, min_position_size)
    
    return position_size


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
    plt.figure(figsize=(16, 10))
    
    # Plot z-score
    plt.plot(z.index, z.values, label='Z-Score', color='tab:blue', linewidth=1.5, alpha=0.7)
    
    # Plot threshold lines
    plt.axhline(z_entry, color='tab:red', linestyle='--', linewidth=2, label=f'+{z_entry:.2f} (Short Entry)', alpha=0.8)
    plt.axhline(-z_entry, color='tab:green', linestyle='--', linewidth=2, label=f'-{z_entry:.2f} (Long Entry)', alpha=0.8)
    plt.axhline(0.0, color='black', linestyle='-', linewidth=1.5, label='Mean (Exit)', alpha=0.8)
    
    # Plot stop-loss levels
    plt.axhline(z_entry + 0.5, color='red', linestyle=':', linewidth=1, label=f'+{z_entry+0.5:.2f} (Short Stop)', alpha=0.6)
    plt.axhline(-z_entry - 0.5, color='green', linestyle=':', linewidth=1, label=f'-{z_entry+0.5:.2f} (Long Stop)', alpha=0.6)
    
    # Mark entries/exits with better visualization
    for t in trades:
        try:
            e = t.entry_date
            x = t.exit_date
            ze = float(z.loc[e]) if e in z.index else None
            zx = float(z.loc[x]) if x in z.index else None
            
            if ze is not None:
                # Entry markers with direction and PnL info
                color = 'red' if t.direction == -1 else 'green'
                marker = 'v' if t.direction == -1 else '^'  # Down for short, up for long
                plt.scatter(e, ze, color=color, marker=marker, s=100, alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
                # Add entry annotation
                plt.annotate(f'{"SHORT" if t.direction == -1 else "LONG"}', 
                           (e, ze), xytext=(0, 20), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
                
            if zx is not None:
                # Exit markers with PnL coloring
                exit_color = 'darkgreen' if t.pnl > 0 else 'darkred'
                plt.scatter(x, zx, color=exit_color, marker='X', s=80, alpha=0.9, edgecolors='black', linewidth=2, zorder=5)
                # Add PnL annotation
                plt.annotate(f'{t.pnl:.0f}', 
                           (x, zx), xytext=(0, -25), textcoords='offset points',
                           ha='center', va='top', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=exit_color, alpha=0.7))
                
        except Exception:
            continue
    
    # Add trade statistics to title
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    plt.title(f'{pair_name} — Z-Score Trading Signals\n'
              f'Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Total PnL: {total_pnl:.0f} | Avg PnL: {avg_pnl:.0f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Z-Score', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    Save a plot of the spread price series with entry/exit markers and Bollinger Bands.
    """
    common_idx = prices.index
    spread = prices[y_name].loc[common_idx] - (alpha + beta * prices[x_name].loc[common_idx])
    
    plt.figure(figsize=(16, 10))
    
    # Calculate Bollinger Bands for visualization
    bb_window = 20
    bb_num_std = 1.0
    mid = spread.rolling(bb_window, min_periods=bb_window).mean()
    vol = spread.rolling(bb_window, min_periods=bb_window).std(ddof=0)
    upper = mid + bb_num_std * vol
    lower = mid - bb_num_std * vol
    
    # Plot spread and Bollinger Bands
    plt.plot(spread.index, spread.values, label='Spread (Y - (α + β·X))', color='tab:purple', linewidth=1.5, alpha=0.8)
    plt.plot(mid.index, mid.values, label='BB Middle (20-day MA)', color='orange', linewidth=1.5, alpha=0.8)
    plt.fill_between(upper.index, upper.values, lower.values, alpha=0.2, color='gray', label='BB Bands (±1σ)')
    plt.plot(upper.index, upper.values, color='red', linewidth=1.5, alpha=0.8, linestyle='--')
    plt.plot(lower.index, lower.values, color='green', linewidth=1.5, alpha=0.8, linestyle='--')
    
    # Mark entries/exits with better visualization
    for t in trades:
        try:
            e = t.entry_date
            x = t.exit_date
            if e in spread.index:
                # Entry markers
                color = 'red' if t.direction == -1 else 'green'
                marker = 'v' if t.direction == -1 else '^'
                plt.scatter(e, float(spread.loc[e]), color=color, marker=marker, s=100, alpha=0.9, 
                           edgecolors='black', linewidth=2, zorder=5)
                # Add entry annotation
                plt.annotate(f'{"SHORT" if t.direction == -1 else "LONG"}', 
                           (e, float(spread.loc[e])), xytext=(0, 20), textcoords='offset points',
                           ha='center', va='bottom', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
                
            if x in spread.index:
                # Exit markers
                exit_color = 'darkgreen' if t.pnl > 0 else 'darkred'
                plt.scatter(x, float(spread.loc[x]), color=exit_color, marker='X', s=80, alpha=0.9,
                           edgecolors='black', linewidth=2, zorder=5)
                # Add PnL annotation
                plt.annotate(f'{t.pnl:.0f}', 
                           (x, float(spread.loc[x])), xytext=(0, -25), textcoords='offset points',
                           ha='center', va='top', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=exit_color, alpha=0.7))
        except Exception:
            continue
    
    # Add trade statistics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t.pnl > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t.pnl for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    
    plt.title(f'{pair_name} — Spread with Bollinger Bands and Trading Signals\n'
              f'Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Total PnL: {total_pnl:.0f} | Avg PnL: {avg_pnl:.0f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Spread Value', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_equity_curve(
    pair_name: str,
    trades: list,
    initial_capital: float,
    output_path: str,
):
    """
    Plot the equity curve showing cumulative P&L over time.
    """
    if not trades:
        return
        
    plt.figure(figsize=(16, 10))
    
    # Sort trades by entry date
    sorted_trades = sorted(trades, key=lambda x: x.entry_date)
    
    # Calculate cumulative P&L
    dates = []
    cumulative_pnl = []
    running_capital = initial_capital
    
    for trade in sorted_trades:
        dates.append(trade.exit_date)
        running_capital += trade.pnl
        cumulative_pnl.append(running_capital)
    
    # Convert to percentage returns
    pct_returns = [(cap - initial_capital) / initial_capital * 100 for cap in cumulative_pnl]
    
    # Plot equity curve
    plt.plot(dates, cumulative_pnl, label='Cumulative Capital', color='tab:blue', linewidth=2)
    plt.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7, label=f'Initial Capital (€{initial_capital:,.0f})')
    
    # Fill area below/above initial capital
    plt.fill_between(dates, initial_capital, cumulative_pnl, 
                    where=[pnl >= initial_capital for pnl in cumulative_pnl], 
                    color='green', alpha=0.3, label='Profit Zone')
    plt.fill_between(dates, initial_capital, cumulative_pnl, 
                    where=[pnl < initial_capital for pnl in cumulative_pnl], 
                    color='red', alpha=0.3, label='Loss Zone')
    
    # Add trade markers
    for i, trade in enumerate(sorted_trades):
        color = 'green' if trade.pnl > 0 else 'red'
        plt.scatter(trade.exit_date, cumulative_pnl[i], color=color, s=50, alpha=0.7, zorder=5)
    
    # Calculate statistics
    final_capital = cumulative_pnl[-1] if cumulative_pnl else initial_capital
    total_return = (final_capital - initial_capital) / initial_capital * 100
    max_capital = max(cumulative_pnl) if cumulative_pnl else initial_capital
    max_drawdown = (max_capital - min(cumulative_pnl)) / max_capital * 100 if cumulative_pnl else 0
    
    # Add statistics to title
    plt.title(f'{pair_name} — Equity Curve\n'
              f'Initial Capital: €{initial_capital:,.0f} | Final Capital: €{final_capital:,.0f} | '
              f'Total Return: {total_return:.1f}% | Max Drawdown: {max_drawdown:.1f}%', 
              fontsize=14, fontweight='bold')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Capital (€)', fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    position_size: float = 1.0,
    risk_amount: float = None,
) -> TradeResult:
    idx = z.index
    e_date = idx[entry_i]
    x_date = idx[exit_i]
    y_e = float(px_y.loc[e_date])
    y_x = float(px_y.loc[x_date])
    x_e = float(px_x.loc[e_date])
    x_x = float(px_x.loc[x_date])
    
    # Position sizing represents the capital allocated to the trade
    # We trade position_size units of Y and position_size * beta units of X
    
    if direction == 1:
        # Long spread: Long Y, Short beta*X (when z < -threshold)
        # Buy Y, Sell beta*X
        leg_y = position_size * (y_x - y_e)  # Long Y: profit when Y goes up
        leg_x = -position_size * beta * (x_x - x_e)  # Short beta*X: profit when X goes down
    else:
        # Short spread: Short Y, Long beta*X (when z > threshold)  
        # Sell Y, Buy beta*X
        leg_y = -position_size * (y_x - y_e)  # Short Y: profit when Y goes down
        leg_x = position_size * beta * (x_x - x_e)  # Long beta*X: profit when X goes up
    
    gross = leg_y + leg_x
    
    # Commissions: entry and exit, both legs
    # Commission is based on the dollar amount traded, not position size
    y_trade_amount = position_size * y_e
    x_trade_amount = position_size * beta * x_e
    total_trade_amount = y_trade_amount + x_trade_amount
    commission = 2 * commission_per_leg * total_trade_amount  # Entry and exit
    
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
        risk_amount=risk_amount if risk_amount is not None else 0.0,
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
            'max_loss': 0.0,
            'avg_position_size': 1.0,
            'total_risk': 0.0,
        }
    pnl_series = pd.Series([t.pnl for t in trades])
    wins = pnl_series[pnl_series > 0]
    losses = pnl_series[pnl_series <= 0]
    avg_days = np.mean([t.days_held for t in trades]) if trades else 0.0
    avg_position_size = np.mean([t.position_size for t in trades]) if trades else 1.0
    total_risk = sum([t.risk_amount for t in trades]) if trades else 0.0
    max_loss = min([t.pnl for t in trades]) if trades else 0.0
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
        'max_loss': float(max_loss),
        'avg_position_size': float(avg_position_size),
        'total_risk': float(total_risk),
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
    initial_capital: float = None,
    risk_per_trade: float = None,
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
            trades = simulate_pair_trades(pair_name, z_is, prices_is, x_name, y_name, alpha, beta, ze, stop_extra, commission_per_leg, initial_capital=initial_capital, risk_per_trade=risk_per_trade)
            metrics = trades_to_metrics(trades)
            score = metrics['net_pnl']
            if best is None or score > best['score']:
                best = {'z_entry': ze, 'score': score, 'metrics': metrics}
        # Evaluate on OOS
        trades_oos = simulate_pair_trades(pair_name, z_oos, prices_oos, x_name, y_name, alpha, beta, best['z_entry'], stop_extra, commission_per_leg, initial_capital=initial_capital, risk_per_trade=risk_per_trade)
        metrics_oos = trades_to_metrics(trades_oos)
        return best['z_entry'], best['metrics'], metrics_oos
    else:
        best = None
        for ze in z_grid:
            trades = simulate_pair_trades(pair_name, z, prices, x_name, y_name, alpha, beta, ze, stop_extra, commission_per_leg, initial_capital=initial_capital, risk_per_trade=risk_per_trade)
            metrics = trades_to_metrics(trades)
            score = metrics['net_pnl']
            if best is None or score > best['score']:
                best = {'z_entry': ze, 'score': score, 'metrics': metrics}
        return best['z_entry'], best['metrics'], None


