"""
Visualization functions for earnings predictions.

Generates various plots to visualize expected moves, direction signals,
historical context, and options analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from datetime import datetime
import os

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')


def plot_expected_move_range(
    symbol: str,
    current_price: float,
    expected_move_percent: float,
    expected_range: Tuple[float, float],
    historical_moves: Optional[List[float]] = None,
    predicted_direction: str = 'neutral',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot the expected move range with current price and bounds.

    Shows:
    - Current price marker
    - Upper and lower expected bounds
    - Direction indicator arrow
    - Historical moves if available

    Args:
        symbol: Stock symbol
        current_price: Current stock price
        expected_move_percent: Expected move percentage
        expected_range: Tuple of (lower_bound, upper_bound)
        historical_moves: List of historical move percentages
        predicted_direction: 'up', 'down', or 'neutral'
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    lower, upper = expected_range

    # Calculate price range for x-axis
    range_width = upper - lower
    x_min = lower - range_width * 0.2
    x_max = upper + range_width * 0.2

    # Draw the expected range band
    ax.axvspan(lower, upper, alpha=0.3, color='#3498db', label='Expected Range')

    # Draw vertical lines for bounds
    ax.axvline(x=lower, color='#e74c3c', linestyle='--', linewidth=2, label=f'Lower: ${lower:.2f}')
    ax.axvline(x=upper, color='#27ae60', linestyle='--', linewidth=2, label=f'Upper: ${upper:.2f}')

    # Draw current price line
    ax.axvline(x=current_price, color='#2c3e50', linestyle='-', linewidth=3, label=f'Current: ${current_price:.2f}')

    # Add direction arrow
    arrow_y = 0.7
    arrow_color = {'up': '#27ae60', 'down': '#e74c3c', 'neutral': '#95a5a6'}[predicted_direction]

    if predicted_direction == 'up':
        ax.annotate('', xy=(upper * 0.98, arrow_y), xytext=(current_price, arrow_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))
        ax.text(current_price + (upper - current_price) / 2, arrow_y + 0.1,
               'PREDICTED UP', ha='center', fontsize=12, fontweight='bold', color=arrow_color)
    elif predicted_direction == 'down':
        ax.annotate('', xy=(lower * 1.02, arrow_y), xytext=(current_price, arrow_y),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))
        ax.text(current_price - (current_price - lower) / 2, arrow_y + 0.1,
               'PREDICTED DOWN', ha='center', fontsize=12, fontweight='bold', color=arrow_color)
    else:
        ax.text(current_price, arrow_y + 0.1, 'NEUTRAL', ha='center',
               fontsize=12, fontweight='bold', color=arrow_color)

    # Plot historical moves if available
    if historical_moves and len(historical_moves) > 0:
        hist_prices = [current_price * (1 + m/100) for m in historical_moves[:5]]
        for i, hp in enumerate(hist_prices):
            if x_min <= hp <= x_max:
                ax.scatter(hp, 0.3 - i * 0.05, color='#9b59b6', s=100, alpha=0.6, zorder=5)
                ax.annotate(f'{historical_moves[i]:+.1f}%', (hp, 0.3 - i * 0.05),
                          textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    # Styling
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Stock Price ($)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'{symbol} Expected Earnings Move: ±{expected_move_percent:.2f}%',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)

    # Add move percentage annotation
    ax.text(0.02, 0.95, f'Expected Move: ±{expected_move_percent:.2f}%\n'
            f'Range: ${lower:.2f} - ${upper:.2f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_direction_signals(
    symbol: str,
    direction_signals: Dict[str, Any],
    direction_confidence: float,
    predicted_direction: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot the direction prediction signals breakdown.

    Shows each signal's contribution to the direction prediction
    as a horizontal bar chart with color-coded sentiment.

    Args:
        symbol: Stock symbol
        direction_signals: Dict of signal names to their values/signals
        direction_confidence: Overall direction confidence (0-1)
        predicted_direction: 'up', 'down', or 'neutral'
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    signal_names = []
    signal_values = []
    signal_colors = []

    color_map = {'bullish': '#27ae60', 'bearish': '#e74c3c', 'neutral': '#95a5a6'}

    for name, data in direction_signals.items():
        signal_names.append(name.replace('_', ' ').title())

        if isinstance(data, dict):
            signal = data.get('signal', 'neutral')
            value = data.get('value', 0)
            if isinstance(value, (int, float)):
                signal_values.append(value)
            else:
                signal_values.append(1 if signal == 'bullish' else -1 if signal == 'bearish' else 0)
            signal_colors.append(color_map.get(signal, '#95a5a6'))
        else:
            signal_values.append(0)
            signal_colors.append('#95a5a6')

    if not signal_names:
        signal_names = ['No Signals']
        signal_values = [0]
        signal_colors = ['#95a5a6']

    # Normalize values for display
    max_abs_val = max(abs(v) for v in signal_values) if signal_values else 1
    if max_abs_val > 0:
        normalized_values = [v / max_abs_val for v in signal_values]
    else:
        normalized_values = signal_values

    y_pos = np.arange(len(signal_names))

    # Create horizontal bars
    bars = ax.barh(y_pos, normalized_values, color=signal_colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, signal_values)):
        width = bar.get_width()
        label_x = width + 0.05 if width >= 0 else width - 0.05
        ha = 'left' if width >= 0 else 'right'

        if isinstance(val, float):
            label = f'{val:.2f}'
        else:
            label = str(val)
        ax.text(label_x, bar.get_y() + bar.get_height()/2, label,
               ha=ha, va='center', fontsize=10)

    # Add center line
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(signal_names, fontsize=11)
    ax.set_xlabel('Signal Strength', fontsize=12)
    ax.set_xlim(-1.5, 1.5)

    # Direction indicator
    direction_color = {'up': '#27ae60', 'down': '#e74c3c', 'neutral': '#95a5a6'}[predicted_direction]
    ax.set_title(f'{symbol} Direction Signals\nPredicted: {predicted_direction.upper()} ({direction_confidence*100:.1f}% confidence)',
                fontsize=14, fontweight='bold', pad=20, color=direction_color)

    # Legend
    bullish_patch = mpatches.Patch(color='#27ae60', label='Bullish')
    bearish_patch = mpatches.Patch(color='#e74c3c', label='Bearish')
    neutral_patch = mpatches.Patch(color='#95a5a6', label='Neutral')
    ax.legend(handles=[bullish_patch, bearish_patch, neutral_patch],
             loc='lower right', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_historical_moves(
    symbol: str,
    historical_moves: List[float],
    expected_move_percent: float,
    avg_historical_move: float,
    beat_rate: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot historical earnings moves comparison.

    Shows bar chart of past earnings moves compared to current expected move.

    Args:
        symbol: Stock symbol
        historical_moves: List of historical move percentages
        expected_move_percent: Current expected move percentage
        avg_historical_move: Average absolute historical move
        beat_rate: Historical beat rate (0-1)
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not historical_moves:
        ax.text(0.5, 0.5, 'No historical data available',
               ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        return fig

    # Prepare data
    moves = historical_moves[:12]  # Last 12 quarters
    x = np.arange(len(moves))
    colors = ['#27ae60' if m >= 0 else '#e74c3c' for m in moves]

    # Create bars
    bars = ax.bar(x, moves, color=colors, alpha=0.8, edgecolor='black', width=0.7)

    # Add expected move range
    ax.axhline(y=expected_move_percent, color='#3498db', linestyle='--',
              linewidth=2, label=f'Expected: +{expected_move_percent:.1f}%')
    ax.axhline(y=-expected_move_percent, color='#3498db', linestyle='--', linewidth=2)
    ax.fill_between([-0.5, len(moves) - 0.5], -expected_move_percent, expected_move_percent,
                   alpha=0.1, color='#3498db', label='Expected Range')

    # Add average line
    ax.axhline(y=avg_historical_move, color='#9b59b6', linestyle=':',
              linewidth=2, label=f'Avg Move: ±{avg_historical_move:.1f}%')
    ax.axhline(y=-avg_historical_move, color='#9b59b6', linestyle=':', linewidth=2)

    # Add value labels
    for bar, val in zip(bars, moves):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + (0.3 if height >= 0 else -0.5),
               f'{val:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
               fontsize=9, fontweight='bold')

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i+1}' for i in range(len(moves))], fontsize=10)
    ax.set_xlabel('Past Earnings (Most Recent First)', fontsize=12)
    ax.set_ylabel('Move %', fontsize=12)
    ax.set_title(f'{symbol} Historical Earnings Moves\nBeat Rate: {beat_rate*100:.0f}%',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add summary box
    textstr = f'Avg Historical: ±{avg_historical_move:.1f}%\nExpected Move: ±{expected_move_percent:.1f}%\nBeat Rate: {beat_rate*100:.0f}%'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_options_analysis(
    symbol: str,
    current_price: float,
    atm_strike: float,
    put_call_ratio: float,
    max_pain_strike: float,
    iv_percentile: float,
    atm_iv: float,
    flow_sentiment: str = 'neutral',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot options analysis metrics dashboard.

    Shows key options metrics in a visual dashboard format.

    Args:
        symbol: Stock symbol
        current_price: Current stock price
        atm_strike: At-the-money strike price
        put_call_ratio: Put/Call ratio
        max_pain_strike: Max pain strike price
        iv_percentile: IV percentile (0-1)
        atm_iv: At-the-money implied volatility (0-1)
        flow_sentiment: 'bullish', 'bearish', or 'neutral'
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Create grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Price vs Strike visualization (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot price levels
    prices = [current_price, atm_strike, max_pain_strike]
    labels = ['Current', 'ATM Strike', 'Max Pain']
    colors = ['#2c3e50', '#3498db', '#e74c3c']
    y_pos = np.arange(len(prices))

    ax1.barh(y_pos, prices, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel('Price ($)', fontsize=10)
    ax1.set_title('Price Levels', fontsize=12, fontweight='bold')

    for i, (p, c) in enumerate(zip(prices, colors)):
        ax1.text(p + 1, i, f'${p:.2f}', va='center', fontsize=9)

    # 2. Put/Call Ratio gauge (top middle)
    ax2 = fig.add_subplot(gs[0, 1])

    # Create gauge-like visualization
    pcr_normalized = min(put_call_ratio / 2, 1)  # Normalize to 0-1 (2.0 = max)

    if put_call_ratio < 0.7:
        pcr_color = '#27ae60'
        pcr_label = 'Bullish'
    elif put_call_ratio > 1.3:
        pcr_color = '#e74c3c'
        pcr_label = 'Bearish'
    else:
        pcr_color = '#f39c12'
        pcr_label = 'Neutral'

    wedge = Wedge((0.5, 0), 0.4, 0, 180, width=0.15, facecolor='#ecf0f1', edgecolor='black')
    ax2.add_patch(wedge)

    angle = 180 - (pcr_normalized * 180)
    wedge_fill = Wedge((0.5, 0), 0.4, angle, 180, width=0.15, facecolor=pcr_color, edgecolor='black')
    ax2.add_patch(wedge_fill)

    ax2.text(0.5, 0.15, f'{put_call_ratio:.2f}', ha='center', va='center',
            fontsize=20, fontweight='bold', color=pcr_color)
    ax2.text(0.5, -0.05, pcr_label, ha='center', va='center', fontsize=11, color=pcr_color)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.15, 0.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Put/Call Ratio', fontsize=12, fontweight='bold', pad=10)

    # 3. IV Percentile gauge (top right)
    ax3 = fig.add_subplot(gs[0, 2])

    if iv_percentile > 0.7:
        iv_color = '#e74c3c'
        iv_label = 'High IV'
    elif iv_percentile < 0.3:
        iv_color = '#27ae60'
        iv_label = 'Low IV'
    else:
        iv_color = '#f39c12'
        iv_label = 'Normal IV'

    wedge_bg = Wedge((0.5, 0), 0.4, 0, 180, width=0.15, facecolor='#ecf0f1', edgecolor='black')
    ax3.add_patch(wedge_bg)

    angle = 180 - (iv_percentile * 180)
    wedge_fill = Wedge((0.5, 0), 0.4, angle, 180, width=0.15, facecolor=iv_color, edgecolor='black')
    ax3.add_patch(wedge_fill)

    ax3.text(0.5, 0.15, f'{iv_percentile*100:.0f}%', ha='center', va='center',
            fontsize=20, fontweight='bold', color=iv_color)
    ax3.text(0.5, -0.05, iv_label, ha='center', va='center', fontsize=11, color=iv_color)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-0.15, 0.5)
    ax3.set_aspect('equal')
    ax3.axis('off')
    ax3.set_title('IV Percentile', fontsize=12, fontweight='bold', pad=10)

    # 4. Flow sentiment indicator (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])

    sentiment_colors = {'bullish': '#27ae60', 'bearish': '#e74c3c', 'neutral': '#95a5a6'}
    sentiment_icons = {'bullish': '^', 'bearish': 'v', 'neutral': 'o'}

    ax4.scatter(0.5, 0.5, s=5000, c=sentiment_colors.get(flow_sentiment, '#95a5a6'),
               marker=sentiment_icons.get(flow_sentiment, 'o'), alpha=0.8)
    ax4.text(0.5, 0.1, flow_sentiment.upper(), ha='center', va='center',
            fontsize=14, fontweight='bold', color=sentiment_colors.get(flow_sentiment, '#95a5a6'))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Flow Sentiment', fontsize=12, fontweight='bold')

    # 5. ATM IV display (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])

    ax5.text(0.5, 0.6, f'{atm_iv*100:.1f}%', ha='center', va='center',
            fontsize=28, fontweight='bold', color='#2c3e50')
    ax5.text(0.5, 0.3, 'ATM Implied Volatility', ha='center', va='center',
            fontsize=11, color='#7f8c8d')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('ATM IV', fontsize=12, fontweight='bold')

    # 6. Summary stats (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])

    summary_text = (
        f"Current Price: ${current_price:.2f}\n"
        f"ATM Strike: ${atm_strike:.2f}\n"
        f"Max Pain: ${max_pain_strike:.2f}\n"
        f"Put/Call Ratio: {put_call_ratio:.2f}\n"
        f"IV Percentile: {iv_percentile*100:.0f}%\n"
        f"ATM IV: {atm_iv*100:.1f}%"
    )
    ax6.text(0.5, 0.5, summary_text, ha='center', va='center',
            fontsize=10, family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Summary', fontsize=12, fontweight='bold')

    fig.suptitle(f'{symbol} Options Analysis', fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_probability_cone(
    symbol: str,
    current_price: float,
    expected_move_percent: float,
    confidence_levels: List[float] = [0.5, 0.68, 0.95],
    days_to_earnings: int = 1,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7)
) -> plt.Figure:
    """
    Plot probability cone showing price ranges at different confidence levels.

    Args:
        symbol: Stock symbol
        current_price: Current stock price
        expected_move_percent: Expected move percentage
        confidence_levels: List of confidence levels (e.g., [0.5, 0.68, 0.95])
        days_to_earnings: Days until earnings
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate move multipliers for each confidence level
    # Using normal distribution z-scores
    z_scores = {
        0.50: 0.674,
        0.68: 1.0,
        0.80: 1.28,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }

    colors = ['#27ae60', '#3498db', '#e74c3c', '#9b59b6']

    x = np.array([0, 1])  # Before and after earnings

    # Sort confidence levels for proper layering
    sorted_levels = sorted(confidence_levels, reverse=True)

    for i, conf in enumerate(sorted_levels):
        z = z_scores.get(conf, 1.0)

        # Scale the expected move by z-score
        move = expected_move_percent * z / z_scores[0.68]  # Normalize relative to 68%

        upper = current_price * (1 + move / 100)
        lower = current_price * (1 - move / 100)

        # Create cone
        y_upper = [current_price, upper]
        y_lower = [current_price, lower]

        ax.fill_between(x, y_lower, y_upper, alpha=0.3, color=colors[i % len(colors)],
                       label=f'{conf*100:.0f}% Confidence: ${lower:.2f} - ${upper:.2f}')
        ax.plot(x, y_upper, color=colors[i % len(colors)], linewidth=2)
        ax.plot(x, y_lower, color=colors[i % len(colors)], linewidth=2)

    # Current price line
    ax.axhline(y=current_price, color='#2c3e50', linestyle='-', linewidth=2)
    ax.plot(0, current_price, 'ko', markersize=12, label=f'Current: ${current_price:.2f}')

    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Now', 'After Earnings'], fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.set_title(f'{symbol} Probability Cone\nExpected Move: ±{expected_move_percent:.2f}%',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(-0.1, 1.1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_confidence_gauge(
    symbol: str,
    overall_confidence: float,
    direction_confidence: float,
    predicted_direction: str,
    data_sources: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot overall confidence gauge with breakdown.

    Args:
        symbol: Stock symbol
        overall_confidence: Overall prediction confidence (0-1)
        direction_confidence: Direction prediction confidence (0-1)
        predicted_direction: 'up', 'down', or 'neutral'
        data_sources: List of data sources used
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Overall confidence gauge
    ax1 = axes[0]

    if overall_confidence > 0.7:
        conf_color = '#27ae60'
        conf_label = 'High'
    elif overall_confidence > 0.5:
        conf_color = '#f39c12'
        conf_label = 'Moderate'
    else:
        conf_color = '#e74c3c'
        conf_label = 'Low'

    # Draw gauge background
    wedge_bg = Wedge((0.5, 0.2), 0.35, 0, 180, width=0.12, facecolor='#ecf0f1', edgecolor='black')
    ax1.add_patch(wedge_bg)

    # Draw filled portion
    angle = 180 - (overall_confidence * 180)
    wedge_fill = Wedge((0.5, 0.2), 0.35, angle, 180, width=0.12, facecolor=conf_color, edgecolor='black')
    ax1.add_patch(wedge_fill)

    ax1.text(0.5, 0.3, f'{overall_confidence*100:.0f}%', ha='center', va='center',
            fontsize=24, fontweight='bold', color=conf_color)
    ax1.text(0.5, 0.08, f'{conf_label} Confidence', ha='center', va='center',
            fontsize=12, color=conf_color)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.8)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Overall Confidence', fontsize=14, fontweight='bold', pad=10)

    # Direction confidence with direction indicator
    ax2 = axes[1]

    direction_color = {'up': '#27ae60', 'down': '#e74c3c', 'neutral': '#95a5a6'}[predicted_direction]
    direction_symbol = {'up': '^', 'down': 'v', 'neutral': 'o'}[predicted_direction]

    # Direction indicator
    ax2.scatter(0.5, 0.6, s=3000, c=direction_color, marker=direction_symbol, alpha=0.8)
    ax2.text(0.5, 0.6, f'{direction_confidence*100:.0f}%', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')
    ax2.text(0.5, 0.35, f'Direction: {predicted_direction.upper()}', ha='center', va='center',
            fontsize=14, fontweight='bold', color=direction_color)

    # Data sources
    sources_text = '\n'.join([f'• {src}' for src in data_sources[:4]])
    ax2.text(0.5, 0.1, f'Data Sources:\n{sources_text}', ha='center', va='center',
            fontsize=9, color='#7f8c8d')

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.85)
    ax2.axis('off')
    ax2.set_title('Direction Prediction', fontsize=14, fontweight='bold', pad=10)

    fig.suptitle(f'{symbol} Prediction Confidence', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def plot_orderbook_3d_surface(
    symbol: str,
    current_price: float,
    expected_range: Tuple[float, float],
    predicted_direction: str,
    strikes: List[float],
    call_oi: List[float],
    put_oi: List[float],
    call_volume: Optional[List[float]] = None,
    put_volume: Optional[List[float]] = None,
    call_iv: Optional[List[float]] = None,
    put_iv: Optional[List[float]] = None,
    max_pain_strike: Optional[float] = None,
    direction_confidence: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 9)
) -> plt.Figure:
    """
    Plot a 3D surface of the option chain order book with prediction overlay.

    The surface shows liquidity (open interest + volume) across strikes for
    both call and put sides. Vertical translucent planes mark the current
    price, the expected move range, the max pain level and the predicted
    target — visualising how the prediction passes through the order book.

    Axes:
      X = strike price
      Y = order book side (negative = puts, positive = calls)
      Z = liquidity (open interest, optionally augmented by volume)

    Args:
        symbol: Stock symbol
        current_price: Current underlying price
        expected_range: (lower_bound, upper_bound) of expected move
        predicted_direction: 'up', 'down', or 'neutral'
        strikes: Sorted list of strike prices
        call_oi: Open interest per strike for calls (same length as strikes)
        put_oi: Open interest per strike for puts (same length as strikes)
        call_volume: Optional call volume per strike
        put_volume: Optional put volume per strike
        call_iv: Optional call implied volatility per strike (used for color)
        put_iv: Optional put implied volatility per strike (used for color)
        max_pain_strike: Optional max pain strike to highlight
        direction_confidence: Direction confidence (0-1) — affects target wall opacity
        save_path: Optional path to save the figure
        figsize: Figure size tuple

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if not strikes:
        ax.text2D(0.5, 0.5, 'No option chain data available',
                  ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        return fig

    strikes_arr = np.asarray(strikes, dtype=float)
    call_oi_arr = np.asarray(call_oi, dtype=float)
    put_oi_arr = np.asarray(put_oi, dtype=float)

    # Combine OI with volume (if given) to form a richer "liquidity" surface
    call_liq = call_oi_arr.copy()
    put_liq = put_oi_arr.copy()
    if call_volume is not None:
        call_liq = call_liq + np.asarray(call_volume, dtype=float)
    if put_volume is not None:
        put_liq = put_liq + np.asarray(put_volume, dtype=float)

    # Build a smooth Y-axis grid: -1 (deep put side) -> 0 (mid) -> +1 (deep call side)
    n_y = 21
    y_grid = np.linspace(-1.0, 1.0, n_y)
    X, Y = np.meshgrid(strikes_arr, y_grid)

    # Z = put liquidity for Y<0, call liquidity for Y>0, taper to 0 at Y=0.
    # Use abs(Y) as a depth weight so the surface peaks at the outer edges
    # (representing the "wall" of liquidity on each side of the book).
    Z = np.zeros_like(X)
    weight = np.abs(Y)
    # Broadcast: each row is one Y level
    call_row = call_liq[np.newaxis, :]  # shape (1, n_strikes)
    put_row = put_liq[np.newaxis, :]
    side_call = (Y > 0).astype(float)
    side_put = (Y < 0).astype(float)
    Z = weight * (side_call * call_row + side_put * put_row)

    # Color by IV if provided, else by Z
    if call_iv is not None and put_iv is not None:
        call_iv_arr = np.asarray(call_iv, dtype=float)
        put_iv_arr = np.asarray(put_iv, dtype=float)
        iv_surface = (
            side_call * call_iv_arr[np.newaxis, :]
            + side_put * put_iv_arr[np.newaxis, :]
        )
        # Normalize IV for colormap
        iv_max = float(np.nanmax(iv_surface)) if np.nanmax(iv_surface) > 0 else 1.0
        facecolors = cm.plasma(iv_surface / iv_max)
        surf = ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, alpha=0.85, shade=False
        )
        mappable = cm.ScalarMappable(cmap=cm.plasma)
        mappable.set_array(iv_surface)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Implied Volatility', fontsize=10)
    else:
        surf = ax.plot_surface(
            X, Y, Z,
            cmap=cm.viridis,
            rstride=1, cstride=1,
            linewidth=0, antialiased=True, alpha=0.85
        )
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Liquidity (OI + Volume)', fontsize=10)

    z_max = float(np.nanmax(Z)) if np.nanmax(Z) > 0 else 1.0

    # --- Prediction overlay: vertical translucent planes through the book ---
    def _vertical_plane(x_val: float, color: str, alpha: float, label: str,
                        linestyle: str = '-', linewidth: float = 2.0):
        yy, zz = np.meshgrid(np.linspace(-1.0, 1.0, 2), np.linspace(0, z_max * 1.05, 2))
        xx = np.full_like(yy, x_val)
        ax.plot_surface(xx, yy, zz, color=color, alpha=alpha,
                        shade=False, linewidth=0)
        # Outline at top edge so the wall reads as a labeled marker
        ax.plot([x_val, x_val], [-1.0, 1.0], [z_max * 1.05, z_max * 1.05],
                color=color, linewidth=linewidth, linestyle=linestyle, label=label)

    lower, upper = expected_range

    # Expected range "tunnel" — translucent bands on lower and upper bounds
    _vertical_plane(lower, '#e74c3c', 0.18, f'Lower: ${lower:.2f}', linestyle='--')
    _vertical_plane(upper, '#27ae60', 0.18, f'Upper: ${upper:.2f}', linestyle='--')

    # Current price wall
    _vertical_plane(current_price, '#2c3e50', 0.30,
                    f'Current: ${current_price:.2f}', linewidth=2.5)

    # Max pain
    if max_pain_strike is not None and max_pain_strike > 0:
        _vertical_plane(max_pain_strike, '#9b59b6', 0.18,
                        f'Max Pain: ${max_pain_strike:.2f}', linestyle=':')

    # Predicted target — a wall biased toward upper or lower bound
    direction_color = {'up': '#27ae60', 'down': '#e74c3c', 'neutral': '#f39c12'}[predicted_direction]
    if predicted_direction == 'up':
        target = current_price + (upper - current_price) * direction_confidence
    elif predicted_direction == 'down':
        target = current_price - (current_price - lower) * direction_confidence
    else:
        target = current_price
    _vertical_plane(target, direction_color, 0.35,
                    f'Predicted ({predicted_direction.upper()}): ${target:.2f}',
                    linewidth=3.0)

    # Direction arrow drawn above the order book surface
    arrow_z = z_max * 1.15
    if predicted_direction in ('up', 'down'):
        ax.quiver(
            current_price, 0, arrow_z,
            target - current_price, 0, 0,
            color=direction_color, linewidth=3, arrow_length_ratio=0.15
        )
        ax.text(
            (current_price + target) / 2, 0, arrow_z * 1.05,
            f'{predicted_direction.upper()} {direction_confidence*100:.0f}%',
            color=direction_color, fontsize=11, fontweight='bold', ha='center'
        )

    # Axis styling
    ax.set_xlabel('Strike Price ($)', fontsize=11, labelpad=10)
    ax.set_ylabel('Order Book Side', fontsize=11, labelpad=10)
    ax.set_zlabel('Liquidity (OI + Volume)', fontsize=11, labelpad=10)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(['Puts', '', 'ATM', '', 'Calls'])
    ax.set_title(
        f'{symbol} Option Chain Order Book — Prediction Surface\n'
        f'Predicted: {predicted_direction.upper()} '
        f'(${target:.2f}, conf {direction_confidence*100:.0f}%)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.view_init(elev=28, azim=-58)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    return fig


def create_prediction_dashboard(
    prediction,
    save_dir: Optional[str] = None,
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Create a complete dashboard of plots for a prediction.

    Args:
        prediction: EarningsPrediction object
        save_dir: Directory to save plots (optional)
        show: Whether to display plots

    Returns:
        Dictionary mapping plot names to Figure objects
    """
    figures = {}

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 1. Expected Move Range
    figures['expected_move'] = plot_expected_move_range(
        symbol=prediction.symbol,
        current_price=prediction.current_price,
        expected_move_percent=prediction.expected_move_percent,
        expected_range=prediction.expected_range,
        historical_moves=prediction.historical_moves,
        predicted_direction=prediction.predicted_direction,
        save_path=os.path.join(save_dir, f'{prediction.symbol}_expected_move.png') if save_dir else None
    )

    # 2. Direction Signals
    figures['direction_signals'] = plot_direction_signals(
        symbol=prediction.symbol,
        direction_signals=prediction.direction_signals,
        direction_confidence=prediction.direction_confidence,
        predicted_direction=prediction.predicted_direction,
        save_path=os.path.join(save_dir, f'{prediction.symbol}_direction_signals.png') if save_dir else None
    )

    # 3. Historical Moves
    figures['historical_moves'] = plot_historical_moves(
        symbol=prediction.symbol,
        historical_moves=prediction.historical_moves,
        expected_move_percent=prediction.expected_move_percent,
        avg_historical_move=prediction.avg_historical_move,
        beat_rate=prediction.historical_beat_rate,
        save_path=os.path.join(save_dir, f'{prediction.symbol}_historical.png') if save_dir else None
    )

    # 4. Options Analysis (if metrics available)
    if prediction.options_metrics:
        figures['options_analysis'] = plot_options_analysis(
            symbol=prediction.symbol,
            current_price=prediction.current_price,
            atm_strike=prediction.options_metrics.atm_strike,
            put_call_ratio=prediction.options_metrics.put_call_ratio,
            max_pain_strike=prediction.options_metrics.max_pain_strike,
            iv_percentile=prediction.iv_percentile,
            atm_iv=prediction.options_metrics.atm_iv,
            flow_sentiment=prediction.flow_sentiment,
            save_path=os.path.join(save_dir, f'{prediction.symbol}_options.png') if save_dir else None
        )

    # 5. Probability Cone
    figures['probability_cone'] = plot_probability_cone(
        symbol=prediction.symbol,
        current_price=prediction.current_price,
        expected_move_percent=prediction.expected_move_percent,
        save_path=os.path.join(save_dir, f'{prediction.symbol}_probability_cone.png') if save_dir else None
    )

    # 6. Confidence Gauge
    figures['confidence'] = plot_confidence_gauge(
        symbol=prediction.symbol,
        overall_confidence=prediction.overall_confidence,
        direction_confidence=prediction.direction_confidence,
        predicted_direction=prediction.predicted_direction,
        data_sources=prediction.data_sources_used,
        save_path=os.path.join(save_dir, f'{prediction.symbol}_confidence.png') if save_dir else None
    )

    # 7. 3D Order Book Surface (when chain data is available)
    chain = getattr(prediction, 'option_chain', None)
    if chain is not None and getattr(chain, 'calls', None) and getattr(chain, 'puts', None):
        strike_set = sorted({c.strike for c in chain.calls} | {p.strike for p in chain.puts})
        call_by_strike = {c.strike: c for c in chain.calls}
        put_by_strike = {p.strike: p for p in chain.puts}

        call_oi = [call_by_strike[s].open_interest if s in call_by_strike else 0 for s in strike_set]
        put_oi = [put_by_strike[s].open_interest if s in put_by_strike else 0 for s in strike_set]
        call_vol = [call_by_strike[s].volume if s in call_by_strike else 0 for s in strike_set]
        put_vol = [put_by_strike[s].volume if s in put_by_strike else 0 for s in strike_set]
        call_iv_list = [call_by_strike[s].implied_volatility if s in call_by_strike else 0 for s in strike_set]
        put_iv_list = [put_by_strike[s].implied_volatility if s in put_by_strike else 0 for s in strike_set]

        figures['orderbook_3d'] = plot_orderbook_3d_surface(
            symbol=prediction.symbol,
            current_price=prediction.current_price,
            expected_range=prediction.expected_range,
            predicted_direction=prediction.predicted_direction,
            strikes=strike_set,
            call_oi=call_oi,
            put_oi=put_oi,
            call_volume=call_vol,
            put_volume=put_vol,
            call_iv=call_iv_list,
            put_iv=put_iv_list,
            max_pain_strike=prediction.options_metrics.max_pain_strike if prediction.options_metrics else None,
            direction_confidence=prediction.direction_confidence,
            save_path=os.path.join(save_dir, f'{prediction.symbol}_orderbook_3d.png') if save_dir else None
        )

    if show:
        plt.show()

    return figures


def save_prediction_plots(prediction, output_dir: str) -> List[str]:
    """
    Convenience function to save all prediction plots.

    Args:
        prediction: EarningsPrediction object
        output_dir: Directory to save plots

    Returns:
        List of saved file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    figures = create_prediction_dashboard(prediction, save_dir=output_dir)

    saved_files = []
    for name, fig in figures.items():
        file_path = os.path.join(output_dir, f'{prediction.symbol}_{name}.png')
        saved_files.append(file_path)
        plt.close(fig)

    return saved_files
