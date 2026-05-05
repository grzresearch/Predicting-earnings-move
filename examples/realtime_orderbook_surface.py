#!/usr/bin/env python3
"""
Build the 3D Order Book Prediction Surface from REAL option-chain data.

Uses the repo's YahooFinanceSource to pull a live option chain for a liquid
ticker that has earnings coming up soon, runs the full EarningsPredictor
pipeline against it, then renders plot_orderbook_3d_surface.

Usage:
    # Auto-pick the first liquid ticker from a watch list whose earnings
    # fall inside the next `--max-days` days:
    python examples/realtime_orderbook_surface.py

    # Force a specific symbol (skips the watch-list scan):
    python examples/realtime_orderbook_surface.py --symbol NVDA

    # Custom output directory:
    python examples/realtime_orderbook_surface.py --output-dir ./my_plots

Requires outbound network access to Yahoo Finance (yfinance).
"""

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Make `src` importable when run from the repo root or examples/
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.analysis.earnings_predictor import EarningsPredictor
from src.data_sources.free_sources import YahooFinanceSource
from src.visualization.plots import plot_orderbook_3d_surface


# Highly liquid US single-name options — typical 24h volume well above 100k contracts.
DEFAULT_WATCHLIST = [
    'NVDA', 'TSLA', 'AAPL', 'AMZN', 'META', 'MSFT', 'GOOGL',
    'AMD', 'NFLX', 'AVGO', 'JPM', 'BAC', 'XOM',
]


def find_liquid_ticker_with_upcoming_earnings(
    candidates,
    max_days_ahead: int = 60,
    min_total_oi: int = 5_000,
):
    """Scan `candidates` for the first symbol whose next earnings is within
    `max_days_ahead` days AND whose post-earnings option chain shows enough
    open interest to be worth visualising."""
    yahoo = YahooFinanceSource()
    today = date.today()
    horizon = today + timedelta(days=max_days_ahead)

    for symbol in candidates:
        try:
            earnings = yahoo.get_earnings_for_symbol(symbol)
            if earnings is None or earnings.earnings_date is None:
                print(f"  {symbol}: no earnings date available, skipping")
                continue

            ed = earnings.earnings_date.date() if hasattr(earnings.earnings_date, 'date') \
                else earnings.earnings_date
            if not (today <= ed <= horizon):
                print(f"  {symbol}: earnings {ed} outside window, skipping")
                continue

            expiration = yahoo.get_nearest_expiration_after_earnings(symbol, ed)
            if expiration is None:
                print(f"  {symbol}: no post-earnings expiration, skipping")
                continue

            chain = yahoo.get_option_chain(symbol, expiration)
            if chain is None or not chain.calls or not chain.puts:
                print(f"  {symbol}: chain unavailable, skipping")
                continue

            total_oi = sum(c.open_interest for c in chain.calls) \
                + sum(p.open_interest for p in chain.puts)
            if total_oi < min_total_oi:
                print(f"  {symbol}: total OI {total_oi} below {min_total_oi}, skipping")
                continue

            print(f"  {symbol}: earnings {ed}, expiration {expiration}, total OI {total_oi:,}")
            return symbol, earnings, chain
        except Exception as exc:
            print(f"  {symbol}: error {exc}")
            continue

    return None, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', help='Force a specific symbol')
    parser.add_argument('--output-dir', default=str(Path(__file__).parent / 'plots'),
                        help='Where to write the PNG and JSON snapshot')
    parser.add_argument('--max-days', type=int, default=60,
                        help='Earnings must fall within this many days (auto mode)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Selecting ticker with real, liquid option chain and upcoming earnings...")
    if args.symbol:
        yahoo = YahooFinanceSource()
        symbol = args.symbol.upper()
        earnings = yahoo.get_earnings_for_symbol(symbol)
        if earnings is None or earnings.earnings_date is None:
            sys.exit(f"No earnings date found for {symbol}")
        ed = earnings.earnings_date.date() if hasattr(earnings.earnings_date, 'date') \
            else earnings.earnings_date
        expiration = yahoo.get_nearest_expiration_after_earnings(symbol, ed)
        if expiration is None:
            sys.exit(f"No post-earnings expiration available for {symbol}")
        chain = yahoo.get_option_chain(symbol, expiration)
        if chain is None or not chain.calls or not chain.puts:
            sys.exit(f"Option chain unavailable for {symbol} at {expiration}")
    else:
        symbol, earnings, chain = find_liquid_ticker_with_upcoming_earnings(
            DEFAULT_WATCHLIST, max_days_ahead=args.max_days
        )
        if symbol is None:
            sys.exit("No liquid ticker with upcoming earnings found in the watch list.")

    print(f"\nUsing {symbol} (earnings {earnings.earnings_date}, expiration {chain.expiration_date})")
    print(f"  Underlying price: ${chain.underlying_price:.2f}")
    print(f"  Calls: {len(chain.calls)}, Puts: {len(chain.puts)}")

    # Run the full predictor so direction/expected range come from the same data.
    predictor = EarningsPredictor(use_free_sources=True)
    prediction = predictor.predict(symbol)

    # Pull per-strike series straight off the live chain.
    strike_set = sorted({c.strike for c in chain.calls} | {p.strike for p in chain.puts})
    call_by_strike = {c.strike: c for c in chain.calls}
    put_by_strike = {p.strike: p for p in chain.puts}

    call_oi = [call_by_strike[s].open_interest if s in call_by_strike else 0 for s in strike_set]
    put_oi = [put_by_strike[s].open_interest if s in put_by_strike else 0 for s in strike_set]
    call_vol = [call_by_strike[s].volume if s in call_by_strike else 0 for s in strike_set]
    put_vol = [put_by_strike[s].volume if s in put_by_strike else 0 for s in strike_set]
    call_iv = [call_by_strike[s].implied_volatility if s in call_by_strike else 0 for s in strike_set]
    put_iv = [put_by_strike[s].implied_volatility if s in put_by_strike else 0 for s in strike_set]

    png_path = output_dir / f'{symbol}_orderbook_3d_surface.png'
    fig = plot_orderbook_3d_surface(
        symbol=symbol,
        current_price=prediction.current_price,
        expected_range=prediction.expected_range,
        predicted_direction=prediction.predicted_direction,
        strikes=strike_set,
        call_oi=call_oi,
        put_oi=put_oi,
        call_volume=call_vol,
        put_volume=put_vol,
        call_iv=call_iv,
        put_iv=put_iv,
        max_pain_strike=prediction.options_metrics.max_pain_strike if prediction.options_metrics else None,
        direction_confidence=prediction.direction_confidence,
        save_path=str(png_path),
    )
    plt.close(fig)
    print(f"\nPNG written: {png_path}")

    # Snapshot the inputs so the figure stays reproducible after the chain moves.
    snapshot = {
        'symbol': symbol,
        'fetched_at': datetime.utcnow().isoformat() + 'Z',
        'earnings_date': earnings.earnings_date.isoformat() if earnings.earnings_date else None,
        'expiration_date': chain.expiration_date.isoformat(),
        'underlying_price': prediction.current_price,
        'expected_range': list(prediction.expected_range),
        'predicted_direction': prediction.predicted_direction,
        'direction_confidence': prediction.direction_confidence,
        'max_pain_strike': prediction.options_metrics.max_pain_strike if prediction.options_metrics else None,
        'strikes': strike_set,
        'call_oi': call_oi,
        'put_oi': put_oi,
        'call_volume': call_vol,
        'put_volume': put_vol,
        'call_iv': call_iv,
        'put_iv': put_iv,
    }
    json_path = output_dir / f'{symbol}_orderbook_3d_surface.json'
    json_path.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"Snapshot written: {json_path}")


if __name__ == '__main__':
    main()
