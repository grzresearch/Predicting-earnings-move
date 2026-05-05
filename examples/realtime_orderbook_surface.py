#!/usr/bin/env python3
"""
Build the 3D Order Book Prediction Surface from REAL option-chain data.

Two modes:

  Live mode (default) — fetches a live chain via YahooFinanceSource:
      python examples/realtime_orderbook_surface.py
      python examples/realtime_orderbook_surface.py --symbol NVDA

  Snapshot mode — re-renders the PNG from a JSON snapshot produced by an
  earlier live run. Useful when the host running the visualisation has no
  outbound network access:
      python examples/realtime_orderbook_surface.py \
          --from-snapshot examples/plots/NVDA_orderbook_3d_surface.json

Each live run also writes the JSON snapshot next to the PNG so the figure
stays reproducible after the chain moves.
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
    from src.data_sources.free_sources import YahooFinanceSource
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


def render_from_payload(payload: dict, output_dir: Path) -> Path:
    """Render the 3D surface from an already-collected payload (dict)."""
    symbol = payload['symbol']
    png_path = output_dir / f'{symbol}_orderbook_3d_surface.png'
    fig = plot_orderbook_3d_surface(
        symbol=symbol,
        current_price=float(payload['underlying_price']),
        expected_range=tuple(payload['expected_range']),
        predicted_direction=payload['predicted_direction'],
        strikes=[float(s) for s in payload['strikes']],
        call_oi=[float(v) for v in payload['call_oi']],
        put_oi=[float(v) for v in payload['put_oi']],
        call_volume=[float(v) for v in payload.get('call_volume') or []] or None,
        put_volume=[float(v) for v in payload.get('put_volume') or []] or None,
        call_iv=[float(v) for v in payload.get('call_iv') or []] or None,
        put_iv=[float(v) for v in payload.get('put_iv') or []] or None,
        max_pain_strike=float(payload['max_pain_strike']) if payload.get('max_pain_strike') is not None else None,
        direction_confidence=float(payload.get('direction_confidence', 0.5)),
        save_path=str(png_path),
    )
    plt.close(fig)
    return png_path


def run_live(args, output_dir: Path) -> dict:
    """Fetch a live chain, build the prediction, render the PNG, return the snapshot dict."""
    from src.analysis.earnings_predictor import EarningsPredictor
    from src.data_sources.free_sources import YahooFinanceSource

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

    predictor = EarningsPredictor(use_free_sources=True)
    prediction = predictor.predict(symbol)

    strike_set = sorted({c.strike for c in chain.calls} | {p.strike for p in chain.puts})
    call_by_strike = {c.strike: c for c in chain.calls}
    put_by_strike = {p.strike: p for p in chain.puts}

    payload = {
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
        'call_oi': [call_by_strike[s].open_interest if s in call_by_strike else 0 for s in strike_set],
        'put_oi': [put_by_strike[s].open_interest if s in put_by_strike else 0 for s in strike_set],
        'call_volume': [call_by_strike[s].volume if s in call_by_strike else 0 for s in strike_set],
        'put_volume': [put_by_strike[s].volume if s in put_by_strike else 0 for s in strike_set],
        'call_iv': [call_by_strike[s].implied_volatility if s in call_by_strike else 0 for s in strike_set],
        'put_iv': [put_by_strike[s].implied_volatility if s in put_by_strike else 0 for s in strike_set],
    }

    png_path = render_from_payload(payload, output_dir)
    print(f"\nPNG written: {png_path}")

    json_path = output_dir / f'{symbol}_orderbook_3d_surface.json'
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Snapshot written: {json_path}")

    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', help='Force a specific symbol (live mode)')
    parser.add_argument('--output-dir', default=str(Path(__file__).parent / 'plots'),
                        help='Where to write the PNG and JSON snapshot')
    parser.add_argument('--max-days', type=int, default=60,
                        help='Earnings must fall within this many days (live, auto mode)')
    parser.add_argument('--from-snapshot', type=Path,
                        help='Render the PNG from a JSON snapshot instead of fetching live data')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_snapshot:
        snapshot_path = Path(args.from_snapshot)
        if not snapshot_path.exists():
            sys.exit(f"Snapshot not found: {snapshot_path}")
        payload = json.loads(snapshot_path.read_text())
        print(f"Rendering {payload['symbol']} from snapshot {snapshot_path}")
        png_path = render_from_payload(payload, output_dir)
        print(f"PNG written: {png_path}")
        return

    print("Selecting ticker with real, liquid option chain and upcoming earnings...")
    run_live(args, output_dir)


if __name__ == '__main__':
    main()
