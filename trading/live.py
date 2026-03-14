#!/usr/bin/env python3
"""
Live paper trading loop.

Usage:
    python -m trading.live [--interval SECONDS] [--symbols BTC,ETH,SOL]

This runs the trading loop continuously with paper trading.
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
import json
import os

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Global flag for graceful shutdown
_shutdown_requested = False
_sleep_task = None

def handle_sigterm(sig, frame):
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown_requested
    print(f"SIGTERM received - initiating graceful shutdown")
    _shutdown_requested = True
    global _sleep_task
    if _sleep_task and not _sleep_task.done():
        _sleep_task.cancel()

# Install signal handlers BEFORE any async code
# SIGINT: Completely ignore (PM2/system noise)
signal.signal(signal.SIGINT, signal.SIG_IGN)
# SIGTERM: Graceful shutdown
signal.signal(signal.SIGTERM, handle_sigterm)

from trading.runner import TradingRunner, TradingConfig
from trading.analysis.regime import RegimeDetector


async def run_live_trading(
    symbols: list[str],
    interval_seconds: int = 3600,
    storage_path: str = "data/trading"
):
    """
    Run live paper trading.

    Args:
        symbols: List of trading symbols
        interval_seconds: Time between steps
        storage_path: Path for storing state
    """
    global _shutdown_requested, _sleep_task

    print(f"Starting live paper trading at {datetime.now(timezone.utc).isoformat()}")
    print(f"Symbols: {symbols}")
    print(f"Interval: {interval_seconds}s")

    # Configure
    config = TradingConfig(
        symbols=symbols,
        horizons=[1, 4, 8, 12, 24],
        min_consensus_strength=0.5,  # Lower threshold for more signals
        min_direction_agreement=0.6,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
    )

    runner = TradingRunner(config=config, storage_path=Path(storage_path))

    # Track performance
    step_count = 0
    total_pnl = 0.0

    try:
        await runner.initialize()
        print("System initialized. Starting trading loop...")
        print(f"DEBUG: runner._running = {runner._running}")

        # Send initial message
        await send_update(f"Trading system started. Monitoring {len(symbols)} symbols.")

        print("DEBUG: Entering main loop...")
        while runner._running and not _shutdown_requested:
            step_count += 1
            step_start = datetime.now(timezone.utc)

            try:
                # Run step
                result = await runner.run_step()

                # Log result
                step_time = result["step_time"]
                signals = result["signals_emitted"]
                trades = result["trades_executed"]
                sharpe = result["performance"]["sharpe"]

                print(f"\n=== Step {step_count} ===")
                print(f"Time: {step_start.isoformat()}")
                print(f"Duration: {step_time:.2f}s")
                print(f"Signals: {signals}, Trades: {trades}")
                print(f"Sharpe: {sharpe:.3f}")
                print(f"Positions: {result['open_positions']}")

                # Track PnL
                if trades > 0:
                    total_pnl = result["performance"]["total_return"]
                    print(f"Total PnL: {total_pnl:.4f}")

                # Periodic update (every 24 steps = 24 hours)
                if step_count % 24 == 0:
                    await send_update(
                        f"24h Update: Sharpe={sharpe:.2f}, "
                        f"PnL={total_pnl:.4f}, "
                        f"Trades={result['performance']['trades']}"
                    )

            except Exception as e:
                print(f"Error in step {step_count}: {e}")
                import traceback
                traceback.print_exc()

            # Wait for next interval (cancellable)
            if not _shutdown_requested:
                print(f"DEBUG: Sleeping for {interval_seconds}s, _running={runner._running}")
                try:
                    _sleep_task = asyncio.create_task(asyncio.sleep(interval_seconds))
                    await _sleep_task
                    print(f"DEBUG: Woke up, _running={runner._running}")
                except asyncio.CancelledError:
                    print("DEBUG: Sleep cancelled by signal")

        if _shutdown_requested:
            print("Shutdown requested, exiting gracefully...")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt in run_live_trading!")
        print("Shutting down...")
    except Exception as e:
        print(f"\nException in run_live_trading: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await runner.close()
        print("Trading stopped.")


async def send_update(message: str):
    """Send update message via arbos.py."""
    try:
        import subprocess
        print(f"Sending update: {message[:50]}...")
        subprocess.run(
            ["python", "arbos.py", "send", message],
            cwd="/Arbos",
            capture_output=True,
            timeout=10
        )
        print("Update sent successfully")
    except Exception as e:
        print(f"Failed to send update: {e}")


def main():
    parser = argparse.ArgumentParser(description="Live paper trading")
    parser.add_argument(
        "--symbols",
        default="BTC,ETH,SOL",
        help="Comma-separated list of symbols"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Interval between steps in seconds"
    )
    parser.add_argument(
        "--storage",
        default="data/trading",
        help="Storage path"
    )

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # Run
    print("DEBUG: Starting asyncio.run()...")
    try:
        asyncio.run(run_live_trading(
            symbols=symbols,
            interval_seconds=args.interval,
            storage_path=args.storage
        ))
        print("DEBUG: asyncio.run() completed normally")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt in main!")
        print("Stopped by user.")
    except Exception as e:
        print(f"\nException in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
