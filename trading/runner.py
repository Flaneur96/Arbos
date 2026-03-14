"""
Trading system runner - Main loop implementation.

This is S₀: the initial trading system that will evolve over time.

The loop:
1. Collect data (features from Hyperliquid + Coinglass)
2. Generate predictions (foundation models + horizon ensemble)
3. Apply consensus gating (filter signals)
4. Execute trades (Hyperliquid)
5. Evaluate performance (Sharpe, PnL, drawdown)
6. Evolve models (mutation/selection)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from pathlib import Path
import json
import pandas as pd
import numpy as np
from collections import deque

from .data.hyperliquid import HyperliquidClient
from .data.coinglass import CoinglassClient
from .data.feature_store import FeatureStore, FeatureConfig
from .models.foundation import (
    FoundationModelRegistry,
    create_default_registry,
    ForecastResult
)
from .models.ensemble import (
    HorizonEnsemble,
    AdaptiveHorizonEnsemble,
    HorizonPrediction,
    EnsemblePrediction
)
from .analysis.regime import RegimeDetector, MarketRegime


@dataclass
class TradingConfig:
    """Configuration for trading system."""
    # Symbols to trade
    symbols: list[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL"])

    # Position sizing
    max_position_size: float = 0.1  # 10% of capital per position
    max_leverage: float = 3.0

    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit

    # Prediction horizons
    horizons: list[int] = field(default_factory=lambda: [1, 4, 8, 12, 24])

    # Consensus gating
    min_consensus_strength: float = 0.6
    min_direction_agreement: float = 0.7

    # Evolution settings
    population_size: int = 10
    mutation_rate: float = 0.1
    elite_ratio: float = 0.2

    # Data settings
    feature_history_hours: int = 720  # 30 days

    # Evaluation
    sharpe_window: int = 168  # 7 days for rolling Sharpe


@dataclass
class Position:
    """Open position."""
    symbol: str
    side: int  # 1 = long, -1 = short
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def current_pnl(self, current_price: float) -> float:
        """Calculate current PnL."""
        return self.side * (current_price - self.entry_price) / self.entry_price


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    direction: int  # -1, 0, 1
    strength: float  # 0.0 to 1.0
    predicted_return: float
    horizon: int
    consensus: Optional[EnsemblePrediction] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class PerformanceMetrics:
    """Performance metrics for evaluation."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades_count: int = 0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class TradingRunner:
    """
    Main trading system runner.

    Implements the loop:
    - Collect data
    - Generate predictions
    - Apply consensus gating
    - Execute trades
    - Evaluate performance
    - Evolve models
    """

    def __init__(
        self,
        config: Optional[TradingConfig] = None,
        storage_path: Optional[Path] = None
    ):
        self.config = config or TradingConfig()
        self.storage_path = storage_path or Path("data/trading")

        # Clients
        self._hyperliquid: Optional[HyperliquidClient] = None
        self._coinglass: Optional[CoinglassClient] = None
        self._feature_store: Optional[FeatureStore] = None

        # Models
        self._model_registry: Optional[FoundationModelRegistry] = None
        self._horizon_ensemble: Optional[AdaptiveHorizonEnsemble] = None

        # Regime detector
        self._regime_detector: Optional[RegimeDetector] = None

        # State
        self._positions: dict[str, Position] = {}
        self._signals_history: deque[Signal] = deque(maxlen=10000)
        self._pnl_history: deque[float] = deque(maxlen=10000)
        self._trades: list[dict] = []

        # Evolution state
        self._model_population: dict[str, dict] = {}
        self._generation: int = 0

        # Performance tracking
        self._performance: PerformanceMetrics = PerformanceMetrics()

        # Running state
        self._running = False
        self._last_run: Optional[datetime] = None

    async def initialize(self):
        """Initialize all components."""
        print("Initializing trading system...")

        # Initialize clients
        self._hyperliquid = HyperliquidClient()
        self._coinglass = CoinglassClient()

        await self._hyperliquid.connect()
        await self._coinglass.connect()

        # Initialize feature store
        self._feature_store = FeatureStore(
            self._hyperliquid,
            self._coinglass,
            storage_path=self.storage_path / "features"
        )

        await self._feature_store.initialize(self.config.symbols)

        # Initialize models
        self._model_registry = create_default_registry()
        await self._model_registry.initialize_all()

        # Initialize horizon ensemble
        self._horizon_ensemble = AdaptiveHorizonEnsemble(
            horizons=self.config.horizons
        )

        # Initialize regime detector
        self._regime_detector = RegimeDetector()

        # Initialize model population for evolution
        self._initialize_population()

        # Load saved state if exists
        await self._load_state()

        # Set running flag
        self._running = True

        print(f"System initialized with {len(self.config.symbols)} symbols")
        print(f"Available models: {self._model_registry.list_models()}")

    async def close(self):
        """Close all connections."""
        if self._hyperliquid:
            await self._hyperliquid.close()
        if self._coinglass:
            await self._coinglass.close()

    def _initialize_population(self):
        """Initialize model population for evolution."""
        available_models = self._model_registry.list_models()

        for model_name in available_models:
            self._model_population[model_name] = {
                "name": model_name,
                "fitness": 0.0,
                "generations": 0,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "sharpe": 0.0,
                "predictions": 0,
                "weight": 1.0 / len(available_models)
            }

    async def _load_state(self):
        """Load saved state from disk."""
        state_file = self.storage_path / "state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)

                self._generation = data.get("generation", 0)

                # Restore performance
                if "performance" in data:
                    self._performance = PerformanceMetrics(**data["performance"])

                # Restore positions
                if "positions" in data:
                    for sym, pos_data in data["positions"].items():
                        self._positions[sym] = Position(
                            symbol=sym,
                            side=pos_data["side"],
                            size=pos_data["size"],
                            entry_price=pos_data["entry_price"],
                            entry_time=datetime.fromisoformat(pos_data["entry_time"]) if isinstance(pos_data["entry_time"], str) else pos_data["entry_time"]
                        )
                    print(f"Restored {len(self._positions)} positions from state")

                # Restore PnL history
                if "pnl_history" in data:
                    self._pnl_history = deque(data["pnl_history"], maxlen=10000)

                print(f"Loaded state: generation {self._generation}, positions: {len(self._positions)}")

            except Exception as e:
                print(f"Error loading state: {e}")

    async def _save_state(self):
        """Save state to disk."""
        self.storage_path.mkdir(parents=True, exist_ok=True)

        state_file = self.storage_path / "state.json"

        data = {
            "generation": self._generation,
            "performance": {
                "total_return": self._performance.total_return,
                "sharpe_ratio": self._performance.sharpe_ratio,
                "max_drawdown": self._performance.max_drawdown,
                "win_rate": self._performance.win_rate,
                "trades_count": self._performance.trades_count,
                "timestamp": self._performance.timestamp.isoformat() if hasattr(self._performance.timestamp, 'isoformat') else str(self._performance.timestamp)
            },
            "positions": {
                sym: {
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "entry_time": pos.entry_time.isoformat() if hasattr(pos.entry_time, 'isoformat') else str(pos.entry_time)
                }
                for sym, pos in self._positions.items()
            },
            "pnl_history": list(self._pnl_history)[-1000:]  # Last 1000
        }

        with open(state_file, "w") as f:
            json.dump(data, f, indent=2)

    # ==================== MAIN LOOP ====================

    async def run_step(self) -> dict:
        """
        Execute one step of the trading loop.

        Returns:
            Dict with step results
        """
        step_start = datetime.now(timezone.utc)

        # 1. Collect data
        features = await self._collect_data()

        # 2. Generate predictions
        predictions = await self._generate_predictions(features)

        # 3. Apply consensus gating
        signals = self._apply_consensus_gating(predictions)

        # 4. Execute trades (or manage existing positions)
        trades = await self._execute_signals(signals)

        # 5. Evaluate performance
        self._update_performance()

        # 6. Evolve models (periodically)
        if self._should_evolve():
            await self._evolve_models()

        # Save state
        await self._save_state()

        step_end = datetime.now(timezone.utc)
        step_duration = (step_end - step_start).total_seconds()

        return {
            "step_time": step_duration,
            "features_collected": len(features),
            "predictions_generated": len(predictions),
            "signals_emitted": len(signals),
            "trades_executed": len(trades),
            "open_positions": len(self._positions),
            "generation": self._generation,
            "performance": {
                "sharpe": self._performance.sharpe_ratio,
                "total_return": self._performance.total_return,
                "trades": self._performance.trades_count
            }
        }

    async def run_continuous(self, interval_seconds: int = 3600):
        """
        Run the trading loop continuously.

        Args:
            interval_seconds: Time between steps (default 1 hour)
        """
        self._running = True
        print(f"Starting continuous trading with {interval_seconds}s interval")

        while self._running:
            try:
                result = await self.run_step()
                print(f"Step completed: {result}")

                self._last_run = datetime.now(timezone.utc)

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                print(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

    def stop(self):
        """Stop the trading loop."""
        self._running = False

    # ==================== DATA COLLECTION ====================

    async def _collect_data(self) -> dict[str, pd.DataFrame]:
        """
        Collect latest data for all symbols.

        Returns:
            Dict of symbol -> features DataFrame
        """
        features = {}

        for symbol in self.config.symbols:
            try:
                df = await self._feature_store.update_features(symbol)
                if not df.empty:
                    features[symbol] = df

            except Exception as e:
                print(f"Error collecting data for {symbol}: {e}")

        return features

    # ==================== PREDICTIONS ====================

    async def _generate_predictions(
        self,
        features: dict[str, pd.DataFrame]
    ) -> dict[str, list[HorizonPrediction]]:
        """
        Generate predictions for all symbols.

        Args:
            features: Dict of symbol -> features

        Returns:
            Dict of symbol -> list of horizon predictions
        """
        predictions = {}

        for symbol, df in features.items():
            if df.empty:
                continue

            symbol_predictions = []

            # Get price series
            close_prices = df["close"].values

            # Generate predictions for each horizon
            for horizon in self.config.horizons:
                try:
                    # Use ensemble of foundation models
                    horizon_preds = await self._predict_horizon(
                        symbol, close_prices, horizon
                    )
                    symbol_predictions.extend(horizon_preds)

                except Exception as e:
                    print(f"Error predicting {symbol} h={horizon}: {e}")

            predictions[symbol] = symbol_predictions

        return predictions

    async def _predict_horizon(
        self,
        symbol: str,
        prices: np.ndarray,
        horizon: int
    ) -> list[HorizonPrediction]:
        """
        Generate prediction for a specific horizon.

        Uses multiple foundation models and aggregates.
        """
        predictions = []

        # Detect regime first
        regime = None
        if self._regime_detector and len(prices) >= 24:
            regime = self._regime_detector.detect(prices)

        # Get available models
        for model_name in self._model_registry.list_models():
            try:
                model = await self._model_registry.get_model(model_name)

                # Handle LLMForecaster separately (LLM-based)
                model_type = str(type(model))
                if 'LLMForecaster' in model_type or 'llama' in model_name.lower():
                    # LLM-based forecast
                    context = {}
                    try:
                        if self._coinglass and hasattr(self._coinglass, '_connected') and self._coinglass._connected:
                            funding = await self._coinglass.get_funding_rate(symbol)
                            context["funding_rate"] = funding
                    except:
                        pass

                    try:
                        llm_result = await model.forecast(symbol, prices, context=context)
                        # Convert to HorizonPrediction
                        idx = min(horizon-1, len(llm_result.predictions)-1)
                        pred = HorizonPrediction(
                            horizon=horizon,
                            predicted_return=(llm_result.predictions[idx] - prices[-1]) / prices[-1],
                            confidence=llm_result.confidence,
                            direction=llm_result.direction
                        )
                        predictions.append(pred)
                        continue
                    except Exception as e:
                        print(f"LLM forecast error: {e}")

                # Standard foundation model forecast
                result = await model.forecast(
                    prices[-168:], # Last week of data
                    horizon=horizon
                )

                # Calculate predicted return
                if len(result.predictions) > 0:
                    current_price = prices[-1]
                    predicted_price = result.predictions[horizon - 1] if horizon <= len(result.predictions) else result.predictions[-1]
                    predicted_return = (predicted_price - current_price) / current_price

                    # Adjust based on regime
                    if regime and regime.is_trending:
                        if predicted_return > 0:
                            predicted_return *= 1.2
                        else:
                            predicted_return *= 0.8

                    # Determine direction
                    if predicted_return > 0.005:
                        direction = 1
                    elif predicted_return < -0.005:
                        direction = -1
                    else:
                        direction = 0

                    # Confidence based on prediction interval
                    if result.lower_bound is not None and result.upper_bound is not None:
                        interval_width = (result.upper_bound[horizon-1] - result.lower_bound[horizon-1]) / current_price
                        confidence = max(0.1, 1.0 - interval_width)
                    else:
                        confidence = 0.5

                    # Boost confidence if regime agrees
                    if regime and regime.confidence > 0.7:
                        confidence = min(1.0, confidence * 1.1)

                    pred = HorizonPrediction(
                        horizon=horizon,
                        predicted_return=predicted_return,
                        confidence=confidence,
                        direction=direction
                    )
                    predictions.append(pred)

            except Exception as e:
                print(f"Error with model {model_name}: {e}")

        return predictions

    def _apply_consensus_gating(
        self,
        predictions: dict[str, list[HorizonPrediction]]
    ) -> list[Signal]:
        """
        Apply consensus gating to filter signals.

        Only emit signals when there is strong agreement
        across models and horizons.
        """
        signals = []

        for symbol, preds in predictions.items():
            if not preds:
                print(f"DEBUG: {symbol} - no predictions")
                continue

            # Combine predictions using horizon ensemble
            ensemble_pred = self._horizon_ensemble.combine_predictions(preds)

            # Debug log - prediction distribution
            up_votes = sum(1 for p in preds if p.direction > 0)
            down_votes = sum(1 for p in preds if p.direction < 0)
            neutral_votes = sum(1 for p in preds if p.direction == 0)
            print(f"DEBUG: {symbol} - preds={len(preds)}, up={up_votes}, down={down_votes}, neutral={neutral_votes}, consensus={ensemble_pred.consensus_strength:.2f}")

            # Check consensus strength
            if ensemble_pred.consensus_strength < self.config.min_consensus_strength:
                print(f"DEBUG: {symbol} - filtered (consensus {ensemble_pred.consensus_strength:.2f} < {self.config.min_consensus_strength})")
                continue

            # Check direction agreement
            direction_votes = sum(1 for p in preds if p.direction == ensemble_pred.consensus_direction)
            direction_agreement = direction_votes / len(preds) if preds else 0

            if direction_agreement < self.config.min_direction_agreement:
                print(f"DEBUG: {symbol} - filtered (direction_agreement {direction_agreement:.2f} < {self.config.min_direction_agreement})")
                continue

            # Create signal
            signal = Signal(
                symbol=symbol,
                direction=ensemble_pred.consensus_direction,
                strength=ensemble_pred.consensus_strength,
                predicted_return=ensemble_pred.weighted_return,
                horizon=self.config.horizons[-1],  # Use longest horizon
                consensus=ensemble_pred
            )

            signals.append(signal)
            self._signals_history.append(signal)

        return signals

    # ==================== TRADE EXECUTION ====================

    async def _execute_signals(self, signals: list[Signal]) -> list[dict]:
        """
        Execute trading signals.

        For now, this is paper trading - no real orders.
        """
        trades = []

        for signal in signals:
            # Check if we already have a position
            if signal.symbol in self._positions:
                pos = self._positions[signal.symbol]

                # Close if signal is opposite
                if pos.side != signal.direction:
                    trade = await self._close_position(signal.symbol, signal)
                    if trade:
                        trades.append(trade)

            # Open new position if signal is strong
            if signal.strength > 0.5 and signal.symbol not in self._positions:
                trade = await self._open_position(signal)
                if trade:
                    trades.append(trade)

        return trades

    async def _open_position(self, signal: Signal) -> Optional[dict]:
        """Open a new position based on signal."""
        # Get current price
        try:
            mids = await self._hyperliquid.get_mids()
            price = mids.get(signal.symbol)

            if not price:
                return None

            # Calculate position size (paper trading)
            size = self.config.max_position_size * signal.strength

            position = Position(
                symbol=signal.symbol,
                side=signal.direction,
                size=size,
                entry_price=price,
                entry_time=datetime.now(timezone.utc),
                stop_loss=price * (1 - signal.direction * self.config.stop_loss_pct),
                take_profit=price * (1 + signal.direction * self.config.take_profit_pct)
            )

            self._positions[signal.symbol] = position

            trade = {
                "action": "open",
                "symbol": signal.symbol,
                "side": "long" if signal.direction > 0 else "short",
                "price": price,
                "size": size,
                "signal_strength": signal.strength,
                "predicted_return": signal.predicted_return,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self._trades.append(trade)
            return trade

        except Exception as e:
            print(f"Error opening position: {e}")
            return None

    async def _close_position(self, symbol: str, signal: Signal) -> Optional[dict]:
        """Close an existing position."""
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]

        try:
            mids = await self._hyperliquid.get_mids()
            price = mids.get(symbol)

            if not price:
                return None

            pnl = position.current_pnl(price)
            self._pnl_history.append(pnl)

            trade = {
                "action": "close",
                "symbol": symbol,
                "side": "long" if position.side > 0 else "short",
                "entry_price": position.entry_price,
                "exit_price": price,
                "pnl": pnl,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self._trades.append(trade)
            del self._positions[symbol]

            return trade

        except Exception as e:
            print(f"Error closing position: {e}")
            return None

    # ==================== PERFORMANCE EVALUATION ====================

    def _update_performance(self):
        """Update performance metrics."""
        if len(self._pnl_history) < 2:
            return

        pnls = np.array(list(self._pnl_history))

        # Total return
        self._performance.total_return = np.sum(pnls)

        # Sharpe ratio (annualized, hourly data)
        if len(pnls) > 24:
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls) + 1e-8
            self._performance.sharpe_ratio = mean_pnl / std_pnl * np.sqrt(365 * 24)

        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        self._performance.max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Win rate
        wins = np.sum(pnls > 0)
        losses = np.sum(pnls < 0)
        total = wins + losses
        self._performance.win_rate = wins / total if total > 0 else 0

        # Trade count
        self._performance.trades_count = len([t for t in self._trades if t.get("action") == "close"])

    # ==================== MODEL EVOLUTION ====================

    def _should_evolve(self) -> bool:
        """Check if we should run evolution."""
        # Evolve every 24 hours worth of data
        return len(self._pnl_history) > 0 and len(self._pnl_history) % 24 == 0

    async def _evolve_models(self):
        """
        Run evolutionary model search.

        - Evaluate fitness of each model
        - Select elite performers
        - Mutate/reproduce to create next generation
        """
        self._generation += 1
        print(f"Running evolution: generation {self._generation}")

        # Calculate fitness for each model
        await self._evaluate_model_fitness()

        # Selection
        elites = self._select_elites()

        # Mutation
        await self._mutate_population(elites)

        # Log evolution results
        print(f"Evolution complete. Top models: {elites[:3]}")

    async def _evaluate_model_fitness(self):
        """Evaluate fitness of each model in population."""
        # For now, use overall performance as fitness
        # In full implementation, would track per-model predictions

        for model_name, model_data in self._model_population.items():
            # Fitness based on Sharpe and win rate
            sharpe = self._performance.sharpe_ratio
            win_rate = self._performance.win_rate

            # Combined fitness
            model_data["fitness"] = sharpe * 0.7 + win_rate * 0.3
            model_data["sharpe"] = sharpe
            model_data["generations"] += 1

    def _select_elites(self) -> list[str]:
        """Select elite models based on fitness."""
        sorted_models = sorted(
            self._model_population.items(),
            key=lambda x: x[1]["fitness"],
            reverse=True
        )

        elite_count = max(1, int(self.config.elite_ratio * len(sorted_models)))
        elites = [name for name, _ in sorted_models[:elite_count]]

        return elites

    async def _mutate_population(self, elites: list[str]):
        """Mutate population based on elites."""
        # Increase weight of elite models
        total_weight = 0

        for model_name, model_data in self._model_population.items():
            if model_name in elites:
                model_data["weight"] *= 1.1  # Boost elite weight
            else:
                model_data["weight"] *= 0.9  # Reduce non-elite weight

            total_weight += model_data["weight"]

        # Normalize weights
        for model_data in self._model_population.values():
            model_data["weight"] /= total_weight


async def main():
    """Test the trading runner."""
    runner = TradingRunner()

    try:
        await runner.initialize()

        # Run a single step
        result = await runner.run_step()
        print(f"Step result: {json.dumps(result, indent=2)}")

    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main())
