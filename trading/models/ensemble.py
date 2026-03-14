"""
Horizon ensembles for multi-horizon forecasting.

H = {1h, 4h, 8h, 12h, 24h}

Ensemble combines predictions from multiple horizons
to generate more robust signals.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd
from collections import deque


# Default horizons
HORIZONS = [1, 4, 8, 12, 24]  # hours


@dataclass
class HorizonPrediction:
    """Prediction from a single horizon model."""
    horizon: int  # hours
    predicted_return: float
    confidence: float
    direction: int  # -1, 0, 1
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class EnsemblePrediction:
    """Combined prediction from horizon ensemble."""
    predictions: list[HorizonPrediction]
    weighted_return: float
    consensus_direction: int
    consensus_strength: float
    horizon_weights: dict[int, float] = field(default_factory=dict)
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    @property
    def is_strong_consensus(self) -> bool:
        """Check if ensemble has strong consensus."""
        return self.consensus_strength > 0.7


class HorizonEnsemble:
    """
    Ensemble that combines forecasts across multiple horizons.

    Uses weighted combination based on:
    - Historical accuracy per horizon
    - Horizon reliability in current regime
    - Signal agreement across horizons
    """

    def __init__(
        self,
        horizons: list[int] = None,
        weights: Optional[dict[int, float]] = None
    ):
        """
        Initialize horizon ensemble.

        Args:
            horizons: List of forecast horizons in hours
            weights: Optional horizon weights
        """
        self.horizons = horizons or HORIZONS
        self.weights = weights or self._default_weights()

        # Performance tracking per horizon
        self._horizon_mae: dict[int, deque] = {
            h: deque(maxlen=1000) for h in self.horizons
        }
        self._horizon_sharpe: dict[int, deque] = {
            h: deque(maxlen=1000) for h in self.horizons
        }

        # Adaptive weights
        self._adaptive_weights = self.weights.copy()
        self._weight_history: list[dict[int, float]] = []

    def _default_weights(self) -> dict[int, float]:
        """Default horizon weights - favor medium horizons."""
        # Weight distribution favoring 4h-12h horizons
        raw = {
            1: 0.1,   # Short-term: lower weight (more noise)
            4: 0.25,  # Medium-term: higher weight
            8: 0.25,  # Medium-term: higher weight
            12: 0.2,  # Medium-long: moderate weight
            24: 0.2,  # Long-term: moderate weight
        }
        # Normalize
        total = sum(raw.values())
        return {h: w / total for h, w in raw.items()}

    def update_weights_from_performance(self):
        """
        Update weights based on historical performance.

        Better performing horizons get higher weights.
        """
        if not all(len(self._horizon_sharpe[h]) > 10 for h in self.horizons):
            return

        # Calculate Sharpe per horizon
        sharpe_scores = {}
        for h in self.horizons:
            returns = list(self._horizon_sharpe[h])
            if len(returns) > 1:
                mean_r = np.mean(returns)
                std_r = np.std(returns) + 1e-8
                sharpe_scores[h] = mean_r / std_r
            else:
                sharpe_scores[h] = 0

        # Convert to weights (softmax)
        temps = np.array([sharpe_scores[h] for h in self.horizons])
        exp_temps = np.exp(temps - np.max(temps))
        weights = exp_temps / exp_temps.sum()

        # Blend with default weights
        for i, h in enumerate(self.horizons):
            self._adaptive_weights[h] = (
                0.7 * weights[i] +  # Performance-based
                0.3 * self.weights[h]  # Prior
            )

        # Normalize
        total = sum(self._adaptive_weights.values())
        self._adaptive_weights = {
            h: w / total for h, w in self._adaptive_weights.items()
        }

        self._weight_history.append(self._adaptive_weights.copy())

    def combine_predictions(
        self,
        predictions: list[HorizonPrediction]
    ) -> EnsemblePrediction:
        """
        Combine predictions from multiple horizons.

        Args:
            predictions: List of horizon predictions

        Returns:
            EnsemblePrediction with combined signal
        """
        if not predictions:
            raise ValueError("No predictions to combine")

        # Get weights
        weights = self._adaptive_weights.copy()

        # Weighted return
        weighted_return = 0
        total_weight = 0

        for pred in predictions:
            w = weights.get(pred.horizon, 1 / len(self.horizons))
            weighted_return += pred.predicted_return * w
            total_weight += w

        if total_weight > 0:
            weighted_return /= total_weight

        # Calculate horizon weights for this ensemble
        horizon_weights = {
            p.horizon: weights.get(p.horizon, 1 / len(self.horizons))
            for p in predictions
        }

        # Consensus direction
        directions = [p.direction for p in predictions]
        up_votes = sum(1 for d in directions if d > 0)
        down_votes = sum(1 for d in directions if d < 0)

        if up_votes > down_votes:
            consensus_direction = 1
        elif down_votes > up_votes:
            consensus_direction = -1
        else:
            consensus_direction = 0

        # Consensus strength (weighted agreement)
        direction_signals = []
        for pred in predictions:
            w = weights.get(pred.horizon, 1 / len(self.horizons))
            if pred.direction > 0:
                direction_signals.append(w)
            elif pred.direction < 0:
                direction_signals.append(-w)

        consensus_strength = abs(sum(direction_signals))

        return EnsemblePrediction(
            predictions=predictions,
            weighted_return=weighted_return,
            consensus_direction=consensus_direction,
            consensus_strength=consensus_strength,
            horizon_weights=horizon_weights
        )

    def record_outcome(
        self,
        horizon: int,
        predicted_return: float,
        actual_return: float
    ):
        """
        Record prediction outcome for performance tracking.

        Args:
            horizon: Prediction horizon
            predicted_return: Predicted return
            actual_return: Actual return
        """
        if horizon not in self._horizon_mae:
            return

        # Record MAE
        mae = abs(predicted_return - actual_return)
        self._horizon_mae[horizon].append(mae)

        # Record return (for Sharpe calculation)
        # Positive if prediction direction was correct
        direction_correct = (
            (predicted_return > 0 and actual_return > 0) or
            (predicted_return < 0 and actual_return < 0)
        )
        realized = actual_return if direction_correct else -abs(actual_return)
        self._horizon_sharpe[horizon].append(realized)

    def get_horizon_performance(self) -> dict[int, dict]:
        """Get performance metrics per horizon."""
        metrics = {}

        for h in self.horizons:
            mae_values = list(self._horizon_mae[h])
            sharpe_values = list(self._horizon_sharpe[h])

            if mae_values:
                metrics[h] = {
                    "mae_mean": np.mean(mae_values),
                    "mae_std": np.std(mae_values),
                    "sharpe": np.mean(sharpe_values) / (np.std(sharpe_values) + 1e-8)
                        if sharpe_values else 0,
                    "samples": len(mae_values)
                }
            else:
                metrics[h] = {
                    "mae_mean": 0,
                    "mae_std": 0,
                    "sharpe": 0,
                    "samples": 0
                }

        return metrics

    def get_weights_history(self) -> pd.DataFrame:
        """Get history of weight adaptations."""
        if not self._weight_history:
            return pd.DataFrame()

        return pd.DataFrame(self._weight_history)


class AdaptiveHorizonEnsemble(HorizonEnsemble):
    """
    Enhanced ensemble with regime-aware weight adaptation.

    Adjusts horizon weights based on market regime:
    - Trending: Favor longer horizons
    - Mean-reverting: Favor shorter horizons
    - High volatility: Favor medium horizons
    """

    def __init__(
        self,
        horizons: list[int] = None,
        weights: Optional[dict[int, float]] = None
    ):
        super().__init__(horizons, weights)

        # Regime-specific weights
        self._regime_weights = {
            "trending_up": {1: 0.05, 4: 0.15, 8: 0.2, 12: 0.25, 24: 0.35},
            "trending_down": {1: 0.05, 4: 0.15, 8: 0.2, 12: 0.25, 24: 0.35},
            "mean_reverting": {1: 0.3, 4: 0.3, 8: 0.2, 12: 0.1, 24: 0.1},
            "high_volatility": {1: 0.1, 4: 0.25, 8: 0.3, 12: 0.25, 24: 0.1},
            "low_volatility": {1: 0.15, 4: 0.2, 8: 0.2, 12: 0.2, 24: 0.25},
        }

    def set_regime(self, regime: str):
        """
        Set current market regime.

        Args:
            regime: One of trending_up, trending_down, mean_reverting,
                   high_volatility, low_volatility
        """
        if regime in self._regime_weights:
            # Blend regime weights with adaptive weights
            regime_w = self._regime_weights[regime]

            for h in self.horizons:
                self._adaptive_weights[h] = (
                    0.5 * self._adaptive_weights.get(h, self.weights[h]) +
                    0.5 * regime_w.get(h, self.weights[h])
                )

            # Normalize
            total = sum(self._adaptive_weights.values())
            self._adaptive_weights = {
                h: w / total for h, w in self._adaptive_weights.items()
            }


async def main():
    """Test horizon ensemble."""
    ensemble = AdaptiveHorizonEnsemble()

    # Create mock predictions
    predictions = [
        HorizonPrediction(horizon=1, predicted_return=0.01, confidence=0.6, direction=1),
        HorizonPrediction(horizon=4, predicted_return=0.02, confidence=0.7, direction=1),
        HorizonPrediction(horizon=8, predicted_return=0.015, confidence=0.65, direction=1),
        HorizonPrediction(horizon=12, predicted_return=-0.005, confidence=0.5, direction=-1),
        HorizonPrediction(horizon=24, predicted_return=0.01, confidence=0.6, direction=1),
    ]

    # Combine
    result = ensemble.combine_predictions(predictions)
    print(f"Weighted return: {result.weighted_return:.4f}")
    print(f"Consensus direction: {result.consensus_direction}")
    print(f"Consensus strength: {result.consensus_strength:.2f}")
    print(f"Strong consensus: {result.is_strong_consensus}")


if __name__ == "__main__":
    asyncio.run(main())
