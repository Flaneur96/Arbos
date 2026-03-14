"""
Consensus gating for trading signals.

Filters signals based on agreement across multiple sources:
- Multiple models
- Multiple horizons
- Multiple features
- Regime awareness
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable
import numpy as np
from collections import deque


@dataclass
class GatingConfig:
    """Configuration for consensus gating."""
    # Minimum agreement thresholds
    min_model_agreement: float = 0.6
    min_horizon_agreement: float = 0.7
    min_confidence: float = 0.5

    # Weight configuration
    model_weight: float = 0.4
    horizon_weight: float = 0.3
    feature_weight: float = 0.2
    regime_weight: float = 0.1

    # Regime adjustments
    trending_threshold: float = 0.02
    volatile_threshold: float = 0.04

    # History
    signal_history_size: int = 1000


@dataclass
class GatedSignal:
    """Signal that passed consensus gating."""
    symbol: str
    direction: int
    strength: float
    model_agreement: float
    horizon_agreement: float
    feature_agreement: float
    regime_alignment: float
    confidence: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    @property
    def overall_agreement(self) -> float:
        """Weighted overall agreement score."""
        return (
            self.model_agreement * 0.4 +
            self.horizon_agreement * 0.3 +
            self.feature_agreement * 0.2 +
            self.regime_alignment * 0.1
        )


class ConsensusGating:
    """
    Consensus gating mechanism for signal filtering.

    Aggregates signals from multiple sources and only
    passes signals with sufficient agreement.
    """

    def __init__(self, config: Optional[GatingConfig] = None):
        self.config = config or GatingConfig()
        self._signal_history: deque[GatedSignal] = deque(maxlen=self.config.signal_history_size)

        # Track recent signals for regime detection
        self._recent_directions: dict[str, deque] = {}

    def gate(
        self,
        predictions: list[dict],
        regime: Optional[str] = None,
        feature_signals: Optional[dict] = None
    ) -> Optional[GatedSignal]:
        """
        Apply consensus gating to predictions.

        Args:
            predictions: List of model predictions
                Each prediction has: model, direction, confidence, horizon
            regime: Current market regime
            feature_signals: Feature-based signals

        Returns:
            GatedSignal if passes gating, None otherwise
        """
        if not predictions:
            return None

        # Extract symbol (assuming all predictions for same symbol)
        symbol = predictions[0].get("symbol", "UNKNOWN")

        # Calculate model agreement
        model_agreement = self._calculate_model_agreement(predictions)

        # Calculate horizon agreement
        horizon_agreement = self._calculate_horizon_agreement(predictions)

        # Calculate feature agreement
        feature_agreement = self._calculate_feature_agreement(feature_signals)

        # Calculate regime alignment
        regime_alignment = self._calculate_regime_alignment(predictions, regime)

        # Check thresholds
        if model_agreement < self.config.min_model_agreement:
            return None

        if horizon_agreement < self.config.min_horizon_agreement:
            return None

        # Calculate overall direction and strength
        direction = self._calculate_direction(predictions)
        strength = self._calculate_strength(predictions)

        # Calculate confidence
        confidence = np.mean([p.get("confidence", 0.5) for p in predictions])

        if confidence < self.config.min_confidence:
            return None

        # Adjust strength by agreements
        adjusted_strength = strength * model_agreement * horizon_agreement

        # Create gated signal
        signal = GatedSignal(
            symbol=symbol,
            direction=direction,
            strength=adjusted_strength,
            model_agreement=model_agreement,
            horizon_agreement=horizon_agreement,
            feature_agreement=feature_agreement,
            regime_alignment=regime_alignment,
            confidence=confidence
        )

        self._signal_history.append(signal)

        # Track for regime detection
        if symbol not in self._recent_directions:
            self._recent_directions[symbol] = deque(maxlen=100)
        self._recent_directions[symbol].append(direction)

        return signal

    def _calculate_model_agreement(self, predictions: list[dict]) -> float:
        """Calculate agreement across models."""
        if not predictions:
            return 0.0

        directions = [p.get("direction", 0) for p in predictions]
        weights = [p.get("confidence", 1.0) for p in predictions]

        # Weighted agreement
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        # Calculate weighted direction sum
        weighted_direction = sum(d * w for d, w in zip(directions, weights))

        # Agreement is how close to unanimous
        agreement = abs(weighted_direction) / total_weight

        return agreement

    def _calculate_horizon_agreement(self, predictions: list[dict]) -> float:
        """Calculate agreement across prediction horizons."""
        if not predictions:
            return 0.0

        # Group by horizon
        horizon_votes = {}
        for p in predictions:
            h = p.get("horizon", 1)
            d = p.get("direction", 0)

            if h not in horizon_votes:
                horizon_votes[h] = {"up": 0, "down": 0, "neutral": 0}

            if d > 0:
                horizon_votes[h]["up"] += 1
            elif d < 0:
                horizon_votes[h]["down"] += 1
            else:
                horizon_votes[h]["neutral"] += 1

        # Check if horizons agree on direction
        if not horizon_votes:
            return 0.0

        up_horizons = sum(1 for h in horizon_votes.values() if h["up"] > h["down"])
        down_horizons = sum(1 for h in horizon_votes.values() if h["down"] > h["up"])

        total_horizons = len(horizon_votes)
        agreement = max(up_horizons, down_horizons) / total_horizons

        return agreement

    def _calculate_feature_agreement(
        self,
        feature_signals: Optional[dict]
    ) -> float:
        """Calculate agreement from feature signals."""
        if not feature_signals:
            return 0.5  # Neutral if no feature signals

        signals = list(feature_signals.values())

        if not signals:
            return 0.5

        # Simple majority agreement
        up = sum(1 for s in signals if s > 0)
        down = sum(1 for s in signals if s < 0)
        neutral = sum(1 for s in signals if s == 0)

        total = len(signals)
        agreement = max(up, down) / total if total > 0 else 0.5

        return agreement

    def _calculate_regime_alignment(
        self,
        predictions: list[dict],
        regime: Optional[str]
    ) -> float:
        """Calculate alignment with current regime."""
        if regime is None:
            return 0.5  # Neutral

        # Get predicted direction
        directions = [p.get("direction", 0) for p in predictions]
        if not directions:
            return 0.5

        predicted_direction = np.sign(sum(directions))

        # Regime expectations
        regime_directions = {
            "trending_up": 1,
            "trending_down": -1,
            "mean_reverting": 0,
            "high_volatility": 0,
            "low_volatility": 0,
        }

        expected = regime_directions.get(regime, 0)

        # Alignment score
        if expected == 0:
            # Mean-reverting or volatile - no strong direction expected
            return 0.5
        elif predicted_direction == expected:
            return 1.0
        elif predicted_direction == 0:
            return 0.5
        else:
            return 0.0  # Opposite direction

    def _calculate_direction(self, predictions: list[dict]) -> int:
        """Calculate consensus direction."""
        directions = [p.get("direction", 0) for p in predictions]
        weights = [p.get("confidence", 1.0) for p in predictions]

        weighted_sum = sum(d * w for d, w in zip(directions, weights))

        if weighted_sum > 0.5:
            return 1
        elif weighted_sum < -0.5:
            return -1
        return 0

    def _calculate_strength(self, predictions: list[dict]) -> float:
        """Calculate signal strength."""
        if not predictions:
            return 0.0

        # Use average confidence as base strength
        confidences = [p.get("confidence", 0.5) for p in predictions]
        return np.mean(confidences)

    def get_signal_stats(self) -> dict:
        """Get statistics on gated signals."""
        if not self._signal_history:
            return {"total": 0}

        signals = list(self._signal_history)

        return {
            "total": len(signals),
            "avg_strength": np.mean([s.strength for s in signals]),
            "avg_model_agreement": np.mean([s.model_agreement for s in signals]),
            "avg_horizon_agreement": np.mean([s.horizon_agreement for s in signals]),
            "direction_balance": sum(s.direction for s in signals) / len(signals),
            "passed_count": len([s for s in signals if s.direction != 0])
        }


class MultiSourceGating:
    """
    Multi-source consensus gating.

    Aggregates signals from multiple independent sources.
    """

    def __init__(self, config: Optional[GatingConfig] = None):
        self.config = config or GatingConfig()
        self._sources: dict[str, Callable] = {}

    def add_source(self, name: str, signal_func: Callable):
        """Add a signal source."""
        self._sources[name] = signal_func

    def remove_source(self, name: str):
        """Remove a signal source."""
        self._sources.pop(name, None)

    def aggregate(self, symbol: str, **kwargs) -> Optional[GatedSignal]:
        """
        Aggregate signals from all sources.

        Args:
            symbol: Trading symbol
            **kwargs: Additional arguments for signal functions

        Returns:
            GatedSignal if consensus reached
        """
        predictions = []

        for source_name, signal_func in self._sources.items():
            try:
                signal = signal_func(symbol, **kwargs)
                if signal:
                    signal["source"] = source_name
                    predictions.append(signal)
            except Exception as e:
                print(f"Error from source {source_name}: {e}")

        if not predictions:
            return None

        # Apply gating
        gating = ConsensusGating(self.config)
        return gating.gate(predictions)

    def get_source_count(self) -> int:
        """Get number of registered sources."""
        return len(self._sources)


# Feature-based signal generators for gating
def momentum_signal(features: dict) -> float:
    """Generate momentum-based signal from features."""
    # Short-term momentum
    mom_24h = features.get("momentum_24h", 0)
    mom_168h = features.get("momentum_168h", 0)

    # Combined momentum signal
    signal = mom_24h * 0.6 + mom_168h * 0.4

    return signal


def volatility_signal(features: dict) -> float:
    """Generate volatility-adjusted signal."""
    volatility = features.get("volatility_24h", 0.5)
    atr_ratio = features.get("atr_ratio", 0.02)

    # Inverse volatility (prefer low volatility)
    if volatility > 0:
        vol_signal = -np.log(volatility + 0.01)
    else:
        vol_signal = 0

    # ATR contribution
    atr_signal = -atr_ratio * 10 if atr_ratio > 0.03 else 0

    return vol_signal + atr_signal


def derivatives_signal(features: dict) -> float:
    """Generate derivatives-based signal."""
    funding = features.get("funding_rate", 0)
    oi_change = features.get("oi_change_24h", 0)
    ls_ratio = features.get("ls_ratio", 0.5)

    signal = 0

    # Contrarian funding
    if funding > 0.0005:  # High positive funding
        signal -= 0.5  # Expect mean reversion down
    elif funding < -0.0005:  # High negative funding
        signal += 0.5  # Expect mean reversion up

    # OI changes
    signal += np.sign(oi_change) * min(abs(oi_change), 0.3)

    # Long/short imbalance
    if ls_ratio > 0.6:
        signal -= 0.2  # Too many longs
    elif ls_ratio < 0.4:
        signal += 0.2  # Too many shorts

    return signal


def main():
    """Test consensus gating."""
    # Create mock predictions
    predictions = [
        {"model": "chronos-small", "direction": 1, "confidence": 0.7, "horizon": 1},
        {"model": "chronos-base", "direction": 1, "confidence": 0.75, "horizon": 4},
        {"model": "timesfm", "direction": 1, "confidence": 0.65, "horizon": 8},
        {"model": "chronos-small", "direction": -1, "confidence": 0.5, "horizon": 12},
        {"model": "chronos-base", "direction": 1, "confidence": 0.6, "horizon": 24},
    ]

    gating = ConsensusGating()
    signal = gating.gate(predictions, regime="trending_up")

    if signal:
        print(f"Signal: direction={signal.direction}, strength={signal.strength:.2f}")
        print(f"Model agreement: {signal.model_agreement:.2f}")
        print(f"Horizon agreement: {signal.horizon_agreement:.2f}")
        print(f"Overall agreement: {signal.overall_agreement:.2f}")
    else:
        print("Signal rejected by gating")


if __name__ == "__main__":
    main()
