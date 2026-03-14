"""
Market regime detection for trading system.

Detects market regimes:
- Trending (up/down)
- Mean-reverting
- High/low volatility
- Risk-on/risk-off

Uses price action, volatility, and derivatives data.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from collections import deque
from enum import Enum


class MarketRegime(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITIONAL = "transitional"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state."""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime] = None
    confidence: float = 0.5
    trend_strength: float = 0.0
    volatility_level: float = 0.5
    risk_sentiment: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    @property
    def is_trending(self) -> bool:
        return self.primary_regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN)

    @property
    def is_volatile(self) -> bool:
        return self.primary_regime == MarketRegime.HIGH_VOLATILITY


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Trend detection
    trend_lookback: int = 24  # hours
    trend_threshold: float = 0.02  # 2% move = trending
    adx_threshold: float = 25.0

    # Volatility
    vol_lookback: int = 24
    high_vol_percentile: float = 0.75
    low_vol_percentile: float = 0.25

    # Mean reversion
    rsi_extreme_high: float = 70.0
    rsi_extreme_low: float = 30.0
    bb_band_threshold: float = 0.1

    # Risk sentiment
    funding_extreme: float = 0.0005  # 0.05%
    oi_change_threshold: float = 0.1  # 10%


class RegimeDetector:
    """
    Market regime detector.

    Analyzes price action, volatility, and derivatives
    to determine current market regime.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._regime_history: deque[RegimeState] = deque(maxlen=1000)
        self._vol_history: deque[float] = deque(maxlen=1000)
        self._return_history: deque[float] = deque(maxlen=1000)

    def detect(
        self,
        prices,
        features: Optional[dict] = None
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            prices: DataFrame with OHLCV data OR numpy array of close prices
            features: Optional derivatives/features data

        Returns:
            RegimeState with detected regime
        """
        # Handle numpy array input
        if isinstance(prices, np.ndarray):
            if len(prices) < 10:
                return RegimeState(MarketRegime.UNKNOWN)
            # Convert to DataFrame
            prices = pd.DataFrame({
                "close": prices,
                "high": prices * 1.001,  # Approximate
                "low": prices * 0.999,
                "volume": np.ones(len(prices))
            })

        if prices.empty:
            return RegimeState(MarketRegime.UNKNOWN)

        # Calculate metrics
        trend = self._detect_trend(prices)
        volatility = self._detect_volatility(prices)
        mean_reversion = self._detect_mean_reversion(prices)
        risk = self._detect_risk_sentiment(features)

        # Determine primary regime
        primary = self._determine_primary_regime(
            trend, volatility, mean_reversion, risk
        )

        # Determine secondary regime
        secondary = self._determine_secondary_regime(
            trend, volatility, mean_reversion, risk, primary
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            trend, volatility, mean_reversion, risk
        )

        state = RegimeState(
            primary_regime=primary,
            secondary_regime=secondary,
            confidence=confidence,
            trend_strength=trend["strength"],
            volatility_level=volatility["level"],
            risk_sentiment=risk["sentiment"]
        )

        self._regime_history.append(state)
        return state

    def _detect_trend(self, prices: pd.DataFrame) -> dict:
        """Detect trend direction and strength."""
        if len(prices) < self.config.trend_lookback:
            return {"direction": 0, "strength": 0.0}

        close = prices["close"]

        # Price change over lookback
        start_price = close.iloc[-self.config.trend_lookback]
        end_price = close.iloc[-1]
        change = (end_price - start_price) / start_price

        # Direction
        if change > self.config.trend_threshold:
            direction = 1
        elif change < -self.config.trend_threshold:
            direction = -1
        else:
            direction = 0

        # Strength (absolute change)
        strength = min(abs(change) / self.config.trend_threshold, 2.0)

        # Check for higher highs / lower lows
        if len(prices) >= 48:
            highs = prices["high"].iloc[-48:]
            lows = prices["low"].iloc[-48:]

            higher_highs = sum(
                1 for i in range(1, len(highs))
                if highs.iloc[i] > highs.iloc[i-1]
            )
            lower_lows = sum(
                1 for i in range(1, len(lows))
                if lows.iloc[i] < lows.iloc[i-1]
            )

            # Adjust strength based on structure
            if direction > 0 and higher_highs > 24:
                strength = min(strength * 1.2, 2.0)
            elif direction < 0 and lower_lows > 24:
                strength = min(strength * 1.2, 2.0)

        return {"direction": direction, "strength": strength, "change": change}

    def _detect_volatility(self, prices: pd.DataFrame) -> dict:
        """Detect volatility level."""
        if len(prices) < self.config.vol_lookback:
            return {"level": 0.5, "regime": "normal"}

        # Calculate returns
        returns = prices["close"].pct_change().dropna()

        if len(returns) < 10:
            return {"level": 0.5, "regime": "normal"}

        # Realized volatility
        vol = returns.std() * np.sqrt(365 * 24)  # Annualized (hourly)
        self._vol_history.append(vol)

        # Compare to history
        if len(self._vol_history) >= 100:
            vol_percentile = sum(
                1 for v in self._vol_history if v < vol
            ) / len(self._vol_history)
        else:
            vol_percentile = 0.5

        # Determine regime
        if vol_percentile > self.config.high_vol_percentile:
            regime = "high"
        elif vol_percentile < self.config.low_vol_percentile:
            regime = "low"
        else:
            regime = "normal"

        return {
            "level": vol_percentile,
            "regime": regime,
            "value": vol
        }

    def _detect_mean_reversion(self, prices: pd.DataFrame) -> dict:
        """Detect mean reversion signals."""
        if len(prices) < 24:
            return {"signal": 0, "strength": 0.0}

        # RSI
        delta = prices["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Mean reversion signal
        if current_rsi > self.config.rsi_extreme_high:
            signal = -1  # Overbought - expect down
            strength = (current_rsi - self.config.rsi_extreme_high) / (100 - self.config.rsi_extreme_high)
        elif current_rsi < self.config.rsi_extreme_low:
            signal = 1  # Oversold - expect up
            strength = (self.config.rsi_extreme_low - current_rsi) / self.config.rsi_extreme_low
        else:
            signal = 0
            strength = 0.0

        # Bollinger Band position
        sma = prices["close"].rolling(20).mean()
        std = prices["close"].rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        current_price = prices["close"].iloc[-1]
        bb_position = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

        # Adjust signal based on BB
        if bb_position > 0.9:
            signal = max(signal, -1)
            strength = max(strength, 0.5)
        elif bb_position < 0.1:
            signal = max(signal, 1)
            strength = max(strength, 0.5)

        return {
            "signal": signal,
            "strength": strength,
            "rsi": current_rsi,
            "bb_position": bb_position
        }

    def _detect_risk_sentiment(self, features: Optional[dict]) -> dict:
        """Detect risk sentiment from derivatives."""
        if not features:
            return {"sentiment": 0.0, "regime": "neutral"}

        sentiment = 0.0

        # Funding rate
        funding = features.get("funding_rate", 0)
        if abs(funding) > self.config.funding_extreme:
            # Contrarian: high funding = crowded, expect reversal
            sentiment -= np.sign(funding) * min(abs(funding) / self.config.funding_extreme, 1.0)

        # OI changes
        oi_change = features.get("oi_change_24h", 0)
        if abs(oi_change) > self.config.oi_change_threshold:
            # Rising OI with price = strong trend
            sentiment += np.sign(oi_change) * 0.3

        # Long/short ratio
        ls_ratio = features.get("ls_ratio", 0.5)
        if ls_ratio > 0.6:
            # Too many longs = risk off
            sentiment -= 0.2
        elif ls_ratio < 0.4:
            # Too many shorts = potential squeeze
            sentiment += 0.2

        # Determine regime
        if sentiment > 0.3:
            regime = "risk_on"
        elif sentiment < -0.3:
            regime = "risk_off"
        else:
            regime = "neutral"

        return {"sentiment": sentiment, "regime": regime}

    def _determine_primary_regime(
        self,
        trend: dict,
        volatility: dict,
        mean_reversion: dict,
        risk: dict
    ) -> MarketRegime:
        """Determine primary regime from signals."""
        # High volatility overrides everything
        if volatility["regime"] == "high":
            return MarketRegime.HIGH_VOLATILITY

        # Low volatility
        if volatility["regime"] == "low":
            return MarketRegime.LOW_VOLATILITY

        # Strong trend
        if trend["strength"] > 1.0:
            if trend["direction"] > 0:
                return MarketRegime.TRENDING_UP
            elif trend["direction"] < 0:
                return MarketRegime.TRENDING_DOWN

        # Mean reversion signals
        if mean_reversion["strength"] > 0.5:
            return MarketRegime.MEAN_REVERTING

        # Default based on trend
        if trend["direction"] > 0:
            return MarketRegime.TRENDING_UP
        elif trend["direction"] < 0:
            return MarketRegime.TRENDING_DOWN

        return MarketRegime.TRANSITIONAL

    def _determine_secondary_regime(
        self,
        trend: dict,
        volatility: dict,
        mean_reversion: dict,
        risk: dict,
        primary: MarketRegime
    ) -> Optional[MarketRegime]:
        """Determine secondary regime."""
        # Risk sentiment as secondary
        if risk["regime"] == "risk_on":
            return MarketRegime.RISK_ON
        elif risk["regime"] == "risk_off":
            return MarketRegime.RISK_OFF

        # Volatility as secondary if not primary
        if primary != MarketRegime.HIGH_VOLATILITY and volatility["regime"] == "high":
            return MarketRegime.HIGH_VOLATILITY

        return None

    def _calculate_confidence(
        self,
        trend: dict,
        volatility: dict,
        mean_reversion: dict,
        risk: dict
    ) -> float:
        """Calculate confidence in regime detection."""
        confidence = 0.5

        # Strong trend increases confidence
        if trend["strength"] > 1.0:
            confidence += 0.2

        # Clear volatility regime
        if volatility["level"] > 0.8 or volatility["level"] < 0.2:
            confidence += 0.1

        # Mean reversion signals
        if mean_reversion["strength"] > 0.5:
            confidence += 0.1

        # Risk sentiment clarity
        if abs(risk["sentiment"]) > 0.5:
            confidence += 0.1

        return min(confidence, 1.0)

    def get_regime_history(self, n: int = 100) -> list[RegimeState]:
        """Get last n regime states."""
        return list(self._regime_history)[-n:]


def main():
    """Test regime detection."""
    # Generate test data
    np.random.seed(42)
    n = 200

    # Trending up with noise
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 1, n)
    prices = trend + noise

    df = pd.DataFrame({
        "close": prices,
        "high": prices + np.abs(np.random.normal(0.5, 0.2, n)),
        "low": prices - np.abs(np.random.normal(0.5, 0.2, n)),
        "volume": np.random.uniform(100, 1000, n)
    })

    detector = RegimeDetector()
    state = detector.detect(df)

    print(f"Primary regime: {state.primary_regime.value}")
    print(f"Secondary regime: {state.secondary_regime.value if state.secondary_regime else 'None'}")
    print(f"Confidence: {state.confidence:.2f}")
    print(f"Trend strength: {state.trend_strength:.2f}")
    print(f"Volatility level: {state.volatility_level:.2f}")


if __name__ == "__main__":
    main()
