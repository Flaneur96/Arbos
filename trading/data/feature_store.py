"""
Feature store for trading system.

Aggregates and stores features from multiple data sources:
- Price/volume data from Hyperliquid
- Derivatives data from Coinglass
- Derived technical indicators
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path
import json

from .hyperliquid import HyperliquidClient, MarketData
from .coinglass import CoinglassClient, FundingData, OpenInterestData


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    # Price features
    return_windows: list[int] = field(default_factory=lambda: [1, 4, 8, 12, 24, 72])  # hours
    volatility_windows: list[int] = field(default_factory=lambda: [12, 24, 72, 168])
    volume_windows: list[int] = field(default_factory=lambda: [1, 4, 24])

    # Technical indicators
    ema_periods: list[int] = field(default_factory=lambda: [8, 21, 55, 144])
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Derivatives features
    funding_windows: list[int] = field(default_factory=lambda: [1, 8, 24])  # hours
    oi_windows: list[int] = field(default_factory=lambda: [1, 4, 24])


class FeatureStore:
    """
    Central feature store for trading system.

    Collects data from multiple sources and generates features:
    - Price-based: returns, volatility, momentum
    - Volume-based: volume ratios, relative volume
    - Derivatives: funding rates, OI changes
    - Cross-asset: correlations, market regime
    """

    def __init__(
        self,
        hyperliquid: HyperliquidClient,
        coinglass: CoinglassClient,
        config: Optional[FeatureConfig] = None,
        storage_path: Optional[Path] = None
    ):
        self.hyperliquid = hyperliquid
        self.coinglass = coinglass
        self.config = config or FeatureConfig()
        self.storage_path = storage_path or Path("data/features")

        # In-memory caches
        self._price_cache: dict[str, pd.DataFrame] = {}
        self._feature_cache: dict[str, pd.DataFrame] = {}
        self._last_update: dict[str, datetime] = {}

        # Symbols to track
        self.symbols: list[str] = []

    async def initialize(self, symbols: list[str]):
        """
        Initialize feature store with symbols.

        Args:
            symbols: List of trading pairs to track
        """
        self.symbols = symbols
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Pre-load historical data
        for symbol in symbols:
            await self._load_historical_data(symbol)

    async def _load_historical_data(self, symbol: str):
        """Load historical price data for a symbol."""
        try:
            # Get candles from different timeframes
            candles_1h = await self.hyperliquid.get_candles(
                symbol,
                interval="1h",
                start_time=datetime.now(timezone.utc) - timedelta(days=30)
            )

            if not candles_1h.empty:
                self._price_cache[symbol] = candles_1h
                self._last_update[symbol] = candles_1h.index[-1]

        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")

    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional feature columns
        """
        features = df.copy()

        # Returns at different horizons
        for window in self.config.return_windows:
            features[f"return_{window}h"] = features["close"].pct_change(window)

        # Log returns
        features["log_return"] = np.log(features["close"] / features["close"].shift(1))

        # Realized volatility
        for window in self.config.volatility_windows:
            features[f"volatility_{window}h"] = (
                features["log_return"]
                .rolling(window)
                .std()
                * np.sqrt(365 * 24)  # Annualize (hourly data)
            )

        # Volume features
        for window in self.config.volume_windows:
            features[f"volume_ratio_{window}h"] = (
                features["volume"] / features["volume"].rolling(window).mean()
            )

        # EMA
        for period in self.config.ema_periods:
            features[f"ema_{period}"] = features["close"].ewm(span=period).mean()
            features[f"price_to_ema_{period}"] = features["close"] / features[f"ema_{period}"]

        # RSI
        delta = features["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(self.config.rsi_period).mean()
        avg_loss = loss.rolling(self.config.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        features["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        high_low = features["high"] - features["low"]
        high_close = abs(features["high"] - features["close"].shift(1))
        low_close = abs(features["low"] - features["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features["atr"] = tr.rolling(self.config.atr_period).mean()
        features["atr_ratio"] = features["atr"] / features["close"]

        # MACD
        ema_fast = features["close"].ewm(span=self.config.macd_fast).mean()
        ema_slow = features["close"].ewm(span=self.config.macd_slow).mean()
        features["macd"] = ema_fast - ema_slow
        features["macd_signal"] = features["macd"].ewm(span=self.config.macd_signal).mean()
        features["macd_hist"] = features["macd"] - features["macd_signal"]

        # Momentum
        features["momentum_24h"] = features["close"] / features["close"].shift(24) - 1
        features["momentum_168h"] = features["close"] / features["close"].shift(168) - 1

        # Higher timeframe trend
        features["higher_high"] = (
            (features["high"] > features["high"].shift(1)) &
            (features["high"].shift(1) > features["high"].shift(2))
        ).astype(int)
        features["lower_low"] = (
            (features["low"] < features["low"].shift(1)) &
            (features["low"].shift(1) < features["low"].shift(2))
        ).astype(int)

        return features

    async def compute_derivatives_features(
        self,
        symbol: str,
        funding_history: Optional[pd.DataFrame] = None,
        oi_history: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        Compute derivatives-based features.

        Args:
            symbol: Trading pair
            funding_history: Historical funding rates
            oi_history: Historical open interest

        Returns:
            Dict of derivative features
        """
        features = {}

        try:
            # Get current funding
            funding_data = await self.coinglass.get_funding_rates(symbol=symbol)
            if funding_data:
                # Average funding across exchanges
                avg_funding = np.mean([f.funding_rate for f in funding_data])
                features["funding_rate"] = avg_funding

                # Extreme funding detection
                features["funding_extreme_long"] = avg_funding > 0.05 / 100  # > 0.05%
                features["funding_extreme_short"] = avg_funding < -0.05 / 100  # < -0.05%

            # Get current OI
            oi_data = await self.coinglass.get_open_interest(symbol=symbol)
            if oi_data:
                oi = oi_data[0]
                features["oi_usd"] = oi.open_interest
                features["oi_change_1h"] = oi.change_1h
                features["oi_change_24h"] = oi.change_24h

            # Long/short ratio
            ratio = await self.coinglass.get_long_short_ratio(symbol)
            features["ls_ratio"] = ratio["long_ratio"]
            features["ls_imbalance"] = abs(ratio["long_ratio"] - 0.5) * 2

        except Exception as e:
            print(f"Error getting derivatives features for {symbol}: {e}")

        return features

    async def update_features(self, symbol: str) -> pd.DataFrame:
        """
        Update features for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            DataFrame with updated features
        """
        # Get latest price data
        candles = await self.hyperliquid.get_candles(
            symbol,
            interval="1h",
            start_time=datetime.now(timezone.utc) - timedelta(hours=100)
        )

        if candles.empty:
            return pd.DataFrame()

        # Merge with cached data
        if symbol in self._price_cache:
            combined = pd.concat([self._price_cache[symbol], candles])
            combined = combined[~combined.index.duplicated(keep="last")]
            self._price_cache[symbol] = combined.sort_index()
        else:
            self._price_cache[symbol] = candles

        # Compute features
        features = self.compute_price_features(self._price_cache[symbol])

        # Add derivatives features (latest row only)
        deriv_features = await self.compute_derivatives_features(symbol)
        for key, value in deriv_features.items():
            features[key] = value

        self._feature_cache[symbol] = features
        self._last_update[symbol] = datetime.now(timezone.utc)

        return features

    async def get_features(self, symbol: str) -> pd.DataFrame:
        """Get current features for symbol."""
        if symbol not in self._feature_cache:
            await self.update_features(symbol)
        return self._feature_cache.get(symbol, pd.DataFrame())

    async def get_latest_features(self, symbol: str) -> dict:
        """Get latest feature values as dict."""
        df = await self.get_features(symbol)
        if df.empty:
            return {}
        return df.iloc[-1].to_dict()

    def save_to_disk(self, symbol: str):
        """Save features to disk."""
        if symbol in self._feature_cache:
            path = self.storage_path / f"{symbol}_features.parquet"
            self._feature_cache[symbol].to_parquet(path)

    def load_from_disk(self, symbol: str) -> pd.DataFrame:
        """Load features from disk."""
        path = self.storage_path / f"{symbol}_features.parquet"
        if path.exists():
            self._feature_cache[symbol] = pd.read_parquet(path)
            return self._feature_cache[symbol]
        return pd.DataFrame()


async def main():
    """Test feature store."""
    from .hyperliquid import HyperliquidClient
    from .coinglass import CoinglassClient

    async with HyperliquidClient() as hl, CoinglassClient() as cg:
        store = FeatureStore(hl, cg)

        # Initialize with BTC
        await store.initialize(["BTC"])

        # Get features
        features = await store.get_features("BTC")
        print(f"Generated {len(features.columns)} features")
        print(features.tail())


if __name__ == "__main__":
    asyncio.run(main())
