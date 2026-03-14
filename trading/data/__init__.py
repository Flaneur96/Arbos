"""
Data pipeline for trading system.

Sources:
- Hyperliquid: capital and market data
- Coinglass: derivatives data (funding, OI, liquidations, leverage)
"""

from .hyperliquid import HyperliquidClient
from .coinglass import CoinglassClient
from .feature_store import FeatureStore

__all__ = ["HyperliquidClient", "CoinglassClient", "FeatureStore"]
