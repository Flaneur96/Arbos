"""
Coinglass API client for derivatives data.

Provides:
- Funding rates across exchanges
- Open interest data
- Liquidation data
- Long/short ratios
- Leverage statistics
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import aiohttp
import pandas as pd
from collections import deque


@dataclass
class FundingData:
    """Funding rate data point."""
    symbol: str
    exchange: str
    timestamp: datetime
    funding_rate: float
    next_funding_time: Optional[datetime] = None


@dataclass
class OpenInterestData:
    """Open interest data point."""
    symbol: str
    timestamp: datetime
    open_interest: float  # in USD
    change_1h: float = 0.0
    change_4h: float = 0.0
    change_24h: float = 0.0


@dataclass
class LiquidationData:
    """Liquidation event."""
    symbol: str
    exchange: str
    timestamp: datetime
    side: str  # "long" or "short"
    price: float
    quantity: float
    value: float  # in USD


@dataclass
class LeverageStats:
    """Leverage statistics."""
    symbol: str
    timestamp: datetime
    long_leverage_avg: float
    short_leverage_avg: float
    long_ratio: float  # % of long positions
    short_ratio: float  # % of short positions


class CoinglassClient:
    """
    Async client for Coinglass API.

    Derivatives data aggregator for:
    - Funding rates (BTC, ETH, etc)
    - Open interest
    - Liquidations
    - Leverage statistics
    """

    BASE_URL = "https://open-api.coinglass.com"

    # Common symbols
    SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "DOT", "MATIC"]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._funding_cache: dict[str, deque[FundingData]] = {}
        self._oi_cache: dict[str, deque[OpenInterestData]] = {}
        self._liq_cache: dict[str, deque[LiquidationData]] = {}

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self):
        """Initialize HTTP session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-CoinGlass-Api-Key"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)

    async def close(self):
        """Close connection."""
        if self._session:
            await self._session.close()

    async def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request."""
        await self.connect()
        url = f"{self.BASE_URL}{endpoint}"
        async with self._session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            elif resp.status == 401:
                # No API key or invalid - return empty data
                return {"success": False, "data": []}
            else:
                return {"success": False, "data": []}

    async def get_funding_rates(
        self,
        symbol: Optional[str] = None,
        exchange: Optional[str] = None
    ) -> list[FundingData]:
        """
        Get current funding rates across exchanges.

        Args:
            symbol: Filter by symbol (e.g., "BTC")
            exchange: Filter by exchange (e.g., "Binance")

        Returns:
            List of FundingData objects
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._get("/api/fundingRate/v2/home", params)

        results = []
        if data.get("success") and data.get("data"):
            for item in data["data"]:
                for rate_data in item.get("uMarginList", []):
                    funding = FundingData(
                        symbol=item.get("symbol", ""),
                        exchange=rate_data.get("exchangeName", ""),
                        timestamp=datetime.now(timezone.utc),
                        funding_rate=float(rate_data.get("rate", 0)) / 100,  # Convert from %
                    )
                    results.append(funding)

                    # Cache by symbol
                    sym = funding.symbol
                    if sym not in self._funding_cache:
                        self._funding_cache[sym] = deque(maxlen=1000)
                    self._funding_cache[sym].append(funding)

        return results

    async def get_funding_history(
        self,
        symbol: str,
        exchange: str = "Binance",
        days: int = 7
    ) -> pd.DataFrame:
        """
        Get historical funding rates.

        Args:
            symbol: Crypto symbol
            exchange: Exchange name
            days: Number of days of history

        Returns:
            DataFrame with timestamp and funding_rate
        """
        params = {
            "symbol": symbol,
            "exchange": exchange,
        }

        data = await self._get("/api/fundingRate/chart", params)

        if not data.get("success"):
            return pd.DataFrame()

        history = data.get("data", {}).get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df.columns = ["timestamp", "funding_rate"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["funding_rate"] = df["funding_rate"].astype(float) / 100
        df = df.set_index("timestamp").sort_index()

        return df

    async def get_open_interest(
        self,
        symbol: Optional[str] = None
    ) -> list[OpenInterestData]:
        """
        Get current open interest data.

        Args:
            symbol: Filter by symbol

        Returns:
            List of OpenInterestData objects
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._get("/api/openInterest", params)

        results = []
        if data.get("success") and data.get("data"):
            for item in data["data"]:
                oi = OpenInterestData(
                    symbol=item.get("symbol", ""),
                    timestamp=datetime.now(timezone.utc),
                    open_interest=float(item.get("openInterest", 0)),
                    change_1h=float(item.get("change1h", 0)),
                    change_4h=float(item.get("change4h", 0)),
                    change_24h=float(item.get("change24h", 0)),
                )
                results.append(oi)

                # Cache
                sym = oi.symbol
                if sym not in self._oi_cache:
                    self._oi_cache[sym] = deque(maxlen=1000)
                self._oi_cache[sym].append(oi)

        return results

    async def get_oi_history(
        self,
        symbol: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get historical open interest.

        Args:
            symbol: Crypto symbol
            days: Days of history

        Returns:
            DataFrame with timestamp and open_interest
        """
        params = {"symbol": symbol}
        data = await self._get("/api/openInterest/chart", params)

        if not data.get("success"):
            return pd.DataFrame()

        history = data.get("data", {}).get("history", [])
        if not history:
            return pd.DataFrame()

        df = pd.DataFrame(history)
        df.columns = ["timestamp", "open_interest"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = df["open_interest"].astype(float)
        df = df.set_index("timestamp").sort_index()

        return df

    async def get_liquidations(
        self,
        symbol: Optional[str] = None,
        hours: int = 24
    ) -> list[LiquidationData]:
        """
        Get recent liquidations.

        Args:
            symbol: Filter by symbol
            hours: Hours of history

        Returns:
            List of LiquidationData objects
        """
        # Coinglass provides liquidation chart data
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._get("/api/liquidation-chart", params)

        results = []
        if data.get("success") and data.get("data"):
            for item in data.get("data", []):
                liq = LiquidationData(
                    symbol=item.get("symbol", ""),
                    exchange=item.get("exchange", ""),
                    timestamp=datetime.now(timezone.utc),
                    side=item.get("side", "unknown"),
                    price=float(item.get("price", 0)),
                    quantity=float(item.get("qty", 0)),
                    value=float(item.get("value", 0)),
                )
                results.append(liq)

        return results

    async def get_long_short_ratio(
        self,
        symbol: str = "BTC",
        exchange: str = "Binance"
    ) -> dict:
        """
        Get long/short ratio for a symbol.

        Args:
            symbol: Crypto symbol
            exchange: Exchange name

        Returns:
            Dict with long_ratio, short_ratio, timestamp
        """
        params = {"symbol": symbol, "exchange": exchange}
        data = await self._get("/api/longShortRatio", params)

        if not data.get("success"):
            return {"long_ratio": 0.5, "short_ratio": 0.5}

        ratio_data = data.get("data", {})
        return {
            "long_ratio": float(ratio_data.get("longRatio", 0.5)),
            "short_ratio": float(ratio_data.get("shortRatio", 0.5)),
            "timestamp": datetime.now(timezone.utc),
        }

    async def get_leverage_stats(self, symbol: str) -> Optional[LeverageStats]:
        """
        Get leverage statistics for a symbol.

        Args:
            symbol: Crypto symbol

        Returns:
            LeverageStats or None
        """
        data = await self._get("/api/leverage", {"symbol": symbol})

        if not data.get("success"):
            return None

        stats = data.get("data", {})
        return LeverageStats(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            long_leverage_avg=float(stats.get("longLeverage", 0)),
            short_leverage_avg=float(stats.get("shortLeverage", 0)),
            long_ratio=float(stats.get("longRatio", 0.5)),
            short_ratio=float(stats.get("shortRatio", 0.5)),
        )

    async def collect_all_derivatives_data(self) -> dict:
        """
        Collect all derivatives data for analysis.

        Returns:
            Dict with funding, OI, liquidations, ratios
        """
        funding = await self.get_funding_rates()
        oi = await self.get_open_interest()

        return {
            "funding_rates": {f.symbol: f.funding_rate for f in funding},
            "open_interest": {o.symbol: o.open_interest for o in oi},
            "timestamp": datetime.now(timezone.utc),
        }


async def main():
    """Test the client."""
    async with CoinglassClient() as client:
        # Get funding rates
        funding = await client.get_funding_rates()
        print(f"Got {len(funding)} funding rate entries")

        # Get open interest
        oi = await client.get_open_interest()
        print(f"Got {len(oi)} OI entries")

        # Get all data
        all_data = await client.collect_all_derivatives_data()
        print(f"Data timestamp: {all_data['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
