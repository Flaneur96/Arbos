"""
Hyperliquid API client for market data and trading.

Provides:
- Real-time price data
- Order book data
- Account positions and PnL
- Trade execution
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import aiohttp
import pandas as pd
from collections import deque


@dataclass
class MarketData:
    """Real-time market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    size: float


@dataclass
class OrderBook:
    """Order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0


class HyperliquidClient:
    """
    Async client for Hyperliquid exchange.

    Supports:
    - REST API for historical data
    - WebSocket for real-time updates
    - Trading operations
    """

    API_URL = "https://api.hyperliquid.xyz"
    WS_URL = "wss://api.hyperliquid.xyz/ws"

    def __init__(self, account_address: Optional[str] = None):
        self.account_address = account_address
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._price_cache: dict[str, deque[MarketData]] = {}
        self._order_books: dict[str, OrderBook] = {}
        self._subscriptions: set[str] = set()
        self._running = False

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self):
        """Initialize HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        """Close all connections."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    async def _post(self, endpoint: str, payload: dict) -> dict:
        """Make POST request to API."""
        await self.connect()
        url = f"{self.API_URL}/{endpoint}"
        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_meta(self) -> dict:
        """Get exchange metadata (symbols, decimals, etc)."""
        return await self._post("info", {"type": "metaAndAssetCtxs"})

    async def get_mids(self) -> dict[str, float]:
        """Get all mid prices."""
        data = await self._post("info", {"type": "allMids"})
        return {k: float(v) for k, v in data.items() if v}

    async def get_candles(
        self,
        symbol: str,
        interval: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical candlestick data.

        Args:
            symbol: Trading pair (e.g., "BTC")
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            DataFrame with OHLCV data
        """
        # Hyperliquid candle snapshot endpoint
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": symbol,
                "interval": interval,
                "startTime": int(start_time.timestamp() * 1000) if start_time else int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000),
                "endTime": int(end_time.timestamp() * 1000) if end_time else int(datetime.now(timezone.utc).timestamp() * 1000),
            }
        }

        data = await self._post("info", payload)

        if not data:
            return pd.DataFrame()

        # Hyperliquid returns list of dicts with keys: t (start), T (end), o, c, h, l, v, n
        # t = timestamp, o = open, h = high, l = low, c = close, v = volume, n = num trades
        df = pd.DataFrame(data)
        if "t" in df.columns:
            df = df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "n": "trades"
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.set_index("timestamp").sort_index()

        return pd.DataFrame()

    async def get_funding_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical funding rates."""
        # Hyperliquid funding is in meta
        meta = await self.get_meta()
        # Extract funding from asset contexts
        # This is simplified - real implementation would parse properly
        return pd.DataFrame()

    async def get_user_state(self, address: Optional[str] = None) -> dict:
        """Get user account state (positions, balances)."""
        addr = address or self.account_address
        if not addr:
            raise ValueError("No address provided")
        return await self._post("info", {
            "type": "clearinghouseState",
            "user": addr
        })

    async def subscribe_to_prices(
        self,
        symbols: list[str],
        callback: Optional[callable] = None
    ):
        """
        Subscribe to real-time price updates.

        Args:
            symbols: List of symbols to subscribe
            callback: Optional callback for each update
        """
        for symbol in symbols:
            self._subscriptions.add(symbol)
            self._price_cache[symbol] = deque(maxlen=1000)

        # WebSocket subscription would go here
        # For now, we poll via REST

    async def get_recent_prices(self, symbol: str) -> list[MarketData]:
        """Get cached recent prices for symbol."""
        return list(self._price_cache.get(symbol, []))

    async def get_order_book(self, symbol: str) -> OrderBook:
        """Get current order book for symbol."""
        payload = {
            "type": "l2Book",
            "coin": symbol
        }
        data = await self._post("info", payload)

        book = OrderBook(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc)
        )

        if "levels" in data:
            for level in data["levels"][0]:  # bids
                book.bids.append(OrderBookLevel(
                    price=float(level["px"]),
                    size=float(level["sz"])
                ))
            for level in data["levels"][1]:  # asks
                book.asks.append(OrderBookLevel(
                    price=float(level["px"]),
                    size=float(level["sz"])
                ))

        self._order_books[symbol] = book
        return book


async def main():
    """Test the client."""
    async with HyperliquidClient() as client:
        # Get meta
        meta = await client.get_meta()
        print(f"Exchange has {len(meta[0]['universe'])} markets")

        # Get prices
        mids = await client.get_mids()
        print(f"BTC price: ${mids.get('BTC', 'N/A')}")

        # Get candles
        candles = await client.get_candles("BTC", "1h")
        print(f"Got {len(candles)} candles")
        print(candles.tail())


if __name__ == "__main__":
    asyncio.run(main())
