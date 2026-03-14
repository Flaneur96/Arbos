"""
LLM-based time series forecaster using OpenAI-compatible APIs.

Supports:
- NVIDIA NIM API
- Lium GPU marketplace
- Any OpenAI-compatible endpoint
"""

import asyncio
import aiohttp
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class LLMForecast:
    """Forecast from LLM with reasoning."""
    model_name: str
    horizon: int
    predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    reasoning: str
    direction: int  # -1, 0, 1
    confidence: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OpenAICompatibleClient:
    """
    Generic OpenAI-compatible API client.

    Works with NVIDIA NIM, Lium, Chutes, and any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        provider_name: str = "openai-compatible"
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_name = provider_name

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Generate completion."""
        url = f"{self.base_url}/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"{self.provider_name} API error: {resp.status} - {error}")

                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    async def list_models(self) -> list[str]:
        """List available models."""
        url = f"{self.base_url}/models"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [m.get("id", m.get("name", "unknown")) for m in data.get("data", data.get("models", []))]
        except Exception:
            pass
        return []


class NVIDIA_NIM_Client(OpenAICompatibleClient):
    """Client for NVIDIA NIM API."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or os.getenv("NVIDIA_API_KEY"),
            provider_name="NVIDIA"
        )
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not set")


class LiumClient(OpenAICompatibleClient):
    """Client for Lium GPU marketplace API."""

    # Known endpoint patterns - try in order
    KNOWN_ENDPOINTS = [
        "https://api.lium.ai/v1",
        "https://inference.lium.ai/v1",
        "https://llm.lium.ai/v1",
        "https://gpu.lium.ai/v1",
    ]

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        api_key = api_key or os.getenv("LIUM_API_KEY")
        if not api_key:
            raise ValueError("LIUM_API_KEY not set")

        # Use provided URL or try to detect
        if base_url:
            super().__init__(base_url, api_key, "Lium")
        else:
            # Will be set after detection
            self.api_key = api_key
            self._detected_url = None
            super().__init__(self.KNOWN_ENDPOINTS[0], api_key, "Lium")

    async def detect_endpoint(self) -> str:
        """Detect working Lium API endpoint."""
        if self._detected_url:
            return self._detected_url

        for endpoint in self.KNOWN_ENDPOINTS:
            try:
                self.base_url = endpoint
                models = await self.list_models()
                if models:
                    self._detected_url = endpoint
                    print(f"Lium endpoint detected: {endpoint}")
                    return endpoint
            except Exception:
                continue

        # Default to first if detection fails
        self._detected_url = self.KNOWN_ENDPOINTS[0]
        return self._detected_url


class LLMForecaster:
    """
    LLM-based price forecaster.

    Uses language models to analyze price history and generate
    predictions with reasoning.
    """

    ANALYSIS_PROMPT = """You are a quantitative trading analyst. Analyze the following price data and predict future movement.

Symbol: {symbol}
Timeframe: Hourly candles
Current price: {current_price}

Recent price history (last 24 hours):
{price_history}

Technical indicators:
- 24h change: {change_24h:.2f}%
- 7d trend: {trend_7d}
- Volatility (24h): {volatility:.2f}%
- RSI (14): {rsi:.1f}
- Volume vs avg: {volume_ratio:.2f}x

Market context:
- Funding rate: {funding_rate:.4f}
- Open interest change: {oi_change:.2f}%

Provide your analysis in this EXACT JSON format:
{{
 "direction": <1 for bullish, -1 for bearish, 0 for neutral>,
 "confidence": <0.0 to 1.0>,
 "predicted_return_1h": <decimal>,
 "predicted_return_4h": <decimal>,
 "predicted_return_8h": <decimal>,
 "predicted_return_24h": <decimal>,
 "reasoning": "<brief explanation of your prediction>"
}}

Respond ONLY with valid JSON, no other text."""

    def __init__(
        self,
        model: str = "meta/llama-3.1-70b-instruct",
        api_key: Optional[str] = None,
        provider: str = "nvidia",  # "nvidia" or "lium"
        base_url: Optional[str] = None
    ):
        self.model = model
        self.provider = provider
        self._initialized = False

        # Create appropriate client
        if provider == "lium":
            self.client = LiumClient(api_key, base_url)
        elif provider == "nvidia":
            self.client = NVIDIA_NIM_Client(api_key)
        else:
            # Generic OpenAI-compatible client
            if not base_url:
                raise ValueError(f"base_url required for provider: {provider}")
            self.client = OpenAICompatibleClient(base_url, api_key, provider)

    async def initialize(self):
        """Test connection."""
        try:
            # Simple test
            await self.client.complete(
                self.model,
                [{"role": "user", "content": "Reply with: OK"}],
                max_tokens=10
            )
            self._initialized = True
            print(f"LLMForecaster initialized with {self.provider}/{self.model}")
        except Exception as e:
            print(f"LLMForecaster init failed: {e}")
            self._initialized = False

    async def forecast(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        context: Optional[dict] = None
    ) -> LLMForecast:
        """
        Generate forecast using LLM analysis.

        Args:
            symbol: Trading symbol
            prices: Price history (hourly)
            volumes: Volume history
            context: Additional market context (funding, OI, etc.)

        Returns:
            LLMForecast with predictions and reasoning
        """
        if not self._initialized:
            await self.initialize()

        # Calculate technical indicators
        current_price = prices[-1]

        # Price changes
        change_24h = (prices[-1] - prices[-24]) / prices[-24] * 100 if len(prices) >= 24 else 0
        change_7d = prices[-1] - prices[-min(168, len(prices))]
        trend_7d = "up" if change_7d > 0 else "down"

        # Volatility
        if len(prices) >= 25:
            returns = np.diff(prices[-25:]) / prices[-25:-1]
            volatility = np.std(returns) * 100 * np.sqrt(24)
        else:
            volatility = 0

        # RSI (simplified)
        rsi = self._calculate_rsi(prices[-14:]) if len(prices) >= 14 else 50

        # Volume ratio
        volume_ratio = 1.0
        if volumes is not None and len(volumes) >= 24:
            recent_vol = np.mean(volumes[-6:])
            avg_vol = np.mean(volumes[-24:])
            volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # Format price history
        price_history = self._format_price_history(prices[-24:])

        # Build prompt
        prompt = self.ANALYSIS_PROMPT.format(
            symbol=symbol,
            current_price=current_price,
            price_history=price_history,
            change_24h=change_24h,
            trend_7d=trend_7d,
            volatility=volatility,
            rsi=rsi,
            volume_ratio=volume_ratio,
            funding_rate=context.get("funding_rate", 0) if context else 0,
            oi_change=context.get("oi_change", 0) if context else 0
        )

        try:
            # Get LLM response
            response = await self.client.complete(
                self.model,
                [{"role": "user", "content": prompt}],
                temperature=0.3
            )

            # Parse JSON response
            analysis = self._parse_response(response)

            # Build predictions array
            predictions = np.array([
                analysis.get("predicted_return_1h", 0),
                analysis.get("predicted_return_4h", 0),
                analysis.get("predicted_return_8h", 0),
                analysis.get("predicted_return_24h", 0)
            ])

            # Convert returns to prices
            price_predictions = current_price * (1 + predictions)
            lower = price_predictions * 0.98  # 2% bounds
            upper = price_predictions * 1.02

            return LLMForecast(
                model_name=f"{self.provider}/{self.model}",
                horizon=24,
                predictions=price_predictions,
                lower_bound=lower,
                upper_bound=upper,
                reasoning=analysis.get("reasoning", ""),
                direction=analysis.get("direction", 0),
                confidence=analysis.get("confidence", 0.5)
            )

        except Exception as e:
            print(f"LLM forecast error: {e}")
            # Fallback to simple prediction
            return self._fallback_forecast(prices)

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _format_price_history(self, prices: np.ndarray) -> str:
        """Format prices for prompt."""
        lines = []
        for i, p in enumerate(prices):
            hour = i
            change = ((p - prices[0]) / prices[0] * 100) if prices[0] > 0 else 0
            lines.append(f" h-{24-hour}: ${p:.2f} ({change:+.2f}%)")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Try to extract JSON
        try:
            # Find JSON object in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback defaults
        return {
            "direction": 0,
            "confidence": 0.3,
            "predicted_return_1h": 0,
            "predicted_return_4h": 0,
            "predicted_return_8h": 0,
            "predicted_return_24h": 0,
            "reasoning": "Failed to parse LLM response"
        }

    def _fallback_forecast(self, prices: np.ndarray) -> LLMForecast:
        """Fallback forecast when LLM fails."""
        if len(prices) < 2:
            return LLMForecast(
                model_name=f"{self.provider}/{self.model}",
                horizon=24,
                predictions=np.full(24, prices[-1] if len(prices) > 0 else 100.0),
                lower_bound=np.full(24, prices[-1] * 0.98 if len(prices) > 0 else 98.0),
                upper_bound=np.full(24, prices[-1] * 1.02 if len(prices) > 0 else 102.0),
                reasoning="Insufficient data for LLM forecast",
                direction=0,
                confidence=0.2
            )

        # Simple momentum
        momentum = (prices[-1] - prices[-24]) / prices[-24] if len(prices) >= 24 else 0
        direction = 1 if momentum > 0.01 else (-1 if momentum < -0.01 else 0)

        predictions = prices[-1] * (1 + momentum * np.linspace(0.5, 1.0, 24))

        return LLMForecast(
            model_name=f"{self.provider}/{self.model}",
            horizon=24,
            predictions=predictions,
            lower_bound=predictions * 0.95,
            upper_bound=predictions * 1.05,
            reasoning="Fallback: momentum-based forecast",
            direction=direction,
            confidence=0.4
        )


async def test_llm_forecaster():
    """Test the LLM forecaster."""
    # Test with NVIDIA
    print("Testing NVIDIA NIM forecaster...")
    forecaster_nvidia = LLMForecaster(provider="nvidia")

    # Generate test data
    np.random.seed(42)
    prices = 70000 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 168)))
    volumes = np.random.uniform(100, 1000, 168)

    print(f"Testing LLM forecaster with {len(prices)} hours of price data...")

    forecast = await forecaster_nvidia.forecast(
        symbol="BTC",
        prices=prices,
        volumes=volumes,
        context={"funding_rate": 0.0001, "oi_change": 2.5}
    )

    print(f"\nForecast from {forecast.model_name}:")
    print(f" Direction: {forecast.direction}")
    print(f" Confidence: {forecast.confidence:.2f}")
    print(f" 1h prediction: ${forecast.predictions[0]:.2f}")
    print(f" 24h prediction: ${forecast.predictions[-1]:.2f}")
    print(f" Reasoning: {forecast.reasoning}")

    # Test Lium if API key available
    if os.getenv("LIUM_API_KEY"):
        print("\n\nTesting Lium forecaster...")
        forecaster_lium = LLMForecaster(provider="lium")
        forecast_lium = await forecaster_lium.forecast(
            symbol="BTC",
            prices=prices,
            volumes=volumes
        )
        print(f"\nLium forecast: {forecast_lium.direction}, confidence={forecast_lium.confidence:.2f}")


if __name__ == "__main__":
    asyncio.run(test_llm_forecaster())
