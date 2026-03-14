"""
Foundation model wrappers for time-series forecasting.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np


@dataclass
class ForecastResult:
    """Forecast output from a model."""
    model_name: str
    horizon: int
    predictions: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    confidence: float = 0.95
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class FoundationModelWrapper(ABC):
    """Abstract base class for foundation model wrappers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._initialized = False

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def forecast(
        self,
        series: np.ndarray,
        horizon: int,
        context_length: Optional[int] = None
    ) -> ForecastResult:
        pass

    @abstractmethod
    async def batch_forecast(
        self,
        series_list: list[np.ndarray],
        horizon: int
    ) -> list[ForecastResult]:
        pass


class ChronosWrapper(FoundationModelWrapper):
    """Wrapper for Amazon's Chronos model."""

    def __init__(self, model_size: str = "small"):
        super().__init__(f"chronos-{model_size}")
        self.model_size = model_size
        self.model_id = f"amazon/chronos-t5-{model_size}"

    async def initialize(self):
        try:
            from chronos import ChronosPipeline
            self._model = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map="auto"
            )
            self._initialized = True
        except ImportError:
            self._initialized = True

    async def forecast(
        self,
        series: np.ndarray,
        horizon: int,
        context_length: Optional[int] = None
    ) -> ForecastResult:
        if not self._initialized:
            await self.initialize()

        if context_length and len(series) > context_length:
            series = series[-context_length:]

        if self._model is not None:
            forecast = self._model.predict(
                series,
                prediction_length=horizon,
                num_samples=100
            )
            predictions = np.median(forecast, axis=0)
            lower = np.percentile(forecast, 5, axis=0)
            upper = np.percentile(forecast, 95, axis=0)
        else:
            predictions = self._mock_forecast(series, horizon)
            lower = predictions * 0.95
            upper = predictions * 1.05

        return ForecastResult(
            model_name=self.model_name,
            horizon=horizon,
            predictions=predictions,
            lower_bound=lower,
            upper_bound=upper
        )

    def _mock_forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """Regime-aware mock forecast."""
        if len(series) < 2:
            return np.full(horizon, series[-1] if len(series) > 0 else 100.0)

        # Calculate returns safely
        if len(series) >= 25:
            returns = np.diff(series[-25:]) / series[-25:-1]
        elif len(series) > 1:
            returns = np.diff(series) / series[:-1]
        else:
            returns = np.array([0.0])

        trend = np.mean(returns) * 24 if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(24) if len(returns) > 1 else 0

        ema_short = np.mean(series[-6:]) if len(series) >= 6 else series[-1]
        ema_long = np.mean(series[-24:]) if len(series) >= 24 else series[-1]
        mean_reversion = (ema_long - ema_short) / series[-1]

        gains = np.sum(returns[returns > 0])
        losses = -np.sum(returns[returns < 0])
        rsi = gains / (gains + losses + 1e-8) * 100 if (gains + losses) > 0 else 50

        predictions = np.zeros(horizon)
        last_val = series[-1]

        for i in range(horizon):
            base = last_val + trend * series[-1] * 0.01
            if rsi > 70:
                reversion_adj = -0.002 * series[-1]
            elif rsi < 30:
                reversion_adj = 0.002 * series[-1]
            else:
                reversion_adj = mean_reversion * series[-1] * 0.1
            vol_adj = volatility * 0.1 if volatility > 0.03 else 0
            predictions[i] = base + reversion_adj + np.random.normal(0, vol_adj)
            last_val = predictions[i]

        return predictions

    async def batch_forecast(
        self,
        series_list: list[np.ndarray],
        horizon: int
    ) -> list[ForecastResult]:
        tasks = [self.forecast(s, horizon) for s in series_list]
        return await asyncio.gather(*tasks)


class TimesFMWrapper(FoundationModelWrapper):
    """Wrapper for Google's TimesFM model."""

    def __init__(self, backend: str = "jax"):
        super().__init__("timesfm")
        self.backend = backend

    async def initialize(self):
        try:
            import timesfm
            self._model = timesfm.TimesFm(
                context_len=512,
                horizon_len=24,
                input_patch_len=32,
                output_patch_len=128,
                backend=self.backend
            )
            self._initialized = True
        except ImportError:
            self._initialized = True

    async def forecast(
        self,
        series: np.ndarray,
        horizon: int,
        context_length: Optional[int] = None
    ) -> ForecastResult:
        if not self._initialized:
            await self.initialize()

        if self._model is not None:
            import timesfm
            freq = timesfm.freq_mask_from_freq("H")
            predictions = self._model.forecast([series], freq=[freq])[0]
        else:
            predictions = self._mock_forecast(series, horizon)

        return ForecastResult(
            model_name=self.model_name,
            horizon=horizon,
            predictions=predictions[:horizon]
        )

    def _mock_forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        if len(series) < 2:
            return np.full(horizon, series[-1] if len(series) > 0 else 100.0)

        mean = np.mean(series[-24:]) if len(series) >= 24 else np.mean(series)
        std = np.std(series[-24:]) + 0.01 if len(series) >= 24 else 1.0

        predictions = np.zeros(horizon)
        predictions[0] = series[-1]

        for i in range(1, horizon):
            predictions[i] = predictions[i-1] + (mean - predictions[i-1]) * 0.1
            predictions[i] += np.random.normal(0, std * 0.1)

        return predictions

    async def batch_forecast(
        self,
        series_list: list[np.ndarray],
        horizon: int
    ) -> list[ForecastResult]:
        tasks = [self.forecast(s, horizon) for s in series_list]
        return await asyncio.gather(*tasks)


class FoundationModelRegistry:
    """Registry for managing foundation models."""

    def __init__(self):
        self._models: dict[str, FoundationModelWrapper] = {}
        self._llm_models: dict[str, any] = {}

    def register(self, model):
        if hasattr(model, 'model_name'):
            if isinstance(model, FoundationModelWrapper):
                self._models[model.model_name] = model
            else:
                key = getattr(model, 'model', model.model_name)
                self._llm_models[key] = model

    async def get_model(self, name: str):
        if name in self._models:
            model = self._models[name]
            if not model._initialized:
                await model.initialize()
            return model

        if name in self._llm_models:
            model = self._llm_models[name]
            if not getattr(model, '_initialized', False):
                await model.initialize()
            return model

        raise ValueError(f"Model {name} not registered")

    async def initialize_all(self):
        tasks = []
        for m in self._models.values():
            tasks.append(m.initialize())
        for m in self._llm_models.values():
            if hasattr(m, 'initialize'):
                tasks.append(m.initialize())
        await asyncio.gather(*tasks, return_exceptions=True)

    def list_models(self) -> list[str]:
        return list(self._models.keys()) + list(self._llm_models.keys())


def create_default_registry() -> FoundationModelRegistry:
    """Create registry with default models."""
    registry = FoundationModelRegistry()

    # Foundation models (mock for now - require local GPU)
    registry.register(ChronosWrapper("small"))
    registry.register(ChronosWrapper("base"))
    registry.register(TimesFMWrapper())

    # LLM-based forecaster with real inference
    try:
        from .llm_forecaster import LLMForecaster
        import os

        # Prefer NVIDIA NIM (free tier available)
        if os.getenv("NVIDIA_API_KEY"):
            registry.register(LLMForecaster(
                provider="nvidia",
                model="meta/llama-3.1-70b-instruct"
            ))
            print("Registered NVIDIA NIM LLM forecaster")

        # Lium if configured
        if os.getenv("LIUM_API_KEY"):
            try:
                registry.register(LLMForecaster(
                    provider="lium",
                    model="meta-llama/Llama-3-70b"
                ))
                print("Registered Lium LLM forecaster")
            except Exception as e:
                print(f"Lium registration failed: {e}")

    except ImportError:
        pass

    return registry
