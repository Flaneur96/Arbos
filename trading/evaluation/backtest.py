"""
Walk-forward validation for trading strategies.

Ensures strict out-of-sample testing by:
- Training on expanding window
- Testing on forward window
- Rolling forward and repeating
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    # Window sizes (in data points)
    min_train_size: int = 168  # 7 days minimum training
    train_window: int = 720  # 30 days training window
    test_window: int = 24  # 1 day test window
    step_size: int = 24  # Step by 1 day

    # Metrics to track
    metrics: list[str] = field(default_factory=lambda: [
        "sharpe", "sortino", "max_dd", "win_rate", "total_return"
    ])

    # Filtering
    min_sharpe_threshold: float = 0.5  # Minimum Sharpe to pass


@dataclass
class ValidationResult:
    """Result of a single validation fold."""
    fold_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_size: int
    test_size: int
    predictions: list[dict] = field(default_factory=list)
    actual_returns: list[float] = field(default_factory=list)
    pnl: list[float] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results."""
    folds: list[ValidationResult]
    avg_sharpe: float
    avg_return: float
    avg_max_dd: float
    avg_win_rate: float
    consistency_score: float  # How consistent across folds
    passed: bool


class WalkForwardValidator:
    """
    Walk-forward validation for trading models.

    Strict out-of-sample testing that prevents overfitting.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()
        self._results: list[ValidationResult] = []

    def validate(
        self,
        data: pd.DataFrame,
        predict_func: Callable,
        metric_func: Optional[Callable] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            data: DataFrame with features and returns
            predict_func: Function(data_train, data_test) -> predictions
            metric_func: Optional custom metric function

        Returns:
            WalkForwardResult with aggregated metrics
        """
        self._results = []

        if len(data) < self.config.min_train_size + self.config.test_window:
            raise ValueError("Not enough data for walk-forward validation")

        fold_id = 0

        # Generate fold boundaries
        for fold_start in range(
            self.config.min_train_size,
            len(data) - self.config.test_window,
            self.config.step_size
        ):
            train_start_idx = max(0, fold_start - self.config.train_window)
            train_end_idx = fold_start
            test_start_idx = fold_start
            test_end_idx = min(len(data), fold_start + self.config.test_window)

            # Extract train and test data
            train_data = data.iloc[train_start_idx:train_end_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]

            # Run prediction
            try:
                predictions = predict_func(train_data, test_data)

                # Calculate actual returns
                actual_returns = self._get_actual_returns(test_data)

                # Calculate PnL
                pnl = self._calculate_pnl(predictions, actual_returns)

                # Calculate metrics
                metrics = self._calculate_metrics(pnl)

                result = ValidationResult(
                    fold_id=fold_id,
                    train_start=train_data.index[0] if len(train_data) > 0 else None,
                    train_end=train_data.index[-1] if len(train_data) > 0 else None,
                    test_start=test_data.index[0] if len(test_data) > 0 else None,
                    test_end=test_data.index[-1] if len(test_data) > 0 else None,
                    train_size=len(train_data),
                    test_size=len(test_data),
                    predictions=predictions,
                    actual_returns=actual_returns,
                    pnl=pnl,
                    metrics=metrics
                )

                self._results.append(result)
                fold_id += 1

            except Exception as e:
                print(f"Error in fold {fold_id}: {e}")
                continue

        return self._aggregate_results()

    def _get_actual_returns(self, test_data: pd.DataFrame) -> list[float]:
        """Extract actual returns from test data."""
        if "return_1h" in test_data.columns:
            return test_data["return_1h"].tolist()
        elif "close" in test_data.columns:
            return test_data["close"].pct_change().dropna().tolist()
        return []

    def _calculate_pnl(
        self,
        predictions: list[dict],
        actual_returns: list[float]
    ) -> list[float]:
        """Calculate PnL from predictions and actual returns."""
        pnl = []

        for i, pred in enumerate(predictions):
            if i < len(actual_returns):
                direction = pred.get("direction", 0)
                confidence = pred.get("confidence", 1.0)

                # Position size based on confidence
                size = confidence

                # PnL = direction * actual_return * size
                trade_pnl = direction * actual_returns[i] * size
                pnl.append(trade_pnl)

        return pnl

    def _calculate_metrics(self, pnl: list[float]) -> dict:
        """Calculate performance metrics from PnL."""
        if not pnl:
            return {}

        pnl_arr = np.array(pnl)

        # Total return
        total_return = np.sum(pnl_arr)

        # Sharpe ratio (annualized for hourly data)
        if len(pnl_arr) > 1 and np.std(pnl_arr) > 0:
            sharpe = np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(365 * 24)
        else:
            sharpe = 0

        # Sortino ratio (downside deviation)
        negative_returns = pnl_arr[pnl_arr < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            sortino = np.mean(pnl_arr) / downside_std * np.sqrt(365 * 24) if downside_std > 0 else 0
        else:
            sortino = sharpe

        # Max drawdown
        cumulative = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Win rate
        wins = np.sum(pnl_arr > 0)
        total = len(pnl_arr)
        win_rate = wins / total if total > 0 else 0

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "total_return": total_return,
            "trades": total
        }

    def _aggregate_results(self) -> WalkForwardResult:
        """Aggregate results across all folds."""
        if not self._results:
            return WalkForwardResult(
                folds=[],
                avg_sharpe=0,
                avg_return=0,
                avg_max_dd=0,
                avg_win_rate=0,
                consistency_score=0,
                passed=False
            )

        # Calculate averages
        sharpes = [r.metrics.get("sharpe", 0) for r in self._results]
        returns = [r.metrics.get("total_return", 0) for r in self._results]
        max_dds = [r.metrics.get("max_dd", 0) for r in self._results]
        win_rates = [r.metrics.get("win_rate", 0) for r in self._results]

        avg_sharpe = np.mean(sharpes)
        avg_return = np.mean(returns)
        avg_max_dd = np.mean(max_dds)
        avg_win_rate = np.mean(win_rates)

        # Consistency: standard deviation of Sharpe across folds
        sharpe_std = np.std(sharpes) if len(sharpes) > 1 else 0
        consistency_score = 1 / (1 + sharpe_std)  # Higher is better

        # Pass criteria
        passed = (
            avg_sharpe > self.config.min_sharpe_threshold and
            avg_win_rate > 0.45 and
            consistency_score > 0.5
        )

        return WalkForwardResult(
            folds=self._results,
            avg_sharpe=avg_sharpe,
            avg_return=avg_return,
            avg_max_dd=avg_max_dd,
            avg_win_rate=avg_win_rate,
            consistency_score=consistency_score,
            passed=passed
        )

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if not self._results:
            return "No validation results"

        result = self._aggregate_results()

        summary = f"""
Walk-Forward Validation Summary
==============================
Folds: {len(self._results)}
Avg Sharpe: {result.avg_sharpe:.3f}
Avg Return: {result.avg_return:.2%}
Avg Max DD: {result.avg_max_dd:.2%}
Avg Win Rate: {result.avg_win_rate:.2%}
Consistency: {result.consistency_score:.3f}
Passed: {result.passed}
"""
        return summary


class OnlineSharpeFilter:
    """
    Online Sharpe filtering for live trading.

    Tracks rolling Sharpe and filters strategies/models
    that fall below threshold.
    """

    def __init__(
        self,
        window: int = 168,  # 7 days rolling
        min_sharpe: float = 0.5,
        lookback: int = 24  # Minimum samples before filtering
    ):
        self.window = window
        self.min_sharpe = min_sharpe
        self.lookback = lookback

        self._returns: dict[str, deque] = {}
        self._sharpe_cache: dict[str, float] = {}

    def update(self, strategy_id: str, pnl: float):
        """Update with new PnL observation."""
        if strategy_id not in self._returns:
            self._returns[strategy_id] = deque(maxlen=self.window)

        self._returns[strategy_id].append(pnl)

        # Recalculate Sharpe
        self._recalculate_sharpe(strategy_id)

    def _recalculate_sharpe(self, strategy_id: str):
        """Recalculate Sharpe for a strategy."""
        returns = list(self._returns.get(strategy_id, []))

        if len(returns) < self.lookback:
            self._sharpe_cache[strategy_id] = 0
            return

        arr = np.array(returns)
        mean_r = np.mean(arr)
        std_r = np.std(arr)

        if std_r > 0:
            # Annualize for hourly data
            self._sharpe_cache[strategy_id] = mean_r / std_r * np.sqrt(365 * 24)
        else:
            self._sharpe_cache[strategy_id] = 0

    def get_sharpe(self, strategy_id: str) -> float:
        """Get current Sharpe for strategy."""
        return self._sharpe_cache.get(strategy_id, 0)

    def is_valid(self, strategy_id: str) -> bool:
        """Check if strategy passes Sharpe filter."""
        returns = self._returns.get(strategy_id, [])

        if len(returns) < self.lookback:
            return True  # Not enough data yet

        sharpe = self.get_sharpe(strategy_id)
        return sharpe >= self.min_sharpe

    def get_all_valid(self) -> list[str]:
        """Get all strategies passing filter."""
        return [
            sid for sid in self._returns.keys()
            if self.is_valid(sid)
        ]

    def get_all_invalid(self) -> list[str]:
        """Get all strategies failing filter."""
        return [
            sid for sid in self._returns.keys()
            if not self.is_valid(sid)
        ]


def main():
    """Test walk-forward validation."""
    # Create mock data
    np.random.seed(42)
    dates = pd.date_range(
        start="2024-01-01",
        periods=1000,
        freq="h"
    )

    # Random walk prices
    returns = np.random.normal(0.0001, 0.01, 1000)
    prices = 100 * np.cumprod(1 + returns)

    data = pd.DataFrame({
        "close": prices,
        "return_1h": returns
    }, index=dates)

    # Mock prediction function
    def mock_predict(train, test):
        predictions = []
        for i in range(len(test)):
            # Simple momentum prediction
            if i > 0:
                momentum = train["return_1h"].tail(24).mean()
                pred = {
                    "direction": 1 if momentum > 0 else -1,
                    "confidence": min(1.0, abs(momentum) * 100)
                }
            else:
                pred = {"direction": 0, "confidence": 0.5}
            predictions.append(pred)
        return predictions

    # Run validation
    validator = WalkForwardValidator()
    result = validator.validate(data, mock_predict)

    print(validator.get_summary())


if __name__ == "__main__":
    main()
