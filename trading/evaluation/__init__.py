"""
Evaluation module for trading strategies.

- Walk-forward validation
- Sharpe filtering
- Performance metrics
"""

from .backtest import (
    WalkForwardValidator,
    WalkForwardConfig,
    ValidationResult,
    WalkForwardResult,
    OnlineSharpeFilter
)

__all__ = [
    "WalkForwardValidator",
    "WalkForwardConfig",
    "ValidationResult",
    "WalkForwardResult",
    "OnlineSharpeFilter"
]
