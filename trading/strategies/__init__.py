"""
Trading strategies module.

- Signal generation
- Consensus gating
- Position management
"""

from .gating import (
    ConsensusGating,
    GatingConfig,
    GatedSignal,
    MultiSourceGating,
    momentum_signal,
    volatility_signal,
    derivatives_signal
)

__all__ = [
    "ConsensusGating",
    "GatingConfig",
    "GatedSignal",
    "MultiSourceGating",
    "momentum_signal",
    "volatility_signal",
    "derivatives_signal"
]
