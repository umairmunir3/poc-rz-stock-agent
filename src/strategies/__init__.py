"""Trading strategies module."""

from src.strategies.base import (
    Direction,
    ExitSignal,
    ExitType,
    Signal,
    Strategy,
    StrategyParameters,
    StrategyRegistry,
    StrategyResult,
    strategy_registry,
)
from src.strategies.breakout import (
    BreakoutConfig,
    BreakoutStrategy,
)
from src.strategies.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)
from src.strategies.macd_divergence import (
    MACDDivergenceConfig,
    MACDDivergenceStrategy,
)
from src.strategies.rsi_mean_reversion import (
    RSIMeanReversionStrategy,
    RSIStrategyConfig,
)
from src.strategies.scanner import (
    DailyScanResult,
    ScanMetrics,
    StrategyScanner,
)
from src.strategies.support_bounce import (
    SupportBounceConfig,
    SupportBounceStrategy,
)

__all__ = [
    "DailyScanResult",
    "BreakoutConfig",
    "BreakoutStrategy",
    "Direction",
    "EMACrossoverConfig",
    "EMACrossoverStrategy",
    "ExitSignal",
    "ExitType",
    "MACDDivergenceConfig",
    "MACDDivergenceStrategy",
    "RSIMeanReversionStrategy",
    "RSIStrategyConfig",
    "ScanMetrics",
    "Signal",
    "Strategy",
    "StrategyScanner",
    "StrategyParameters",
    "StrategyRegistry",
    "StrategyResult",
    "SupportBounceConfig",
    "SupportBounceStrategy",
    "strategy_registry",
]
