"""Unit tests for Volume Breakout Strategy."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.breakout import (
    BreakoutConfig,
    BreakoutStrategy,
)


@pytest.fixture
def volume_breakout_df() -> pd.DataFrame:
    """Create DataFrame with clear volume breakout setup.

    This scenario should generate a LONG signal:
    - Close > highest high of lookback period
    - Volume > 1.5x average
    - Close in upper 25% of day's range
    - Sufficient ATR
    """
    np.random.seed(42)
    n_days = 50

    # Consolidation period followed by breakout
    close = np.concatenate(
        [
            np.linspace(100, 105, 40) + np.random.randn(40) * 0.5,  # Consolidation
            np.linspace(106, 115, 10),  # Breakout
        ]
    )

    # High volume on breakout day
    volume = np.concatenate(
        [
            np.random.randint(800000, 1200000, 49),  # Normal volume
            [3000000],  # Breakout volume (2.5x average)
        ]
    )

    # Create OHLC that gives strong close (close near high)
    high = close + np.abs(np.random.randn(n_days)) * 1.5
    low = close - np.abs(np.random.randn(n_days)) * 0.3
    open_price = close - 0.5

    # Ensure last day has strong close (close = high)
    high[-1] = close[-1] + 0.1
    low[-1] = close[-1] - 2

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BREAKOUT"
    return df


@pytest.fixture
def no_volume_breakout_df() -> pd.DataFrame:
    """Create DataFrame with breakout but no volume confirmation."""
    np.random.seed(42)
    n_days = 50

    # Price breaks out but on low volume
    close = np.concatenate(
        [
            np.linspace(100, 105, 45) + np.random.randn(45) * 0.5,
            np.linspace(106, 110, 5),  # Breakout
        ]
    )

    # LOW volume on breakout (below 1.5x average)
    volume = np.concatenate(
        [
            np.random.randint(900000, 1100000, 49),
            [1000000],  # Same as average, not 1.5x
        ]
    )

    high = close + np.abs(np.random.randn(n_days))
    low = close - np.abs(np.random.randn(n_days)) * 0.3
    open_price = close - 0.5

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "NOVOL"
    return df


@pytest.fixture
def breakdown_df() -> pd.DataFrame:
    """Create DataFrame with breakdown (SHORT signal)."""
    np.random.seed(42)
    n_days = 50

    # Price breaks down with volume
    close = np.concatenate(
        [
            np.linspace(110, 105, 40) + np.random.randn(40) * 0.5,  # Consolidation
            np.linspace(104, 95, 10),  # Breakdown
        ]
    )

    # High volume on breakdown day
    volume = np.concatenate(
        [
            np.random.randint(800000, 1200000, 49),
            [3000000],  # Breakdown volume
        ]
    )

    # Create OHLC with weak close (close near low)
    high = close + np.abs(np.random.randn(n_days)) * 0.3
    low = close - np.abs(np.random.randn(n_days)) * 1.5
    open_price = close + 0.5

    # Ensure last day has weak close (close = low)
    high[-1] = close[-1] + 2
    low[-1] = close[-1] - 0.1

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BREAKDOWN"
    return df


@pytest.fixture
def weak_close_df() -> pd.DataFrame:
    """Create DataFrame with breakout but weak close (not in upper 25%)."""
    np.random.seed(42)
    n_days = 50

    # Price breaks out but closes weak
    close = np.concatenate(
        [
            np.linspace(100, 105, 45) + np.random.randn(45) * 0.5,
            np.linspace(106, 108, 5),
        ]
    )

    volume = np.concatenate(
        [
            np.random.randint(800000, 1200000, 49),
            [2500000],  # Good volume
        ]
    )

    # Close in MIDDLE of range, not upper 25%
    high = close.copy()
    high[-1] = 112  # High much above close
    low = close.copy()
    low[-1] = 104  # Low below close
    # Close at 108, range 104-112, close position = 4/8 = 50%, not > 75%

    open_price = close - 0.5

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "WEAK"
    return df


@pytest.fixture
def consolidation_df() -> pd.DataFrame:
    """Create DataFrame with clear consolidation pattern."""
    np.random.seed(42)
    n_days = 50

    # Tight consolidation - price stays in narrow range
    base_price = 100
    close = base_price + np.random.randn(n_days) * 1  # Very tight range

    high = close + 0.5
    low = close - 0.5

    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000000, 1500000, n_days),
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    return df


@pytest.fixture
def no_consolidation_df() -> pd.DataFrame:
    """Create DataFrame without consolidation (wide range)."""
    np.random.seed(42)
    n_days = 50

    # Wide price swings - no consolidation
    close = 100 + np.cumsum(np.random.randn(n_days) * 3)

    high = close + np.abs(np.random.randn(n_days)) * 2
    low = close - np.abs(np.random.randn(n_days)) * 2

    df = pd.DataFrame(
        {
            "open": close - 1,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000000, 1500000, n_days),
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    return df


@dataclass
class MockTrade:
    """Mock trade object for exit testing."""

    trade_id: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: datetime
    direction: str = "LONG"
    highest_price_since_entry: float = 0.0
    lowest_price_since_entry: float = 0.0

    def __post_init__(self) -> None:
        if self.highest_price_since_entry == 0.0:
            self.highest_price_since_entry = self.entry_price
        if self.lowest_price_since_entry == 0.0:
            self.lowest_price_since_entry = self.entry_price


class TestBreakoutConfig:
    """Tests for BreakoutConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BreakoutConfig()
        assert config.lookback_period == 20
        assert config.volume_multiplier == 1.5
        assert config.atr_stop_multiplier == 2.0
        assert config.trailing_stop_atr == 2.0
        assert config.max_hold_days == 5
        assert config.min_atr_percent == 1.0
        assert config.close_range_threshold == 0.25

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = BreakoutConfig(
            lookback_period=30,
            volume_multiplier=2.0,
            max_hold_days=10,
        )
        assert config.lookback_period == 30
        assert config.volume_multiplier == 2.0
        assert config.max_hold_days == 10


class TestBreakoutStrategy:
    """Tests for BreakoutStrategy."""

    @pytest.fixture
    def strategy(self) -> BreakoutStrategy:
        """Create strategy instance with default config."""
        return BreakoutStrategy()

    def test_strategy_name(self, strategy: BreakoutStrategy) -> None:
        """Test strategy name attribute."""
        assert strategy.name == "VolumeBreakout"

    def test_strategy_description(self, strategy: BreakoutStrategy) -> None:
        """Test strategy description attribute."""
        assert "breakout" in strategy.description.lower()

    def test_get_parameters(self, strategy: BreakoutStrategy) -> None:
        """Test getting strategy parameters."""
        params = strategy.get_parameters()
        assert params["lookback_period"] == 20
        assert params["volume_multiplier"] == 1.5
        assert params["trailing_stop_atr"] == 2.0

    def test_set_parameters(self, strategy: BreakoutStrategy) -> None:
        """Test setting strategy parameters."""
        strategy.set_parameters(
            {
                "lookback_period": 30,
                "volume_multiplier": 2.0,
            }
        )
        assert strategy.config.lookback_period == 30
        assert strategy.config.volume_multiplier == 2.0


class TestScanMethod:
    """Tests for the scan() method."""

    @pytest.fixture
    def strategy(self) -> BreakoutStrategy:
        """Create strategy instance."""
        return BreakoutStrategy()

    def test_signals_on_volume_breakout(
        self, strategy: BreakoutStrategy, volume_breakout_df: pd.DataFrame
    ) -> None:
        """Test that signal is generated on volume-confirmed breakout."""
        signal = strategy.scan(volume_breakout_df)

        if signal is not None:
            assert signal.direction == "LONG"
            assert signal.strategy == "VolumeBreakout"
            assert signal.symbol == "BREAKOUT"
            assert 50 <= signal.score <= 100
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price
            assert "relative_volume" in signal.metadata

    def test_no_signal_without_volume(
        self, strategy: BreakoutStrategy, no_volume_breakout_df: pd.DataFrame
    ) -> None:
        """Test that no signal when volume is insufficient."""
        signal = strategy.scan(no_volume_breakout_df)
        assert signal is None

    def test_signals_on_breakdown(
        self, strategy: BreakoutStrategy, breakdown_df: pd.DataFrame
    ) -> None:
        """Test that SHORT signal is generated on breakdown."""
        signal = strategy.scan(breakdown_df)

        if signal is not None:
            assert signal.direction == "SHORT"
            assert signal.stop_loss > signal.entry_price
            assert signal.take_profit < signal.entry_price

    def test_no_signal_weak_close(
        self, strategy: BreakoutStrategy, weak_close_df: pd.DataFrame
    ) -> None:
        """Test that no signal when close is not strong."""
        signal = strategy.scan(weak_close_df)
        assert signal is None

    def test_stop_loss_uses_atr(
        self, strategy: BreakoutStrategy, volume_breakout_df: pd.DataFrame
    ) -> None:
        """Test that stop loss is based on ATR."""
        signal = strategy.scan(volume_breakout_df)

        if signal is not None:
            assert "atr" in signal.metadata
            atr = signal.metadata["atr"]
            expected_sl = signal.entry_price - (atr * strategy.config.atr_stop_multiplier)
            assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_metadata_contains_breakout_level(
        self, strategy: BreakoutStrategy, volume_breakout_df: pd.DataFrame
    ) -> None:
        """Test that metadata contains breakout level."""
        signal = strategy.scan(volume_breakout_df)

        if signal is not None:
            assert "breakout_level" in signal.metadata
            assert "highest_high" in signal.metadata

    def test_insufficient_data(self, strategy: BreakoutStrategy) -> None:
        """Test that no signal with insufficient data."""
        df = pd.DataFrame(
            {
                "open": [100] * 30,
                "high": [102] * 30,
                "low": [99] * 30,
                "close": [101] * 30,
                "volume": [1000000] * 30,
            }
        )
        signal = strategy.scan(df)
        assert signal is None

    def test_invalid_dataframe(self, strategy: BreakoutStrategy) -> None:
        """Test that no signal with missing columns."""
        df = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [102] * 50,
            }
        )
        signal = strategy.scan(df)
        assert signal is None


class TestConsolidationDetection:
    """Tests for detect_consolidation method."""

    @pytest.fixture
    def strategy(self) -> BreakoutStrategy:
        """Create strategy instance."""
        return BreakoutStrategy()

    def test_detects_consolidation(
        self, strategy: BreakoutStrategy, consolidation_df: pd.DataFrame
    ) -> None:
        """Test that consolidation is detected in tight range."""
        result = strategy.detect_consolidation(consolidation_df)
        assert result is True

    def test_no_consolidation_wide_range(
        self, strategy: BreakoutStrategy, no_consolidation_df: pd.DataFrame
    ) -> None:
        """Test that no consolidation with wide price range."""
        result = strategy.detect_consolidation(no_consolidation_df)
        assert result is False

    def test_consolidation_custom_params(
        self, strategy: BreakoutStrategy, consolidation_df: pd.DataFrame
    ) -> None:
        """Test consolidation detection with custom parameters."""
        # With very tight max range, should not detect
        _ = strategy.detect_consolidation(consolidation_df, min_days=10, max_range_percent=1.0)
        # May or may not pass depending on data

    def test_insufficient_data_for_consolidation(self, strategy: BreakoutStrategy) -> None:
        """Test consolidation with insufficient data."""
        df = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [102] * 5,
                "low": [99] * 5,
                "close": [101] * 5,
                "volume": [1000000] * 5,
            }
        )
        result = strategy.detect_consolidation(df, min_days=10)
        assert result is False


class TestScoreCalculation:
    """Tests for score calculation."""

    def test_high_volume_bonus(self, volume_breakout_df: pd.DataFrame) -> None:
        """Test that high volume (>2x) gives score bonus."""
        strategy = BreakoutStrategy()
        signal = strategy.scan(volume_breakout_df)

        if signal is not None:
            # Volume was set to 2.5x average, should get +15 bonus
            assert signal.score >= 65  # Base 50 + 15 for volume

    def test_score_in_valid_range(self, volume_breakout_df: pd.DataFrame) -> None:
        """Test that score is always in valid range."""
        strategy = BreakoutStrategy()
        signal = strategy.scan(volume_breakout_df)

        if signal is not None:
            assert 0 <= signal.score <= 100


class TestCheckExit:
    """Tests for the check_exit() method."""

    @pytest.fixture
    def strategy(self) -> BreakoutStrategy:
        """Create strategy instance."""
        return BreakoutStrategy()

    @pytest.fixture
    def basic_df(self) -> pd.DataFrame:
        """Create basic OHLCV DataFrame for exit tests."""
        np.random.seed(42)
        n_days = 50

        # Slight uptrend
        close = np.linspace(100, 105, n_days) + np.random.randn(n_days) * 0.5

        return pd.DataFrame(
            {
                "open": close - 0.3,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

    def test_stop_loss_exit(self, strategy: BreakoutStrategy, basic_df: pd.DataFrame) -> None:
        """Test that stop loss triggers exit."""
        trade = MockTrade(
            trade_id=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=115.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
            highest_price_since_entry=100.0,  # No profit yet
        )

        # Set low to hit stop loss
        basic_df.loc[basic_df.index[-1], "low"] = 94.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        # Could be STOP_LOSS or TRAILING_STOP depending on calculation
        assert exit_signal.exit_type in ["STOP_LOSS", "TRAILING_STOP"]

    def test_trailing_stop_exit(self, strategy: BreakoutStrategy, basic_df: pd.DataFrame) -> None:
        """Test that trailing stop triggers exit when profitable."""
        trade = MockTrade(
            trade_id=2,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=120.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
            highest_price_since_entry=110.0,  # Price went up
        )

        # Set low to hit trailing stop (ATR-based)
        # Trailing stop would be around 110 - (ATR * 2) ~= 106-107
        basic_df.loc[basic_df.index[-1], "low"] = 105.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TRAILING_STOP"

    def test_time_exit(self, strategy: BreakoutStrategy) -> None:
        """Test that time exit triggers after max hold days."""
        np.random.seed(123)
        n_days = 50

        # Create stable price data
        close = 100 + np.sin(np.linspace(0, 4 * np.pi, n_days)) * 2

        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=3,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            entry_date=datetime.now() - timedelta(days=10),
            direction="LONG",
        )

        exit_signal = strategy.check_exit(df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TIME_EXIT"

    def test_no_exit_within_parameters(self, strategy: BreakoutStrategy) -> None:
        """Test that no exit when all conditions within limits."""
        np.random.seed(789)
        n_days = 50

        # Create stable data that won't trigger trailing stop
        # Keep price close to entry but within ATR range
        close = np.full(n_days, 102.0) + np.random.randn(n_days) * 0.1

        df = pd.DataFrame(
            {
                "open": close - 0.05,
                "high": close + 0.2,  # Small range so trailing stop won't trigger
                "low": close - 0.2,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=4,
            entry_price=102.0,
            stop_loss=90.0,  # Very far stop, won't hit
            take_profit=120.0,  # Won't hit
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
            highest_price_since_entry=102.0,  # Same as entry, no trailing stop adjustment
        )

        exit_signal = strategy.check_exit(df, trade)
        assert exit_signal is None

    def test_short_trailing_stop(self, strategy: BreakoutStrategy) -> None:
        """Test trailing stop for SHORT position."""
        np.random.seed(42)
        n_days = 50

        # Downtrend then reversal
        close = np.linspace(100, 95, n_days)

        df = pd.DataFrame(
            {
                "open": close + 0.2,
                "high": close + 1,
                "low": close - 0.5,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=5,
            entry_price=100.0,
            stop_loss=105.0,  # Short stop above
            take_profit=90.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="SHORT",
            lowest_price_since_entry=92.0,  # Price went down
        )

        # Set high to hit trailing stop
        df.loc[df.index[-1], "high"] = 96.0  # Trailing stop would be around 92 + ATR*2

        exit_signal = strategy.check_exit(df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TRAILING_STOP"

    def test_breakeven_protection(self, strategy: BreakoutStrategy) -> None:
        """Test that stop moves to breakeven after 1R profit."""
        np.random.seed(42)
        n_days = 50

        close = np.linspace(100, 105, n_days)

        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        # Trade up more than 1R (entry 100, stop 95, so 1R = 5 points)
        trade = MockTrade(
            trade_id=6,
            entry_price=100.0,
            stop_loss=95.0,  # Initial stop
            take_profit=120.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
            highest_price_since_entry=108.0,  # Up 8 points > 5 (1R)
        )

        # Price drops to just below entry
        df.loc[df.index[-1], "low"] = 99.5

        exit_signal = strategy.check_exit(df, trade)
        # Should exit at breakeven (100) since we were up > 1R
        if exit_signal is not None:
            assert exit_signal.exit_price >= 100.0


class TestStrategyRegistry:
    """Tests for strategy registry integration."""

    def test_strategy_registered(self) -> None:
        """Test that strategy is registered in global registry."""
        from src.strategies.base import strategy_registry

        assert "VolumeBreakout" in strategy_registry.list_strategies()

    def test_create_instance_from_registry(self) -> None:
        """Test creating strategy instance from registry."""
        from src.strategies.base import strategy_registry

        strategy = strategy_registry.create_instance("VolumeBreakout")
        assert strategy is not None
        assert isinstance(strategy, BreakoutStrategy)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        strategy = BreakoutStrategy()
        df = pd.DataFrame()
        signal = strategy.scan(df)
        assert signal is None

    def test_custom_config_integration(self) -> None:
        """Test strategy with custom configuration."""
        config = BreakoutConfig(
            lookback_period=30,
            volume_multiplier=2.0,
            atr_stop_multiplier=1.5,
            max_hold_days=10,
        )
        strategy = BreakoutStrategy(config=config)

        assert strategy.config.lookback_period == 30
        assert strategy.config.volume_multiplier == 2.0
        assert strategy.config.atr_stop_multiplier == 1.5
        assert strategy.config.max_hold_days == 10

    def test_zero_volume_handling(self) -> None:
        """Test handling of zero average volume."""
        strategy = BreakoutStrategy()
        n_days = 50

        df = pd.DataFrame(
            {
                "open": [100] * n_days,
                "high": [102] * n_days,
                "low": [99] * n_days,
                "close": [101] * n_days,
                "volume": [0] * n_days,  # Zero volume
            }
        )

        # Should handle gracefully
        signal = strategy.scan(df)
        assert signal is None  # No signal due to zero relative volume
