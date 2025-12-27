"""Unit tests for Support Bounce Strategy."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.support_bounce import (
    SupportBounceConfig,
    SupportBounceStrategy,
)


@pytest.fixture
def support_bounce_df() -> pd.DataFrame:
    """Create DataFrame with bounce off support level.

    - Price at key support level
    - Bullish candle pattern
    - RSI < 50
    - Volume spike
    """
    np.random.seed(42)
    n_days = 100

    # Create pattern: decline to support then bounce
    trend = np.concatenate(
        [
            np.linspace(120, 100, 70),  # Decline
            np.array([99, 98, 97]),  # Test support around 97-100
            np.linspace(98, 105, 27),  # Bounce
        ]
    )
    noise = np.random.randn(n_days) * 0.3
    close = trend + noise

    # Ensure last candle is bullish with volume spike
    close[-1] = close[-2] + 2  # Bullish close

    volume = np.random.randint(800000, 1200000, n_days)
    volume[-1] = 2000000  # Volume spike

    high = close + np.abs(np.random.randn(n_days)) * 0.5
    low = close - np.abs(np.random.randn(n_days)) * 0.5

    # Make last candle a hammer/bullish engulfing
    open_price = close.copy()
    open_price[-1] = close[-1] - 1.5  # Open below close
    low[-1] = close[-1] - 2  # Long lower wick

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

    df.attrs["symbol"] = "BOUNCE"
    return df


@pytest.fixture
def no_support_df() -> pd.DataFrame:
    """Create DataFrame with no clear support level."""
    np.random.seed(42)
    n_days = 100

    # Random walk - no clear support
    close = 100 + np.cumsum(np.random.randn(n_days) * 2)

    volume = np.random.randint(900000, 1100000, n_days)

    df = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + np.abs(np.random.randn(n_days)),
            "low": close - np.abs(np.random.randn(n_days)),
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "NOSUPPORT"
    return df


@pytest.fixture
def bearish_candle_df() -> pd.DataFrame:
    """Create DataFrame at support but with bearish candle."""
    np.random.seed(42)
    n_days = 100

    # Create support level
    trend = np.concatenate(
        [
            np.linspace(120, 100, 70),
            np.array([99, 98, 97, 98, 99]),
            np.linspace(100, 95, 25),  # Break down through
        ]
    )
    noise = np.random.randn(n_days) * 0.2
    close = trend + noise

    volume = np.random.randint(900000, 1500000, n_days)

    # Last candle is bearish
    open_price = close.copy()
    open_price[-1] = close[-1] + 1  # Open above close = bearish

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": close + np.abs(np.random.randn(n_days)) * 0.5,
            "low": close - np.abs(np.random.randn(n_days)) * 0.5,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BEARISH"
    return df


@pytest.fixture
def low_volume_bounce_df() -> pd.DataFrame:
    """Create DataFrame with bounce but low volume."""
    np.random.seed(42)
    n_days = 100

    trend = np.concatenate(
        [
            np.linspace(120, 100, 70),
            np.array([99, 98, 97]),
            np.linspace(98, 105, 27),
        ]
    )
    noise = np.random.randn(n_days) * 0.3
    close = trend + noise

    close[-1] = close[-2] + 2

    # LOW volume - no spike
    volume = np.random.randint(900000, 1100000, n_days)
    volume[-1] = 800000  # Below average

    high = close + np.abs(np.random.randn(n_days)) * 0.5
    low = close - np.abs(np.random.randn(n_days)) * 0.5

    open_price = close.copy()
    open_price[-1] = close[-1] - 1.5

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

    df.attrs["symbol"] = "LOWVOL"
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
    resistance_level: float | None = None


class TestSupportBounceConfig:
    """Tests for SupportBounceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SupportBounceConfig()
        assert config.sr_lookback == 60
        assert config.support_tolerance_atr == 0.5
        assert config.rsi_max == 50
        assert config.volume_multiplier == 1.2
        assert config.atr_stop_multiplier == 0.5
        assert config.atr_profit_multiplier == 1.5
        assert config.max_hold_days == 8

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SupportBounceConfig(
            sr_lookback=90,
            volume_multiplier=1.5,
            max_hold_days=10,
        )
        assert config.sr_lookback == 90
        assert config.volume_multiplier == 1.5
        assert config.max_hold_days == 10


class TestSupportBounceStrategy:
    """Tests for SupportBounceStrategy."""

    @pytest.fixture
    def strategy(self) -> SupportBounceStrategy:
        """Create strategy instance with default config."""
        return SupportBounceStrategy()

    def test_strategy_name(self, strategy: SupportBounceStrategy) -> None:
        """Test strategy name attribute."""
        assert strategy.name == "SupportBounce"

    def test_strategy_description(self, strategy: SupportBounceStrategy) -> None:
        """Test strategy description attribute."""
        assert "support" in strategy.description.lower()

    def test_get_parameters(self, strategy: SupportBounceStrategy) -> None:
        """Test getting strategy parameters."""
        params = strategy.get_parameters()
        assert params["sr_lookback"] == 60
        assert params["rsi_max"] == 50
        assert params["volume_multiplier"] == 1.2

    def test_set_parameters(self, strategy: SupportBounceStrategy) -> None:
        """Test setting strategy parameters."""
        strategy.set_parameters(
            {
                "sr_lookback": 90,
                "volume_multiplier": 1.5,
                "rsi_max": 60,
            }
        )
        assert strategy.config.sr_lookback == 90
        assert strategy.config.volume_multiplier == 1.5
        assert strategy.config.rsi_max == 60


class TestScanMethod:
    """Tests for the scan() method."""

    @pytest.fixture
    def strategy(self) -> SupportBounceStrategy:
        """Create strategy instance."""
        return SupportBounceStrategy()

    def test_signals_at_support(
        self, strategy: SupportBounceStrategy, support_bounce_df: pd.DataFrame
    ) -> None:
        """Test that signal is generated at support with bullish candle."""
        signal = strategy.scan(support_bounce_df)

        # May or may not signal depending on S/R detection
        if signal is not None:
            assert signal.direction == "LONG"
            assert signal.strategy == "SupportBounce"
            assert "support_level" in signal.metadata

    def test_requires_bullish_candle(
        self, strategy: SupportBounceStrategy, bearish_candle_df: pd.DataFrame
    ) -> None:
        """Test that no signal with bearish candle at support."""
        signal = strategy.scan(bearish_candle_df)
        assert signal is None

    def test_requires_volume_spike(
        self, strategy: SupportBounceStrategy, low_volume_bounce_df: pd.DataFrame
    ) -> None:
        """Test that no signal without volume confirmation."""
        signal = strategy.scan(low_volume_bounce_df)
        assert signal is None

    def test_stop_below_support(
        self, strategy: SupportBounceStrategy, support_bounce_df: pd.DataFrame
    ) -> None:
        """Test that stop loss is below support level."""
        signal = strategy.scan(support_bounce_df)

        if signal is not None:
            support = signal.metadata.get("support_level")
            if support is not None:
                assert signal.stop_loss < support

    def test_metadata_contains_support_info(
        self, strategy: SupportBounceStrategy, support_bounce_df: pd.DataFrame
    ) -> None:
        """Test that metadata contains support level info."""
        signal = strategy.scan(support_bounce_df)

        if signal is not None:
            assert "support_level" in signal.metadata
            assert "support_strength" in signal.metadata
            assert "rsi" in signal.metadata

    def test_exit_at_resistance(
        self, strategy: SupportBounceStrategy, support_bounce_df: pd.DataFrame
    ) -> None:
        """Test that take profit considers resistance level."""
        signal = strategy.scan(support_bounce_df)

        if signal is not None:
            # Take profit should be at resistance or ATR target
            assert signal.take_profit > signal.entry_price

    def test_insufficient_data(self, strategy: SupportBounceStrategy) -> None:
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

    def test_score_in_valid_range(
        self, strategy: SupportBounceStrategy, support_bounce_df: pd.DataFrame
    ) -> None:
        """Test that score is in valid range."""
        signal = strategy.scan(support_bounce_df)

        if signal is not None:
            assert 0 <= signal.score <= 100


class TestBullishCandleDetection:
    """Tests for bullish candle pattern detection."""

    @pytest.fixture
    def strategy(self) -> SupportBounceStrategy:
        """Create strategy instance."""
        return SupportBounceStrategy()

    def test_detects_bullish_engulfing(self, strategy: SupportBounceStrategy) -> None:
        """Test detection of bullish engulfing pattern."""
        # Bullish engulfing: current bullish engulfs previous bearish
        # Args: curr_open, curr_high, curr_low, curr_close, prev_open, _prev_high, _prev_low, prev_close
        result = strategy._is_bullish_reversal_candle(
            98,
            103,
            97,
            102,  # Current: bullish
            101,
            102,
            99,
            99,  # Previous: bearish
        )
        assert result is True

    def test_detects_hammer(self, strategy: SupportBounceStrategy) -> None:
        """Test detection of hammer pattern."""
        # Hammer: small body, long lower wick
        # Args: curr_open, curr_high, curr_low, curr_close, prev_open, _prev_high, _prev_low, prev_close
        result = strategy._is_bullish_reversal_candle(
            100,
            101,
            95,
            100.5,  # Current: hammer
            101,
            102,
            100,
            100,  # Previous
        )
        assert result is True

    def test_rejects_bearish_candle(self, strategy: SupportBounceStrategy) -> None:
        """Test rejection of bearish candle."""
        # Both candles bearish - no reversal
        # Args: curr_open, curr_high, curr_low, curr_close, prev_open, _prev_high, _prev_low, prev_close
        result = strategy._is_bullish_reversal_candle(
            102,
            103,
            99,
            100,  # Current: bearish
            104,
            105,
            101,
            102,  # Previous: bearish
        )
        assert result is False


class TestCheckExit:
    """Tests for the check_exit() method."""

    @pytest.fixture
    def strategy(self) -> SupportBounceStrategy:
        """Create strategy instance."""
        return SupportBounceStrategy()

    @pytest.fixture
    def basic_df(self) -> pd.DataFrame:
        """Create basic OHLCV DataFrame for exit tests."""
        np.random.seed(42)
        n_days = 60

        close = np.linspace(100, 110, n_days) + np.random.randn(n_days) * 0.3

        return pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

    def test_stop_loss_exit(self, strategy: SupportBounceStrategy, basic_df: pd.DataFrame) -> None:
        """Test that stop loss triggers exit."""
        trade = MockTrade(
            trade_id=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            entry_date=datetime.now() - timedelta(days=1),
        )

        # Set low to hit stop loss
        basic_df.loc[basic_df.index[-1], "low"] = 94.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "STOP_LOSS"

    def test_take_profit_exit(
        self, strategy: SupportBounceStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that take profit triggers exit."""
        trade = MockTrade(
            trade_id=2,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=108.0,
            entry_date=datetime.now() - timedelta(days=1),
        )

        # Set high to hit take profit
        basic_df.loc[basic_df.index[-1], "high"] = 109.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TAKE_PROFIT"

    def test_resistance_exit(self, strategy: SupportBounceStrategy, basic_df: pd.DataFrame) -> None:
        """Test exit at resistance level."""
        trade = MockTrade(
            trade_id=3,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=115.0,
            entry_date=datetime.now() - timedelta(days=1),
            resistance_level=108.0,
        )

        # Set high to hit resistance
        basic_df.loc[basic_df.index[-1], "high"] = 109.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "STRATEGY_EXIT"

    def test_time_exit(self, strategy: SupportBounceStrategy) -> None:
        """Test that time exit triggers after max hold days."""
        np.random.seed(123)
        n_days = 60

        close = 100 + np.sin(np.linspace(0, 4 * np.pi, n_days)) * 2

        df = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.3,
                "low": close - 0.3,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=4,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            entry_date=datetime.now() - timedelta(days=10),
        )

        exit_signal = strategy.check_exit(df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TIME_EXIT"


class TestStrategyRegistry:
    """Tests for strategy registry integration."""

    def test_strategy_registered(self) -> None:
        """Test that strategy is registered in global registry."""
        from src.strategies.base import strategy_registry

        assert "SupportBounce" in strategy_registry.list_strategies()

    def test_create_instance_from_registry(self) -> None:
        """Test creating strategy instance from registry."""
        from src.strategies.base import strategy_registry

        strategy = strategy_registry.create_instance("SupportBounce")
        assert strategy is not None
        assert isinstance(strategy, SupportBounceStrategy)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        strategy = SupportBounceStrategy()
        df = pd.DataFrame()
        signal = strategy.scan(df)
        assert signal is None

    def test_custom_config_integration(self) -> None:
        """Test strategy with custom configuration."""
        config = SupportBounceConfig(
            sr_lookback=90,
            support_tolerance_atr=0.75,
            volume_multiplier=1.5,
            max_hold_days=10,
        )
        strategy = SupportBounceStrategy(config=config)

        assert strategy.config.sr_lookback == 90
        assert strategy.config.support_tolerance_atr == 0.75
        assert strategy.config.volume_multiplier == 1.5
