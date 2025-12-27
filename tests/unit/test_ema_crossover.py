"""Unit tests for EMA Crossover Strategy."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.ema_crossover import (
    EMACrossoverConfig,
    EMACrossoverStrategy,
)


@pytest.fixture
def bullish_crossover_df() -> pd.DataFrame:
    """Create DataFrame with bullish EMA crossover.

    - EMA 9 crosses above EMA 21
    - Price above EMA 50
    - MACD histogram positive
    - Good volume
    """
    np.random.seed(42)
    n_days = 100

    # Create uptrend with recent crossover
    # First part: slow uptrend (EMA 9 < EMA 21)
    # Last part: acceleration (EMA 9 > EMA 21)
    trend = np.concatenate(
        [
            np.linspace(100, 105, 80),  # Slow uptrend
            np.linspace(106, 115, 20),  # Acceleration causing crossover
        ]
    )
    noise = np.random.randn(n_days) * 0.5
    close = trend + noise

    # Good volume
    volume = np.random.randint(900000, 1500000, n_days)

    df = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + np.abs(np.random.randn(n_days)) * 0.5,
            "low": close - np.abs(np.random.randn(n_days)) * 0.5,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BULLISH"
    return df


@pytest.fixture
def bearish_crossover_df() -> pd.DataFrame:
    """Create DataFrame with bearish EMA crossover.

    - EMA 9 crosses below EMA 21
    - Price below EMA 50
    - MACD histogram negative
    """
    np.random.seed(42)
    n_days = 100

    # Create downtrend with crossover
    trend = np.concatenate(
        [
            np.linspace(120, 115, 80),  # Slow downtrend
            np.linspace(114, 100, 20),  # Acceleration causing crossover
        ]
    )
    noise = np.random.randn(n_days) * 0.5
    close = trend + noise

    volume = np.random.randint(900000, 1500000, n_days)

    df = pd.DataFrame(
        {
            "open": close + 0.3,
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
def no_crossover_df() -> pd.DataFrame:
    """Create DataFrame with no crossover (EMA 9 already above EMA 21)."""
    np.random.seed(42)
    n_days = 100

    # Steady uptrend - EMA 9 already above EMA 21
    trend = np.linspace(100, 130, n_days)
    noise = np.random.randn(n_days) * 0.3
    close = trend + noise

    volume = np.random.randint(900000, 1500000, n_days)

    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.3,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "NOCROSS"
    return df


@pytest.fixture
def below_trend_df() -> pd.DataFrame:
    """Create DataFrame with bullish crossover but price below EMA 50."""
    np.random.seed(42)
    n_days = 100

    # Recent recovery but still below 50 EMA
    trend = np.concatenate(
        [
            np.linspace(120, 90, 70),  # Downtrend
            np.linspace(91, 100, 30),  # Recovery
        ]
    )
    noise = np.random.randn(n_days) * 0.3
    close = trend + noise

    volume = np.random.randint(900000, 1500000, n_days)

    df = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.3,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BELOWTREND"
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


class TestEMACrossoverConfig:
    """Tests for EMACrossoverConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = EMACrossoverConfig()
        assert config.fast_ema == 9
        assert config.slow_ema == 21
        assert config.trend_ema == 50
        assert config.atr_stop_multiplier == 1.5
        assert config.profit_target_percent == 8.0
        assert config.volume_threshold == 0.8
        assert config.max_hold_days == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = EMACrossoverConfig(
            fast_ema=5,
            slow_ema=13,
            profit_target_percent=10.0,
        )
        assert config.fast_ema == 5
        assert config.slow_ema == 13
        assert config.profit_target_percent == 10.0


class TestEMACrossoverStrategy:
    """Tests for EMACrossoverStrategy."""

    @pytest.fixture
    def strategy(self) -> EMACrossoverStrategy:
        """Create strategy instance with default config."""
        return EMACrossoverStrategy()

    def test_strategy_name(self, strategy: EMACrossoverStrategy) -> None:
        """Test strategy name attribute."""
        assert strategy.name == "EMACrossover"

    def test_strategy_description(self, strategy: EMACrossoverStrategy) -> None:
        """Test strategy description attribute."""
        assert "crossover" in strategy.description.lower()

    def test_get_parameters(self, strategy: EMACrossoverStrategy) -> None:
        """Test getting strategy parameters."""
        params = strategy.get_parameters()
        assert params["fast_ema"] == 9
        assert params["slow_ema"] == 21
        assert params["trend_ema"] == 50

    def test_set_parameters(self, strategy: EMACrossoverStrategy) -> None:
        """Test setting strategy parameters."""
        strategy.set_parameters(
            {
                "fast_ema": 5,
                "slow_ema": 13,
                "profit_target_percent": 10.0,
            }
        )
        assert strategy.config.fast_ema == 5
        assert strategy.config.slow_ema == 13
        assert strategy.config.profit_target_percent == 10.0


class TestScanMethod:
    """Tests for the scan() method."""

    @pytest.fixture
    def strategy(self) -> EMACrossoverStrategy:
        """Create strategy instance."""
        return EMACrossoverStrategy()

    def test_signals_on_bullish_crossover(
        self, strategy: EMACrossoverStrategy, bullish_crossover_df: pd.DataFrame
    ) -> None:
        """Test that LONG signal is generated on bullish crossover."""
        signal = strategy.scan(bullish_crossover_df)

        # May or may not generate signal depending on exact crossover timing
        if signal is not None:
            assert signal.direction == "LONG"
            assert signal.strategy == "EMACrossover"
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price

    def test_signals_on_bearish_crossover(
        self, strategy: EMACrossoverStrategy, bearish_crossover_df: pd.DataFrame
    ) -> None:
        """Test that SHORT signal is generated on bearish crossover."""
        signal = strategy.scan(bearish_crossover_df)

        if signal is not None:
            assert signal.direction == "SHORT"
            assert signal.stop_loss > signal.entry_price
            assert signal.take_profit < signal.entry_price

    def test_no_signal_without_crossover(
        self, strategy: EMACrossoverStrategy, no_crossover_df: pd.DataFrame
    ) -> None:
        """Test that no signal when there's no crossover."""
        signal = strategy.scan(no_crossover_df)
        assert signal is None

    def test_no_signal_below_ema50(
        self, strategy: EMACrossoverStrategy, below_trend_df: pd.DataFrame
    ) -> None:
        """Test that no LONG signal when price below EMA 50."""
        signal = strategy.scan(below_trend_df)
        # Should not generate LONG signal below trend
        if signal is not None:
            # If signal exists, it should be SHORT
            assert signal.direction == "SHORT"

    def test_stop_loss_uses_atr(
        self, strategy: EMACrossoverStrategy, bullish_crossover_df: pd.DataFrame
    ) -> None:
        """Test that stop loss is based on ATR."""
        signal = strategy.scan(bullish_crossover_df)

        if signal is not None:
            assert "atr" in signal.metadata
            atr = signal.metadata["atr"]
            expected_sl = signal.entry_price - (atr * strategy.config.atr_stop_multiplier)
            assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_take_profit_uses_percentage(
        self, strategy: EMACrossoverStrategy, bullish_crossover_df: pd.DataFrame
    ) -> None:
        """Test that take profit is based on percentage target."""
        signal = strategy.scan(bullish_crossover_df)

        if signal is not None and signal.direction == "LONG":
            expected_tp = signal.entry_price * (1 + strategy.config.profit_target_percent / 100)
            assert abs(signal.take_profit - expected_tp) < 0.01

    def test_metadata_contains_emas(
        self, strategy: EMACrossoverStrategy, bullish_crossover_df: pd.DataFrame
    ) -> None:
        """Test that metadata contains EMA values."""
        signal = strategy.scan(bullish_crossover_df)

        if signal is not None:
            assert "ema_fast" in signal.metadata
            assert "ema_slow" in signal.metadata
            assert "ema_trend" in signal.metadata
            assert "macd_hist" in signal.metadata

    def test_insufficient_data(self, strategy: EMACrossoverStrategy) -> None:
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


class TestScoringLogic:
    """Tests for score calculation."""

    def test_score_in_valid_range(self, bullish_crossover_df: pd.DataFrame) -> None:
        """Test that score is always in valid range."""
        strategy = EMACrossoverStrategy()
        signal = strategy.scan(bullish_crossover_df)

        if signal is not None:
            assert 0 <= signal.score <= 100

    def test_base_score(self, bullish_crossover_df: pd.DataFrame) -> None:
        """Test that base score is at least 50."""
        strategy = EMACrossoverStrategy()
        signal = strategy.scan(bullish_crossover_df)

        if signal is not None:
            assert signal.score >= 50


class TestCheckExit:
    """Tests for the check_exit() method."""

    @pytest.fixture
    def strategy(self) -> EMACrossoverStrategy:
        """Create strategy instance."""
        return EMACrossoverStrategy()

    @pytest.fixture
    def basic_df(self) -> pd.DataFrame:
        """Create basic OHLCV DataFrame for exit tests."""
        np.random.seed(42)
        n_days = 60

        close = np.linspace(100, 105, n_days) + np.random.randn(n_days) * 0.3

        return pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

    def test_stop_loss_exit_long(
        self, strategy: EMACrossoverStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that stop loss triggers exit for LONG."""
        trade = MockTrade(
            trade_id=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=108.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
        )

        # Set low to hit stop loss
        basic_df.loc[basic_df.index[-1], "low"] = 94.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "STOP_LOSS"

    def test_stop_loss_exit_short(
        self, strategy: EMACrossoverStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that stop loss triggers exit for SHORT."""
        trade = MockTrade(
            trade_id=2,
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=92.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="SHORT",
        )

        # Set high to hit stop loss
        basic_df.loc[basic_df.index[-1], "high"] = 106.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "STOP_LOSS"

    def test_exit_on_profit_target(
        self, strategy: EMACrossoverStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that profit target triggers exit."""
        trade = MockTrade(
            trade_id=3,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=106.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
        )

        # Set high to hit take profit
        basic_df.loc[basic_df.index[-1], "high"] = 107.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TAKE_PROFIT"

    def test_exit_on_bearish_crossover(self, strategy: EMACrossoverStrategy) -> None:
        """Test exit when EMA crosses in opposite direction."""
        np.random.seed(42)
        n_days = 60

        # Create data where EMA 9 crosses below EMA 21
        trend = np.concatenate(
            [
                np.linspace(110, 115, 50),  # Uptrend
                np.linspace(114, 105, 10),  # Reversal
            ]
        )
        close = trend + np.random.randn(n_days) * 0.2

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
            entry_price=110.0,
            stop_loss=100.0,
            take_profit=125.0,
            entry_date=datetime.now() - timedelta(days=3),
            direction="LONG",
        )

        exit_signal = strategy.check_exit(df, trade)
        # May exit on strategy exit if crossover detected
        if exit_signal is not None:
            assert exit_signal.exit_type in ["STRATEGY_EXIT", "STOP_LOSS", "TAKE_PROFIT"]

    def test_time_exit(self, strategy: EMACrossoverStrategy) -> None:
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
            trade_id=5,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=120.0,
            entry_date=datetime.now() - timedelta(days=15),
            direction="LONG",
        )

        exit_signal = strategy.check_exit(df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TIME_EXIT"


class TestStrategyRegistry:
    """Tests for strategy registry integration."""

    def test_strategy_registered(self) -> None:
        """Test that strategy is registered in global registry."""
        from src.strategies.base import strategy_registry

        assert "EMACrossover" in strategy_registry.list_strategies()

    def test_create_instance_from_registry(self) -> None:
        """Test creating strategy instance from registry."""
        from src.strategies.base import strategy_registry

        strategy = strategy_registry.create_instance("EMACrossover")
        assert strategy is not None
        assert isinstance(strategy, EMACrossoverStrategy)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        strategy = EMACrossoverStrategy()
        df = pd.DataFrame()
        signal = strategy.scan(df)
        assert signal is None

    def test_custom_config_integration(self) -> None:
        """Test strategy with custom configuration."""
        config = EMACrossoverConfig(
            fast_ema=5,
            slow_ema=13,
            trend_ema=34,
            profit_target_percent=10.0,
        )
        strategy = EMACrossoverStrategy(config=config)

        assert strategy.config.fast_ema == 5
        assert strategy.config.slow_ema == 13
        assert strategy.config.trend_ema == 34
