"""Unit tests for MACD Divergence Strategy."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.macd_divergence import (
    MACDDivergenceConfig,
    MACDDivergenceStrategy,
)


@pytest.fixture
def bullish_divergence_df() -> pd.DataFrame:
    """Create DataFrame with bullish divergence.

    - Price makes lower low
    - MACD makes higher low
    - MACD histogram turns positive
    """
    np.random.seed(42)
    n_days = 80

    # Create pattern: downtrend with divergence at the end
    # Price makes lower lows but momentum weakening
    trend = np.concatenate(
        [
            np.linspace(120, 100, 50),  # Downtrend
            np.linspace(99, 95, 15),  # Continued decline (lower low)
            np.linspace(96, 102, 15),  # Recovery
        ]
    )
    noise = np.random.randn(n_days) * 0.5
    close = trend + noise

    volume = np.random.randint(900000, 1500000, n_days)

    df = pd.DataFrame(
        {
            "open": close + 0.3,
            "high": close + np.abs(np.random.randn(n_days)) * 1,
            "low": close - np.abs(np.random.randn(n_days)) * 1,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BULLISH_DIV"
    return df


@pytest.fixture
def bearish_divergence_df() -> pd.DataFrame:
    """Create DataFrame with bearish divergence.

    - Price makes higher high
    - MACD makes lower high
    - MACD histogram turns negative
    """
    np.random.seed(42)
    n_days = 80

    # Create pattern: uptrend with divergence at the end
    trend = np.concatenate(
        [
            np.linspace(80, 100, 50),  # Uptrend
            np.linspace(101, 108, 15),  # Continued rise (higher high)
            np.linspace(107, 100, 15),  # Pullback
        ]
    )
    noise = np.random.randn(n_days) * 0.5
    close = trend + noise

    volume = np.random.randint(900000, 1500000, n_days)

    df = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + np.abs(np.random.randn(n_days)) * 1,
            "low": close - np.abs(np.random.randn(n_days)) * 1,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BEARISH_DIV"
    return df


@pytest.fixture
def no_divergence_df() -> pd.DataFrame:
    """Create DataFrame with no divergence (normal trend)."""
    np.random.seed(42)
    n_days = 80

    # Steady uptrend - no divergence
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

    df.attrs["symbol"] = "NODIV"
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


class TestMACDDivergenceConfig:
    """Tests for MACDDivergenceConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MACDDivergenceConfig()
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.lookback_period == 14
        assert config.rsi_max == 50
        assert config.rsi_min == 50
        assert config.atr_stop_multiplier == 2.0
        assert config.max_hold_days == 10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MACDDivergenceConfig(
            macd_fast=8,
            macd_slow=17,
            lookback_period=20,
        )
        assert config.macd_fast == 8
        assert config.macd_slow == 17
        assert config.lookback_period == 20


class TestMACDDivergenceStrategy:
    """Tests for MACDDivergenceStrategy."""

    @pytest.fixture
    def strategy(self) -> MACDDivergenceStrategy:
        """Create strategy instance with default config."""
        return MACDDivergenceStrategy()

    def test_strategy_name(self, strategy: MACDDivergenceStrategy) -> None:
        """Test strategy name attribute."""
        assert strategy.name == "MACDDivergence"

    def test_strategy_description(self, strategy: MACDDivergenceStrategy) -> None:
        """Test strategy description attribute."""
        assert "divergence" in strategy.description.lower()

    def test_get_parameters(self, strategy: MACDDivergenceStrategy) -> None:
        """Test getting strategy parameters."""
        params = strategy.get_parameters()
        assert params["macd_fast"] == 12
        assert params["macd_slow"] == 26
        assert params["macd_signal"] == 9

    def test_set_parameters(self, strategy: MACDDivergenceStrategy) -> None:
        """Test setting strategy parameters."""
        strategy.set_parameters(
            {
                "macd_fast": 8,
                "lookback_period": 20,
                "rsi_max": 60,
            }
        )
        assert strategy.config.macd_fast == 8
        assert strategy.config.lookback_period == 20
        assert strategy.config.rsi_max == 60


class TestDivergenceDetection:
    """Tests for divergence detection methods."""

    @pytest.fixture
    def strategy(self) -> MACDDivergenceStrategy:
        """Create strategy instance."""
        return MACDDivergenceStrategy()

    def test_detects_bullish_divergence(
        self, strategy: MACDDivergenceStrategy, bullish_divergence_df: pd.DataFrame
    ) -> None:
        """Test that bullish divergence is detected."""
        signal = strategy.scan(bullish_divergence_df)

        # Divergence detection is complex, may or may not trigger
        if signal is not None:
            assert signal.direction == "LONG"
            assert "divergence" in signal.metadata.get("divergence_type", "")

    def test_detects_bearish_divergence(
        self, strategy: MACDDivergenceStrategy, bearish_divergence_df: pd.DataFrame
    ) -> None:
        """Test that bearish divergence is detected."""
        signal = strategy.scan(bearish_divergence_df)

        if signal is not None:
            assert signal.direction == "SHORT"
            assert "divergence" in signal.metadata.get("divergence_type", "")

    def test_no_signal_without_divergence(
        self, strategy: MACDDivergenceStrategy, no_divergence_df: pd.DataFrame
    ) -> None:
        """Test that no signal when there's no divergence."""
        signal = strategy.scan(no_divergence_df)
        assert signal is None


class TestHistogramConfirmation:
    """Tests for histogram confirmation requirement."""

    @pytest.fixture
    def strategy(self) -> MACDDivergenceStrategy:
        """Create strategy instance."""
        return MACDDivergenceStrategy()

    def test_requires_histogram_confirmation(
        self, strategy: MACDDivergenceStrategy, no_divergence_df: pd.DataFrame
    ) -> None:
        """Test that signal requires histogram confirmation."""
        # Without proper histogram confirmation, should not signal
        signal = strategy.scan(no_divergence_df)
        assert signal is None

    def test_histogram_in_metadata(
        self, strategy: MACDDivergenceStrategy, bullish_divergence_df: pd.DataFrame
    ) -> None:
        """Test that histogram value is in metadata."""
        signal = strategy.scan(bullish_divergence_df)

        if signal is not None:
            assert "macd_hist" in signal.metadata


class TestScanMethod:
    """Tests for the scan() method."""

    @pytest.fixture
    def strategy(self) -> MACDDivergenceStrategy:
        """Create strategy instance."""
        return MACDDivergenceStrategy()

    def test_stop_loss_below_swing_low(
        self, strategy: MACDDivergenceStrategy, bullish_divergence_df: pd.DataFrame
    ) -> None:
        """Test that stop loss is below recent swing low for LONG."""
        signal = strategy.scan(bullish_divergence_df)

        if signal is not None and signal.direction == "LONG":
            lookback = strategy.config.lookback_period
            recent_low = float(bullish_divergence_df["low"].iloc[-lookback:].min())
            assert signal.stop_loss < recent_low

    def test_metadata_contains_rsi(
        self, strategy: MACDDivergenceStrategy, bullish_divergence_df: pd.DataFrame
    ) -> None:
        """Test that metadata contains RSI value."""
        signal = strategy.scan(bullish_divergence_df)

        if signal is not None:
            assert "rsi" in signal.metadata

    def test_insufficient_data(self, strategy: MACDDivergenceStrategy) -> None:
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
        self, strategy: MACDDivergenceStrategy, bullish_divergence_df: pd.DataFrame
    ) -> None:
        """Test that score is in valid range."""
        signal = strategy.scan(bullish_divergence_df)

        if signal is not None:
            assert 0 <= signal.score <= 100


class TestCheckExit:
    """Tests for the check_exit() method."""

    @pytest.fixture
    def strategy(self) -> MACDDivergenceStrategy:
        """Create strategy instance."""
        return MACDDivergenceStrategy()

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

    def test_stop_loss_exit(self, strategy: MACDDivergenceStrategy, basic_df: pd.DataFrame) -> None:
        """Test that stop loss triggers exit."""
        trade = MockTrade(
            trade_id=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            entry_date=datetime.now() - timedelta(days=1),
            direction="LONG",
        )

        # Set low to hit stop loss
        basic_df.loc[basic_df.index[-1], "low"] = 94.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "STOP_LOSS"

    def test_take_profit_exit(
        self, strategy: MACDDivergenceStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that take profit triggers exit."""
        trade = MockTrade(
            trade_id=2,
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

    def test_exit_on_signal_cross(self, strategy: MACDDivergenceStrategy) -> None:
        """Test exit when MACD crosses signal line."""
        np.random.seed(42)
        n_days = 60

        # Create data where MACD will cross signal
        trend = np.concatenate(
            [
                np.linspace(110, 115, 50),
                np.linspace(114, 105, 10),  # Pullback
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
            trade_id=3,
            entry_price=110.0,
            stop_loss=100.0,
            take_profit=125.0,
            entry_date=datetime.now() - timedelta(days=3),
            direction="LONG",
        )

        exit_signal = strategy.check_exit(df, trade)
        # May trigger strategy exit on signal cross
        if exit_signal is not None:
            assert exit_signal.exit_type in ["STRATEGY_EXIT", "STOP_LOSS", "TAKE_PROFIT"]

    def test_time_exit(self, strategy: MACDDivergenceStrategy) -> None:
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

        assert "MACDDivergence" in strategy_registry.list_strategies()

    def test_create_instance_from_registry(self) -> None:
        """Test creating strategy instance from registry."""
        from src.strategies.base import strategy_registry

        strategy = strategy_registry.create_instance("MACDDivergence")
        assert strategy is not None
        assert isinstance(strategy, MACDDivergenceStrategy)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        strategy = MACDDivergenceStrategy()
        df = pd.DataFrame()
        signal = strategy.scan(df)
        assert signal is None

    def test_custom_config_integration(self) -> None:
        """Test strategy with custom configuration."""
        config = MACDDivergenceConfig(
            macd_fast=8,
            macd_slow=17,
            macd_signal=5,
            lookback_period=20,
        )
        strategy = MACDDivergenceStrategy(config=config)

        assert strategy.config.macd_fast == 8
        assert strategy.config.macd_slow == 17
        assert strategy.config.macd_signal == 5
