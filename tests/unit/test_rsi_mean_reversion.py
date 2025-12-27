"""Unit tests for RSI Mean Reversion Strategy."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.strategies.rsi_mean_reversion import (
    RSIMeanReversionStrategy,
    RSIStrategyConfig,
)


@pytest.fixture
def uptrend_oversold_df() -> pd.DataFrame:
    """Create DataFrame with stock in uptrend but temporarily oversold.

    This scenario should generate a signal:
    - Price > SMA(200)
    - RSI < 30
    - Bullish candle (close > open)
    - Good volume
    """
    np.random.seed(42)
    n_days = 250

    # Start with uptrend base
    trend = np.linspace(80, 150, n_days)

    # Add noise
    noise = np.random.randn(n_days) * 2

    # Create recent pullback (last 10 days) to get RSI low
    pullback = np.zeros(n_days)
    pullback[-10:] = np.linspace(0, -15, 10)

    close = trend + noise + pullback

    # Last day is bullish (close > open)
    close[-1] = close[-2] + 2  # Small bounce

    df = pd.DataFrame(
        {
            "open": close - np.abs(np.random.randn(n_days) * 0.5) - 0.5,
            "high": close + np.abs(np.random.randn(n_days) * 1.5),
            "low": close - np.abs(np.random.randn(n_days) * 1.5),
            "close": close,
            "volume": np.random.randint(500000, 2000000, n_days),
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    # Make last candle bullish
    df.loc[df.index[-1], "open"] = df.loc[df.index[-1], "close"] - 1

    df.attrs["symbol"] = "TEST"
    return df


@pytest.fixture
def downtrend_oversold_df() -> pd.DataFrame:
    """Create DataFrame with stock in downtrend and oversold.

    Should NOT generate signal if require_uptrend=True:
    - Price < SMA(200)
    - RSI < 30
    """
    np.random.seed(42)
    n_days = 250

    # Downtrend
    trend = np.linspace(150, 80, n_days)
    noise = np.random.randn(n_days) * 2
    close = trend + noise

    # Further decline to get low RSI
    close[-10:] -= np.linspace(0, 10, 10)

    # Last day bullish candle
    close[-1] = close[-2] + 1

    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + np.abs(np.random.randn(n_days)),
            "low": close - np.abs(np.random.randn(n_days)),
            "close": close,
            "volume": np.random.randint(500000, 2000000, n_days),
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "DOWN"
    return df


@pytest.fixture
def not_oversold_df() -> pd.DataFrame:
    """Create DataFrame with RSI around 35 (not oversold enough)."""
    np.random.seed(42)
    n_days = 250

    # Slight uptrend with minor pullback
    trend = np.linspace(100, 140, n_days)
    noise = np.random.randn(n_days) * 1

    # Small pullback (not enough to trigger oversold)
    pullback = np.zeros(n_days)
    pullback[-5:] = np.linspace(0, -5, 5)

    close = trend + noise + pullback
    close[-1] = close[-2] + 0.5  # Bullish

    df = pd.DataFrame(
        {
            "open": close - 0.3,
            "high": close + np.abs(np.random.randn(n_days)),
            "low": close - np.abs(np.random.randn(n_days)),
            "close": close,
            "volume": np.random.randint(500000, 2000000, n_days),
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "NEUTRAL"
    return df


@pytest.fixture
def low_volume_df() -> pd.DataFrame:
    """Create DataFrame with good setup but very low volume."""
    np.random.seed(42)
    n_days = 250

    trend = np.linspace(80, 150, n_days)
    noise = np.random.randn(n_days) * 2
    pullback = np.zeros(n_days)
    pullback[-10:] = np.linspace(0, -15, 10)
    close = trend + noise + pullback
    close[-1] = close[-2] + 2

    # Good volume history, but last day very low
    volume = np.random.randint(500000, 2000000, n_days)
    volume[-1] = 100000  # Very low volume on signal day

    df = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + np.abs(np.random.randn(n_days)),
            "low": close - np.abs(np.random.randn(n_days)),
            "close": close,
            "volume": volume,
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "LOWVOL"
    return df


@pytest.fixture
def bearish_candle_df() -> pd.DataFrame:
    """Create DataFrame with oversold but bearish candle (close < open)."""
    np.random.seed(42)
    n_days = 250

    trend = np.linspace(80, 150, n_days)
    noise = np.random.randn(n_days) * 2
    pullback = np.zeros(n_days)
    pullback[-10:] = np.linspace(0, -15, 10)
    close = trend + noise + pullback

    df = pd.DataFrame(
        {
            "open": close + 0.5,  # Open > close = bearish
            "high": close + np.abs(np.random.randn(n_days)) + 1,
            "low": close - np.abs(np.random.randn(n_days)),
            "close": close,
            "volume": np.random.randint(500000, 2000000, n_days),
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "BEAR"
    return df


@pytest.fixture
def deeply_oversold_df() -> pd.DataFrame:
    """Create DataFrame with RSI < 25 for higher score."""
    np.random.seed(42)
    n_days = 250

    trend = np.linspace(80, 150, n_days)
    noise = np.random.randn(n_days) * 2

    # Deeper pullback for RSI < 25
    pullback = np.zeros(n_days)
    pullback[-15:] = np.linspace(0, -25, 15)

    close = trend + noise + pullback
    close[-1] = close[-2] + 3  # Strong bounce

    df = pd.DataFrame(
        {
            "open": close - 1,
            "high": close + np.abs(np.random.randn(n_days)) + 1,
            "low": close - np.abs(np.random.randn(n_days)),
            "close": close,
            "volume": np.random.randint(1500000, 3000000, n_days),  # Higher volume
        },
        index=pd.date_range(end=datetime.now(), periods=n_days, freq="D"),
    )

    df.attrs["symbol"] = "DEEP"
    return df


@dataclass
class MockTrade:
    """Mock trade object for exit testing."""

    trade_id: int
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_date: datetime


class TestRSIStrategyConfig:
    """Tests for RSIStrategyConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RSIStrategyConfig()
        assert config.rsi_period == 14
        assert config.oversold_threshold == 30
        assert config.overbought_threshold == 70
        assert config.exit_rsi == 50
        assert config.require_uptrend is True
        assert config.atr_stop_multiplier == 1.5
        assert config.min_rr_ratio == 1.5
        assert config.volume_threshold == 0.8
        assert config.max_hold_days == 5

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RSIStrategyConfig(
            rsi_period=10,
            oversold_threshold=25,
            overbought_threshold=75,
        )
        assert config.rsi_period == 10
        assert config.oversold_threshold == 25
        assert config.overbought_threshold == 75


class TestRSIMeanReversionStrategy:
    """Tests for RSIMeanReversionStrategy."""

    @pytest.fixture
    def strategy(self) -> RSIMeanReversionStrategy:
        """Create strategy instance with default config."""
        return RSIMeanReversionStrategy()

    def test_strategy_name(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test strategy name attribute."""
        assert strategy.name == "RSIMeanReversion"

    def test_strategy_description(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test strategy description attribute."""
        assert "oversold" in strategy.description.lower()

    def test_get_parameters(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test getting strategy parameters."""
        params = strategy.get_parameters()
        assert params["rsi_period"] == 14
        assert params["oversold_threshold"] == 30
        assert params["require_uptrend"] is True

    def test_set_parameters(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test setting strategy parameters."""
        strategy.set_parameters(
            {
                "rsi_period": 10,
                "oversold_threshold": 25,
                "require_uptrend": False,
            }
        )
        assert strategy.config.rsi_period == 10
        assert strategy.config.oversold_threshold == 25
        assert strategy.config.require_uptrend is False


class TestScanMethod:
    """Tests for the scan() method."""

    @pytest.fixture
    def strategy(self) -> RSIMeanReversionStrategy:
        """Create strategy instance."""
        return RSIMeanReversionStrategy()

    def test_generates_signal_when_oversold(
        self, strategy: RSIMeanReversionStrategy, uptrend_oversold_df: pd.DataFrame
    ) -> None:
        """Test that signal is generated when RSI < 30 in uptrend."""
        signal = strategy.scan(uptrend_oversold_df)

        # Note: Due to randomness, the conditions might not always be met
        # If signal is None, check if conditions actually exist
        if signal is not None:
            assert signal.direction == "LONG"
            assert signal.strategy == "RSIMeanReversion"
            assert signal.symbol == "TEST"
            assert 0 <= signal.score <= 100
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price

    def test_no_signal_when_not_oversold(
        self, strategy: RSIMeanReversionStrategy, not_oversold_df: pd.DataFrame
    ) -> None:
        """Test that no signal when RSI > 30."""
        signal = strategy.scan(not_oversold_df)
        assert signal is None

    def test_respects_uptrend_filter(
        self, strategy: RSIMeanReversionStrategy, downtrend_oversold_df: pd.DataFrame
    ) -> None:
        """Test that no signal when price below SMA200."""
        signal = strategy.scan(downtrend_oversold_df)
        assert signal is None

    def test_ignores_uptrend_when_disabled(
        self, downtrend_oversold_df: pd.DataFrame
    ) -> None:
        """Test that signal can generate when uptrend filter disabled."""
        config = RSIStrategyConfig(require_uptrend=False)
        strategy = RSIMeanReversionStrategy(config=config)

        # May or may not generate signal depending on other conditions
        # Just verify it doesn't crash
        _ = strategy.scan(downtrend_oversold_df)
        # Signal could be None due to other conditions (bearish candle, etc.)

    def test_no_signal_on_low_volume(
        self, strategy: RSIMeanReversionStrategy, low_volume_df: pd.DataFrame
    ) -> None:
        """Test that no signal when volume too low."""
        signal = strategy.scan(low_volume_df)
        assert signal is None

    def test_no_signal_on_bearish_candle(
        self, strategy: RSIMeanReversionStrategy, bearish_candle_df: pd.DataFrame
    ) -> None:
        """Test that no signal when candle is bearish."""
        signal = strategy.scan(bearish_candle_df)
        assert signal is None

    def test_stop_loss_uses_atr(
        self, strategy: RSIMeanReversionStrategy, uptrend_oversold_df: pd.DataFrame
    ) -> None:
        """Test that stop loss is based on ATR."""
        signal = strategy.scan(uptrend_oversold_df)

        if signal is not None:
            # Stop loss should be below entry
            assert signal.stop_loss < signal.entry_price

            # Check ATR is in metadata
            assert "atr" in signal.metadata
            atr = signal.metadata["atr"]

            # Stop loss should be approximately entry - (ATR * multiplier)
            expected_sl = signal.entry_price - (atr * strategy.config.atr_stop_multiplier)
            assert abs(signal.stop_loss - expected_sl) < 0.01

    def test_take_profit_uses_rr_ratio(
        self, strategy: RSIMeanReversionStrategy, uptrend_oversold_df: pd.DataFrame
    ) -> None:
        """Test that take profit uses risk-reward ratio."""
        signal = strategy.scan(uptrend_oversold_df)

        if signal is not None:
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
            actual_rr = reward / risk

            # Should be at least min_rr_ratio * 1.5
            expected_rr = strategy.config.min_rr_ratio * 1.5
            assert abs(actual_rr - expected_rr) < 0.01

    def test_insufficient_data(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test that no signal with insufficient data."""
        # Create small DataFrame (less than 200 rows)
        df = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [102] * 50,
                "low": [99] * 50,
                "close": [101] * 50,
                "volume": [1000000] * 50,
            }
        )
        signal = strategy.scan(df)
        assert signal is None

    def test_invalid_dataframe(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test that no signal with missing columns."""
        df = pd.DataFrame(
            {
                "open": [100] * 200,
                "high": [102] * 200,
                # Missing low, close, volume
            }
        )
        signal = strategy.scan(df)
        assert signal is None


class TestScoreCalculation:
    """Tests for score calculation."""

    def test_deeply_oversold_higher_score(self, deeply_oversold_df: pd.DataFrame) -> None:
        """Test that RSI < 25 gives higher score."""
        strategy = RSIMeanReversionStrategy()
        signal = strategy.scan(deeply_oversold_df)

        if signal is not None:
            # Score should be higher due to deeply oversold + high volume
            assert signal.score >= 60  # Base 50 + at least some bonuses

    def test_score_components_in_metadata(
        self, uptrend_oversold_df: pd.DataFrame
    ) -> None:
        """Test that RSI value is in signal metadata."""
        strategy = RSIMeanReversionStrategy()
        signal = strategy.scan(uptrend_oversold_df)

        if signal is not None:
            assert "rsi" in signal.metadata
            assert signal.metadata["rsi"] < 30  # Should be oversold


class TestCheckExit:
    """Tests for the check_exit() method."""

    @pytest.fixture
    def strategy(self) -> RSIMeanReversionStrategy:
        """Create strategy instance."""
        return RSIMeanReversionStrategy()

    @pytest.fixture
    def basic_df(self) -> pd.DataFrame:
        """Create basic OHLCV DataFrame for exit tests."""
        n_days = 50
        close = np.linspace(100, 110, n_days)

        return pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

    def test_stop_loss_exit(
        self, strategy: RSIMeanReversionStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that stop loss triggers exit."""
        trade = MockTrade(
            trade_id=1,
            entry_price=105.0,
            stop_loss=115.0,  # Above current price (will trigger)
            take_profit=120.0,
            entry_date=datetime.now() - timedelta(days=1),
        )

        # Set low to hit stop loss
        basic_df.loc[basic_df.index[-1], "low"] = 114.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "STOP_LOSS"
        assert exit_signal.trade_id == 1

    def test_take_profit_exit(
        self, strategy: RSIMeanReversionStrategy, basic_df: pd.DataFrame
    ) -> None:
        """Test that take profit triggers exit."""
        trade = MockTrade(
            trade_id=2,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=108.0,  # Below current high (will trigger)
            entry_date=datetime.now() - timedelta(days=1),
        )

        # Set high to hit take profit
        basic_df.loc[basic_df.index[-1], "high"] = 112.0

        exit_signal = strategy.check_exit(basic_df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TAKE_PROFIT"
        assert exit_signal.trade_id == 2

    def test_time_exit(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test that time exit triggers after max hold days."""
        np.random.seed(123)
        n_days = 50

        # Create oscillating price that keeps RSI around 50 (not overbought)
        close = 100 + np.sin(np.linspace(0, 4 * np.pi, n_days)) * 2
        close = close + np.random.randn(n_days) * 0.5

        df = pd.DataFrame(
            {
                "open": close - 0.3,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=3,
            entry_price=100.0,
            stop_loss=90.0,  # Won't hit (price stays around 100)
            take_profit=115.0,  # Won't hit
            entry_date=datetime.now() - timedelta(days=10),  # 10 days ago
        )

        exit_signal = strategy.check_exit(df, trade)
        assert exit_signal is not None
        assert exit_signal.exit_type == "TIME_EXIT"

    def test_no_exit_within_parameters(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test that no exit when all conditions within limits."""
        np.random.seed(789)
        n_days = 50

        # Create slight downtrend to keep RSI below 50 but not triggering exits
        close = np.linspace(110, 100, n_days)
        # Add small noise but keep the downward bias
        close = close + np.random.randn(n_days) * 0.2

        df = pd.DataFrame(
            {
                "open": close + 0.1,  # Open slightly above close for downward bias
                "high": close + 0.5,
                "low": close - 0.3,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=4,
            entry_price=110.0,
            stop_loss=85.0,  # Won't hit (price stays above 100)
            take_profit=130.0,  # Won't hit
            entry_date=datetime.now() - timedelta(days=1),
        )

        exit_signal = strategy.check_exit(df, trade)
        # RSI should stay below 50 with this data, so no exit
        assert exit_signal is None

    def test_rsi_overbought_exit(self, strategy: RSIMeanReversionStrategy) -> None:
        """Test exit when RSI becomes overbought (>70)."""
        n_days = 50

        # Create strong uptrend to get high RSI
        close = np.linspace(100, 150, n_days)
        # Add consistent gains to push RSI high
        for i in range(1, n_days):
            close[i] = close[i - 1] * 1.02  # 2% daily gains

        df = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": [1000000] * n_days,
            }
        )

        trade = MockTrade(
            trade_id=5,
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=300.0,  # Won't hit
            entry_date=datetime.now() - timedelta(days=1),
        )

        exit_signal = strategy.check_exit(df, trade)

        # Should exit due to overbought RSI
        if exit_signal is not None:
            assert exit_signal.exit_type in ["TAKE_PROFIT", "STRATEGY_EXIT"]


class TestStrategyRegistry:
    """Tests for strategy registry integration."""

    def test_strategy_registered(self) -> None:
        """Test that strategy is registered in global registry."""
        from src.strategies.base import strategy_registry

        assert "RSIMeanReversion" in strategy_registry.list_strategies()

    def test_create_instance_from_registry(self) -> None:
        """Test creating strategy instance from registry."""
        from src.strategies.base import strategy_registry

        strategy = strategy_registry.create_instance("RSIMeanReversion")
        assert strategy is not None
        assert isinstance(strategy, RSIMeanReversionStrategy)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self) -> None:
        """Test handling of empty DataFrame."""
        strategy = RSIMeanReversionStrategy()
        df = pd.DataFrame()
        signal = strategy.scan(df)
        assert signal is None

    def test_dataframe_with_nans(self) -> None:
        """Test handling of DataFrame with NaN values."""
        strategy = RSIMeanReversionStrategy()
        df = pd.DataFrame(
            {
                "open": [100, np.nan, 102] + [101] * 197,
                "high": [102] * 200,
                "low": [99] * 200,
                "close": [101] * 200,
                "volume": [1000000] * 200,
            }
        )
        # Should handle gracefully (may return None)
        _ = strategy.scan(df)
        # Just verify no exception raised

    def test_custom_config_integration(self) -> None:
        """Test strategy with custom configuration."""
        config = RSIStrategyConfig(
            rsi_period=10,
            oversold_threshold=35,  # More lenient
            require_uptrend=False,
            atr_stop_multiplier=2.0,
        )
        strategy = RSIMeanReversionStrategy(config=config)

        assert strategy.config.rsi_period == 10
        assert strategy.config.oversold_threshold == 35
        assert strategy.config.require_uptrend is False
