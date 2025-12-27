"""Unit tests for technical indicators module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.exceptions import InsufficientDataError, InvalidDataError
from src.indicators.technical import TechnicalIndicators

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_ohlcv_df() -> pd.DataFrame:
    """Create simple OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.5 for i in range(50)],
            "high": [102.0 + i * 0.5 for i in range(50)],
            "low": [99.0 + i * 0.5 for i in range(50)],
            "close": [101.0 + i * 0.5 for i in range(50)],
            "volume": [1000000 + i * 10000 for i in range(50)],
        },
        index=dates,
    )


@pytest.fixture
def large_ohlcv_df() -> pd.DataFrame:
    """Create larger OHLCV data (250 rows) for EMA 200 tests."""
    dates = pd.date_range("2023-01-01", periods=250, freq="D")
    np.random.seed(42)

    # Generate more realistic price data with random walk
    close = 100.0
    closes = []
    for _ in range(250):
        close = close * (1 + np.random.normal(0, 0.02))
        closes.append(close)

    closes = np.array(closes)

    return pd.DataFrame(
        {
            "open": closes * 0.995,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.random.randint(500000, 2000000, 250),
        },
        index=dates,
    )


@pytest.fixture
def rsi_test_data() -> pd.DataFrame:
    """Create data specifically for RSI testing with known values."""
    # Use a sequence where we can calculate RSI by hand
    prices = [44, 44.5, 44, 43.5, 44, 44.5, 45, 45.5, 46, 45.5, 45, 45.5, 46, 46.5, 47, 47.5, 48]
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="D")

    return pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000000] * len(prices),
        },
        index=dates,
    )


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """Create data with clear uptrend for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    # Clear uptrend: each day closes higher
    closes = [100 + i * 2 for i in range(50)]

    return pd.DataFrame(
        {
            "open": [c - 1 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 2 for c in closes],
            "close": closes,
            "volume": [1000000] * 50,
        },
        index=dates,
    )


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """Create data with clear downtrend for testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    # Clear downtrend: each day closes lower
    closes = [200 - i * 2 for i in range(50)]

    return pd.DataFrame(
        {
            "open": [c + 1 for c in closes],
            "high": [c + 2 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000000] * 50,
        },
        index=dates,
    )


@pytest.fixture
def volatile_df() -> pd.DataFrame:
    """Create volatile price data for Bollinger Bands testing."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(123)

    # Add significant volatility
    base = 100
    closes = []
    for _ in range(50):
        change = np.random.normal(0, 5)  # High volatility
        base = max(50, base + change)
        closes.append(base)

    return pd.DataFrame(
        {
            "open": [c * 0.99 for c in closes],
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [1000000] * 50,
        },
        index=dates,
    )


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe_raises_error(self) -> None:
        """Empty DataFrame should raise InvalidDataError."""
        df = pd.DataFrame()

        with pytest.raises(InvalidDataError, match="empty"):
            TechnicalIndicators(df)

    def test_missing_columns_raises_error(self) -> None:
        """Missing required columns should raise InvalidDataError."""
        df = pd.DataFrame({"close": [100, 101, 102]})

        with pytest.raises(InvalidDataError, match="Missing required columns"):
            TechnicalIndicators(df)

    def test_all_nan_column_raises_error(self) -> None:
        """Column with all NaN values should raise InvalidDataError."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [float("nan")] * 3,
                "volume": [1000, 1000, 1000],
            }
        )

        with pytest.raises(InvalidDataError, match="all NaN"):
            TechnicalIndicators(df)

    def test_valid_dataframe_accepted(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Valid DataFrame should be accepted without errors."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        assert ti.df is not None
        assert len(ti.df) == 50

    def test_dataframe_is_copied(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Input DataFrame should be copied, not referenced."""
        ti = TechnicalIndicators(simple_ohlcv_df)

        # Modify original
        simple_ohlcv_df.loc[simple_ohlcv_df.index[0], "close"] = 999

        # TechnicalIndicators should have original value
        assert ti.df.loc[ti.df.index[0], "close"] != 999


# ============================================================================
# RSI Tests
# ============================================================================


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """RSI should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        rsi = ti.calculate_rsi()

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(simple_ohlcv_df)

    def test_rsi_values_between_0_and_100(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """RSI values should be between 0 and 100."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        rsi = ti.calculate_rsi()

        # Filter out NaN values for valid range check
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_uptrend_high(self, trending_up_df: pd.DataFrame) -> None:
        """RSI should be high (>70) in strong uptrend."""
        ti = TechnicalIndicators(trending_up_df)
        rsi = ti.calculate_rsi()

        # Last RSI value should be high in uptrend
        assert rsi.iloc[-1] > 70

    def test_rsi_downtrend_low(self, trending_down_df: pd.DataFrame) -> None:
        """RSI should be low (<30) in strong downtrend."""
        ti = TechnicalIndicators(trending_down_df)
        rsi = ti.calculate_rsi()

        # Last RSI value should be low in downtrend
        assert rsi.iloc[-1] < 30

    def test_rsi_custom_period(self, volatile_df: pd.DataFrame) -> None:
        """RSI should work with custom period."""
        ti = TechnicalIndicators(volatile_df)

        rsi_7 = ti.calculate_rsi(period=7)
        rsi_14 = ti.calculate_rsi(period=14)

        # Different periods should give different results on volatile data
        assert not rsi_7.equals(rsi_14)

    def test_rsi_insufficient_data_raises_error(self) -> None:
        """RSI with insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            }
        )
        ti = TechnicalIndicators(df)

        with pytest.raises(InsufficientDataError, match="RSI"):
            ti.calculate_rsi(period=14)


# ============================================================================
# MACD Tests
# ============================================================================


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_returns_three_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """MACD should return three Series: line, signal, histogram."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        macd, signal, hist = ti.calculate_macd()

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_macd_histogram_is_difference(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """MACD histogram should be MACD line minus signal line."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        macd, signal, hist = ti.calculate_macd()

        expected_hist = macd - signal
        pd.testing.assert_series_equal(hist, expected_hist)

    def test_macd_uptrend_positive(self, trending_up_df: pd.DataFrame) -> None:
        """MACD should be positive in uptrend."""
        ti = TechnicalIndicators(trending_up_df)
        macd, _, _ = ti.calculate_macd()

        # MACD should be positive at the end of uptrend
        assert macd.iloc[-1] > 0

    def test_macd_downtrend_negative(self, trending_down_df: pd.DataFrame) -> None:
        """MACD should be negative in downtrend."""
        ti = TechnicalIndicators(trending_down_df)
        macd, _, _ = ti.calculate_macd()

        # MACD should be negative at the end of downtrend
        assert macd.iloc[-1] < 0

    def test_macd_custom_periods(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """MACD should work with custom periods."""
        ti = TechnicalIndicators(simple_ohlcv_df)

        macd1, _, _ = ti.calculate_macd(fast=12, slow=26, signal=9)
        macd2, _, _ = ti.calculate_macd(fast=8, slow=17, signal=9)

        # Different parameters should give different results
        assert not macd1.equals(macd2)

    def test_macd_insufficient_data_raises_error(self) -> None:
        """MACD with insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": [100] * 20,
                "high": [101] * 20,
                "low": [99] * 20,
                "close": [100] * 20,
                "volume": [1000] * 20,
            }
        )
        ti = TechnicalIndicators(df)

        with pytest.raises(InsufficientDataError, match="MACD"):
            ti.calculate_macd()


# ============================================================================
# EMA Tests
# ============================================================================


class TestEMA:
    """Tests for EMA calculation."""

    def test_ema_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """EMA should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        ema = ti.calculate_ema(9)

        assert isinstance(ema, pd.Series)
        assert len(ema) == len(simple_ohlcv_df)

    def test_ema_smoothing_effect(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """EMA should smooth out price data."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        ema = ti.calculate_ema(20)

        # EMA should have lower standard deviation than close prices
        close_std = ti.df["close"].std()
        ema_std = ema.dropna().std()

        assert ema_std <= close_std

    def test_ema_shorter_period_more_responsive(self, volatile_df: pd.DataFrame) -> None:
        """Shorter EMA period should be more responsive to price changes."""
        ti = TechnicalIndicators(volatile_df)

        ema_9 = ti.calculate_ema(9)
        ema_21 = ti.calculate_ema(21)

        # Shorter EMA should have higher variance (more responsive)
        ema_9_std = ema_9.dropna().std()
        ema_21_std = ema_21.dropna().std()

        assert ema_9_std >= ema_21_std

    def test_multiple_emas_returns_dict(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """calculate_multiple_emas should return dictionary."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        emas = ti.calculate_multiple_emas([9, 21])

        assert isinstance(emas, dict)
        assert 9 in emas
        assert 21 in emas

    def test_multiple_emas_default_periods(self, large_ohlcv_df: pd.DataFrame) -> None:
        """Default periods should be [9, 21, 50, 200]."""
        ti = TechnicalIndicators(large_ohlcv_df)
        emas = ti.calculate_multiple_emas()

        assert set(emas.keys()) == {9, 21, 50, 200}

    def test_ema_insufficient_data_raises_error(self) -> None:
        """EMA with insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [101] * 5,
                "low": [99] * 5,
                "close": [100] * 5,
                "volume": [1000] * 5,
            }
        )
        ti = TechnicalIndicators(df)

        with pytest.raises(InsufficientDataError, match="EMA"):
            ti.calculate_ema(10)


# ============================================================================
# ATR Tests
# ============================================================================


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """ATR should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        atr = ti.calculate_atr()

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(simple_ohlcv_df)

    def test_atr_always_positive(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """ATR values should always be positive."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        atr = ti.calculate_atr()

        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_atr_higher_for_volatile_data(
        self, simple_ohlcv_df: pd.DataFrame, volatile_df: pd.DataFrame
    ) -> None:
        """ATR should be higher for more volatile data."""
        ti_simple = TechnicalIndicators(simple_ohlcv_df)
        ti_volatile = TechnicalIndicators(volatile_df)

        atr_simple = ti_simple.calculate_atr().dropna().mean()
        atr_volatile = ti_volatile.calculate_atr().dropna().mean()

        # Volatile data should have higher ATR
        assert atr_volatile > atr_simple

    def test_atr_true_range_calculation(self) -> None:
        """Test ATR uses correct True Range formula."""
        # Create data where each TR component matters
        df = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100] * 5,
                "high": [105, 103, 102, 101, 100] * 5,  # Varies
                "low": [95, 97, 98, 99, 100] * 5,  # Varies
                "close": [100, 100, 100, 100, 100] * 5,
                "volume": [1000000] * 25,
            },
            index=pd.date_range("2024-01-01", periods=25, freq="D"),
        )

        ti = TechnicalIndicators(df)
        atr = ti.calculate_atr(period=5)

        # ATR should be calculated and positive
        valid_atr = atr.dropna()
        assert len(valid_atr) > 0
        assert (valid_atr > 0).all()

    def test_atr_custom_period(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """ATR should work with custom period."""
        ti = TechnicalIndicators(simple_ohlcv_df)

        atr_7 = ti.calculate_atr(period=7)
        atr_14 = ti.calculate_atr(period=14)

        # Different periods should give different results
        assert not atr_7.equals(atr_14)

    def test_atr_insufficient_data_raises_error(self) -> None:
        """ATR with insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            }
        )
        ti = TechnicalIndicators(df)

        with pytest.raises(InsufficientDataError, match="ATR"):
            ti.calculate_atr(period=14)


# ============================================================================
# Bollinger Bands Tests
# ============================================================================


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_returns_three_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Bollinger should return three Series: upper, middle, lower."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        upper, middle, lower = ti.calculate_bollinger()

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_bollinger_band_order(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Upper band should be above middle, middle above lower."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        upper, middle, lower = ti.calculate_bollinger()

        # Get valid values (after warm-up period)
        valid_idx = upper.dropna().index

        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()

    def test_bollinger_middle_is_sma(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Middle band should be SMA of close prices."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        _, middle, _ = ti.calculate_bollinger(period=20)

        expected_sma = ti.df["close"].rolling(window=20).mean()

        pd.testing.assert_series_equal(middle, expected_sma)

    def test_bollinger_width_increases_with_std_dev(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Band width should increase with std_dev parameter."""
        ti = TechnicalIndicators(simple_ohlcv_df)

        upper1, middle1, lower1 = ti.calculate_bollinger(std_dev=1.0)
        upper2, middle2, lower2 = ti.calculate_bollinger(std_dev=2.0)

        width1 = (upper1 - lower1).dropna().mean()
        width2 = (upper2 - lower2).dropna().mean()

        assert width2 > width1

    def test_bollinger_custom_period(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Bollinger should work with custom period."""
        ti = TechnicalIndicators(simple_ohlcv_df)

        _, middle1, _ = ti.calculate_bollinger(period=10)
        _, middle2, _ = ti.calculate_bollinger(period=20)

        # Different periods should give different middle bands
        assert not middle1.equals(middle2)

    def test_bollinger_insufficient_data_raises_error(self) -> None:
        """Bollinger with insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            }
        )
        ti = TechnicalIndicators(df)

        with pytest.raises(InsufficientDataError, match="Bollinger"):
            ti.calculate_bollinger(period=20)


# ============================================================================
# Volume Indicator Tests
# ============================================================================


class TestVolumeIndicators:
    """Tests for volume-based indicators."""

    def test_volume_sma_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Volume SMA should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        vol_sma = ti.calculate_volume_sma()

        assert isinstance(vol_sma, pd.Series)
        assert len(vol_sma) == len(simple_ohlcv_df)

    def test_volume_sma_smooths_volume(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Volume SMA should smooth volume data."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        vol_sma = ti.calculate_volume_sma(20)

        # SMA should have lower variance than raw volume
        vol_std = ti.df["volume"].std()
        sma_std = vol_sma.dropna().std()

        assert sma_std <= vol_std

    def test_relative_volume_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Relative volume should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        rvol = ti.calculate_relative_volume()

        assert isinstance(rvol, pd.Series)

    def test_relative_volume_average_around_one(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Average relative volume should be around 1.0."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        rvol = ti.calculate_relative_volume()

        # Mean RVOL should be close to 1.0 (it's current/average)
        valid_rvol = rvol.dropna()
        assert 0.5 < valid_rvol.mean() < 2.0

    def test_relative_volume_high_volume_day(self) -> None:
        """Relative volume should be high on high volume day."""
        # Create data with one very high volume day
        volumes = [1000000] * 30 + [5000000]  # Last day 5x normal
        dates = pd.date_range("2024-01-01", periods=31, freq="D")

        df = pd.DataFrame(
            {
                "open": [100] * 31,
                "high": [101] * 31,
                "low": [99] * 31,
                "close": [100] * 31,
                "volume": volumes,
            },
            index=dates,
        )

        ti = TechnicalIndicators(df)
        rvol = ti.calculate_relative_volume(20)

        # Last day RVOL should be high
        assert rvol.iloc[-1] > 3.0

    def test_volume_sma_insufficient_data_raises_error(self) -> None:
        """Volume SMA with insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [101] * 10,
                "low": [99] * 10,
                "close": [100] * 10,
                "volume": [1000] * 10,
            }
        )
        ti = TechnicalIndicators(df)

        with pytest.raises(InsufficientDataError, match="Volume SMA"):
            ti.calculate_volume_sma(period=20)


# ============================================================================
# Additional Indicator Tests
# ============================================================================


class TestAdditionalIndicators:
    """Tests for additional indicators (SMA, Stochastic, ADX, OBV)."""

    def test_sma_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """SMA should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        sma = ti.calculate_sma(20)

        assert isinstance(sma, pd.Series)
        assert len(sma) == len(simple_ohlcv_df)

    def test_sma_calculation_correct(self) -> None:
        """SMA calculation should match hand calculation."""
        prices = [10, 20, 30, 40, 50]
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": [1000] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )

        ti = TechnicalIndicators(df)
        sma = ti.calculate_sma(3)

        # SMA of [10, 20, 30] = 20
        assert sma.iloc[2] == pytest.approx(20.0)
        # SMA of [20, 30, 40] = 30
        assert sma.iloc[3] == pytest.approx(30.0)

    def test_stochastic_returns_two_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Stochastic should return two Series: %K and %D."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        stoch_k, stoch_d = ti.calculate_stochastic()

        assert isinstance(stoch_k, pd.Series)
        assert isinstance(stoch_d, pd.Series)

    def test_stochastic_values_between_0_and_100(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Stochastic values should be between 0 and 100."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        stoch_k, stoch_d = ti.calculate_stochastic()

        valid_k = stoch_k.dropna()
        valid_d = stoch_d.dropna()

        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()

    def test_adx_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """ADX should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        adx = ti.calculate_adx()

        assert isinstance(adx, pd.Series)

    def test_adx_values_between_0_and_100(self, large_ohlcv_df: pd.DataFrame) -> None:
        """ADX values should be between 0 and 100."""
        # Use large_ohlcv_df which has random walk, not perfect trend
        ti = TechnicalIndicators(large_ohlcv_df)
        adx = ti.calculate_adx()

        valid_adx = adx.dropna()
        assert (valid_adx >= 0).all()
        # ADX can technically reach 100 in perfect trends
        assert (valid_adx <= 100).all()

    def test_obv_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """OBV should return a pandas Series."""
        ti = TechnicalIndicators(simple_ohlcv_df)
        obv = ti.calculate_obv()

        assert isinstance(obv, pd.Series)
        assert len(obv) == len(simple_ohlcv_df)

    def test_obv_increases_on_up_days(self, trending_up_df: pd.DataFrame) -> None:
        """OBV should increase during uptrend."""
        ti = TechnicalIndicators(trending_up_df)
        obv = ti.calculate_obv()

        # OBV should be positive and increasing in uptrend
        assert obv.iloc[-1] > obv.iloc[1]


# ============================================================================
# calculate_all() Tests
# ============================================================================


class TestCalculateAll:
    """Tests for calculate_all() convenience method."""

    def test_calculate_all_returns_dataframe(self, large_ohlcv_df: pd.DataFrame) -> None:
        """calculate_all should return a DataFrame."""
        ti = TechnicalIndicators(large_ohlcv_df)
        result = ti.calculate_all()

        assert isinstance(result, pd.DataFrame)

    def test_calculate_all_has_required_columns(self, large_ohlcv_df: pd.DataFrame) -> None:
        """calculate_all should add all standard indicator columns."""
        ti = TechnicalIndicators(large_ohlcv_df)
        result = ti.calculate_all()

        expected_columns = [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "ema_9",
            "ema_21",
            "ema_50",
            "ema_200",
            "atr_14",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "volume_sma_20",
            "relative_volume",
        ]

        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_calculate_all_preserves_original_columns(self, large_ohlcv_df: pd.DataFrame) -> None:
        """calculate_all should preserve original OHLCV columns."""
        ti = TechnicalIndicators(large_ohlcv_df)
        result = ti.calculate_all()

        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_calculate_all_insufficient_data_raises_error(
        self, simple_ohlcv_df: pd.DataFrame
    ) -> None:
        """calculate_all with insufficient data for EMA 200 should raise error."""
        ti = TechnicalIndicators(simple_ohlcv_df)  # Only 50 rows

        with pytest.raises(InsufficientDataError, match="EMA 200"):
            ti.calculate_all()

    def test_calculate_all_does_not_modify_original(self, large_ohlcv_df: pd.DataFrame) -> None:
        """calculate_all should not modify the original DataFrame."""
        original_columns = set(large_ohlcv_df.columns)

        ti = TechnicalIndicators(large_ohlcv_df)
        ti.calculate_all()

        # Original DataFrame should be unchanged
        assert set(large_ohlcv_df.columns) == original_columns


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handles_nan_in_middle(self) -> None:
        """Indicators should handle NaN values in the middle of data."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        closes = [100.0 + i * 0.5 for i in range(50)]
        closes[25] = float("nan")  # NaN in middle

        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 if not pd.isna(c) else float("nan") for c in closes],
                "low": [c - 1 if not pd.isna(c) else float("nan") for c in closes],
                "close": closes,
                "volume": [1000000] * 50,
            },
            index=dates,
        )

        ti = TechnicalIndicators(df)

        # Should not raise, but will have NaN propagation
        rsi = ti.calculate_rsi()
        assert isinstance(rsi, pd.Series)

    def test_handles_zero_volume(self) -> None:
        """Indicators should handle zero volume."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")

        df = pd.DataFrame(
            {
                "open": [100] * 30,
                "high": [101] * 30,
                "low": [99] * 30,
                "close": [100] * 30,
                "volume": [0] * 30,  # Zero volume
            },
            index=dates,
        )

        ti = TechnicalIndicators(df)

        # Volume SMA should work but be zero
        vol_sma = ti.calculate_volume_sma(20)
        assert vol_sma.iloc[-1] == 0

    def test_handles_constant_prices(self) -> None:
        """Indicators should handle constant (unchanging) prices."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")

        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [100.0] * 30,
                "low": [100.0] * 30,
                "close": [100.0] * 30,
                "volume": [1000000] * 30,
            },
            index=dates,
        )

        ti = TechnicalIndicators(df)

        # RSI should be NaN or 50 for constant prices (no gains or losses)
        rsi = ti.calculate_rsi()
        # With constant prices, there are no gains or losses after first diff
        # This results in 0/0 = NaN or handled as 100 in our implementation
        assert isinstance(rsi, pd.Series)

        # EMA should equal the constant price
        ema = ti.calculate_ema(10)
        assert ema.iloc[-1] == pytest.approx(100.0)

    def test_handles_large_price_movements(self) -> None:
        """Indicators should handle very large price movements."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        # 100% gain
        closes = [100] * 15 + [200] * 15

        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c * 1.01 for c in closes],
                "low": [c * 0.99 for c in closes],
                "close": closes,
                "volume": [1000000] * 30,
            },
            index=dates,
        )

        ti = TechnicalIndicators(df)

        # All indicators should work
        rsi = ti.calculate_rsi()
        assert isinstance(rsi, pd.Series)
        assert (rsi.dropna() >= 0).all()
        assert (rsi.dropna() <= 100).all()

    def test_minimum_data_for_each_indicator(self) -> None:
        """Test minimum data requirements for each indicator."""
        # Minimum data for RSI(14) = 15 rows
        df_15 = pd.DataFrame(
            {
                "open": range(100, 115),
                "high": range(101, 116),
                "low": range(99, 114),
                "close": range(100, 115),
                "volume": [1000] * 15,
            },
            index=pd.date_range("2024-01-01", periods=15, freq="D"),
        )

        ti = TechnicalIndicators(df_15)

        # RSI should work with exactly 15 rows
        rsi = ti.calculate_rsi(14)
        assert isinstance(rsi, pd.Series)
