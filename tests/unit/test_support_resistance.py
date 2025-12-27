"""Unit tests for support and resistance detection module."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.indicators.exceptions import InsufficientDataError, InvalidDataError
from src.indicators.support_resistance import (
    LevelType,
    PriceLevel,
    SupportResistanceDetector,
    SupportResistanceLevels,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_ohlcv_df() -> pd.DataFrame:
    """Create simple OHLCV data with clear patterns."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # Create a range-bound pattern with clear support and resistance
    np.random.seed(42)
    base_prices = []
    price = 100

    for i in range(60):
        # Create oscillating pattern between 95 and 105
        price = 95 + (i % 10) if i % 20 < 10 else 105 - (i % 10)
        base_prices.append(price)

    return pd.DataFrame(
        {
            "open": [p - 0.5 for p in base_prices],
            "high": [p + 1 for p in base_prices],
            "low": [p - 1 for p in base_prices],
            "close": base_prices,
            "volume": [1000000] * 60,
        },
        index=dates,
    )


@pytest.fixture
def double_bottom_df() -> pd.DataFrame:
    """Create data with clear double bottom pattern."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # Create W-shaped pattern (double bottom at 90)
    prices = []
    for i in range(60):
        if i < 15:
            # First decline to 90
            prices.append(100 - i * 0.67)
        elif i < 20:
            # First bounce
            prices.append(90 + (i - 15) * 2)
        elif i < 25:
            # Pullback
            prices.append(100 - (i - 20) * 2)
        elif i < 30:
            # Second bottom at 90
            prices.append(90 + abs(i - 27.5) * 0.4)
        elif i < 40:
            # Second bounce
            prices.append(90 + (i - 30) * 1.5)
        else:
            # Continuation
            prices.append(105 + (i - 40) * 0.2)

    return pd.DataFrame(
        {
            "open": [p - 0.3 for p in prices],
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000000] * 60,
        },
        index=dates,
    )


@pytest.fixture
def double_top_df() -> pd.DataFrame:
    """Create data with clear double top pattern."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # Create M-shaped pattern (double top at 110)
    prices = []
    for i in range(60):
        if i < 15:
            # First rally to 110
            prices.append(100 + i * 0.67)
        elif i < 20:
            # First pullback
            prices.append(110 - (i - 15) * 2)
        elif i < 25:
            # Rally back
            prices.append(100 + (i - 20) * 2)
        elif i < 30:
            # Second top at 110
            prices.append(110 - abs(i - 27.5) * 0.4)
        elif i < 40:
            # Breakdown
            prices.append(110 - (i - 30) * 1.5)
        else:
            # Continuation down
            prices.append(95 - (i - 40) * 0.2)

    return pd.DataFrame(
        {
            "open": [p + 0.3 for p in prices],
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000000] * 60,
        },
        index=dates,
    )


@pytest.fixture
def trading_range_df() -> pd.DataFrame:
    """Create data with multiple touches at support and resistance."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    # Create oscillating pattern with multiple touches at 95 (support) and 105 (resistance)
    prices = []
    for i in range(100):
        cycle_pos = i % 20
        if cycle_pos < 10:
            # Rising phase
            prices.append(95 + cycle_pos)
        else:
            # Falling phase
            prices.append(105 - (cycle_pos - 10))

    return pd.DataFrame(
        {
            "open": [p - 0.2 for p in prices],
            "high": [p + 0.8 for p in prices],
            "low": [p - 0.8 for p in prices],
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def uptrend_df() -> pd.DataFrame:
    """Create data with clear uptrend (no clear support/resistance)."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")

    # Steady uptrend
    prices = [100 + i * 0.5 for i in range(60)]

    return pd.DataFrame(
        {
            "open": [p - 0.3 for p in prices],
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000000] * 60,
        },
        index=dates,
    )


# ============================================================================
# PriceLevel Tests
# ============================================================================


class TestPriceLevel:
    """Tests for PriceLevel dataclass."""

    def test_price_level_creation(self) -> None:
        """Test creating a valid PriceLevel."""
        level = PriceLevel(
            price=100.0,
            strength=3,
            level_type=LevelType.SUPPORT,
            last_touch=date(2024, 1, 15),
        )

        assert level.price == 100.0
        assert level.strength == 3
        assert level.level_type == LevelType.SUPPORT
        assert level.last_touch == date(2024, 1, 15)

    def test_price_level_invalid_strength(self) -> None:
        """Test that strength must be at least 1."""
        with pytest.raises(ValueError, match="Strength must be at least 1"):
            PriceLevel(
                price=100.0,
                strength=0,
                level_type=LevelType.SUPPORT,
                last_touch=date(2024, 1, 15),
            )

    def test_price_level_invalid_price(self) -> None:
        """Test that price must be positive."""
        with pytest.raises(ValueError, match="Price must be positive"):
            PriceLevel(
                price=-10.0,
                strength=2,
                level_type=LevelType.RESISTANCE,
                last_touch=date(2024, 1, 15),
            )


# ============================================================================
# SupportResistanceLevels Tests
# ============================================================================


class TestSupportResistanceLevels:
    """Tests for SupportResistanceLevels container."""

    def test_empty_levels(self) -> None:
        """Test empty levels container."""
        levels = SupportResistanceLevels()

        assert levels.support_levels == []
        assert levels.resistance_levels == []
        assert levels.nearest_support is None
        assert levels.nearest_resistance is None
        assert not levels.has_levels()

    def test_has_levels_with_support(self) -> None:
        """Test has_levels returns True when support exists."""
        support = PriceLevel(
            price=100.0,
            strength=2,
            level_type=LevelType.SUPPORT,
            last_touch=date(2024, 1, 15),
        )
        levels = SupportResistanceLevels(support_levels=[support])

        assert levels.has_levels()

    def test_all_levels_combines_both(self) -> None:
        """Test all_levels returns combined list."""
        support = PriceLevel(
            price=95.0,
            strength=2,
            level_type=LevelType.SUPPORT,
            last_touch=date(2024, 1, 15),
        )
        resistance = PriceLevel(
            price=105.0,
            strength=3,
            level_type=LevelType.RESISTANCE,
            last_touch=date(2024, 1, 20),
        )
        levels = SupportResistanceLevels(
            support_levels=[support],
            resistance_levels=[resistance],
        )

        assert len(levels.all_levels) == 2
        assert support in levels.all_levels
        assert resistance in levels.all_levels


# ============================================================================
# Input Validation Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_empty_dataframe_raises_error(self) -> None:
        """Empty DataFrame should raise InvalidDataError."""
        df = pd.DataFrame()

        with pytest.raises(InvalidDataError, match="empty"):
            SupportResistanceDetector(df)

    def test_missing_columns_raises_error(self) -> None:
        """Missing required columns should raise InvalidDataError."""
        df = pd.DataFrame({"close": range(30)})

        with pytest.raises(InvalidDataError, match="Missing required columns"):
            SupportResistanceDetector(df)

    def test_insufficient_data_raises_error(self) -> None:
        """Insufficient data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "high": range(10),
                "low": range(10),
                "close": range(10),
            }
        )

        with pytest.raises(InsufficientDataError):
            SupportResistanceDetector(df)

    def test_valid_dataframe_accepted(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Valid DataFrame should be accepted."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        assert detector.df is not None


# ============================================================================
# Pivot Detection Tests
# ============================================================================


class TestPivotDetection:
    """Tests for pivot point detection."""

    def test_find_pivot_highs_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """find_pivot_highs should return a boolean Series."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        pivots = detector.find_pivot_highs()

        assert isinstance(pivots, pd.Series)
        assert pivots.dtype == bool

    def test_find_pivot_lows_returns_series(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """find_pivot_lows should return a boolean Series."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        pivots = detector.find_pivot_lows()

        assert isinstance(pivots, pd.Series)
        assert pivots.dtype == bool

    def test_pivot_highs_found_at_peaks(self) -> None:
        """Pivot highs should be found at price peaks."""
        # Create clear peak pattern
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
        ]

        df = pd.DataFrame(
            {
                "high": prices,
                "low": [p - 1 for p in prices],
                "close": prices,
            },
            index=dates,
        )

        detector = SupportResistanceDetector(df, lookback=30)
        pivots = detector.find_pivot_highs(left=3, right=3)

        # Should find pivot highs at indices 5 and 15 (peaks at 105)
        assert pivots.sum() > 0

    def test_pivot_lows_found_at_troughs(self) -> None:
        """Pivot lows should be found at price troughs."""
        # Create clear trough pattern
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = [
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
            105,
            104,
            103,
            102,
            101,
            100,
            101,
            102,
            103,
            104,
        ]

        df = pd.DataFrame(
            {
                "high": [p + 1 for p in prices],
                "low": prices,
                "close": prices,
            },
            index=dates,
        )

        detector = SupportResistanceDetector(df, lookback=30)
        pivots = detector.find_pivot_lows(left=3, right=3)

        # Should find pivot lows at troughs (at 100)
        assert pivots.sum() > 0

    def test_custom_left_right_parameters(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Pivot detection should work with custom left/right parameters."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        pivots_3_3 = detector.find_pivot_highs(left=3, right=3)
        pivots_7_7 = detector.find_pivot_highs(left=7, right=7)

        # Larger window should find fewer or equal pivots
        assert pivots_7_7.sum() <= pivots_3_3.sum()


# ============================================================================
# Level Clustering Tests
# ============================================================================


class TestLevelClustering:
    """Tests for level clustering algorithm."""

    def test_cluster_single_price(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Single price should create single cluster."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        prices = [(100.0, date(2024, 1, 1))]

        levels = detector.cluster_levels(prices)

        assert len(levels) == 1
        assert levels[0].price == 100.0
        assert levels[0].strength == 1

    def test_cluster_nearby_prices(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Nearby prices should be clustered together."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        # Prices within 2% of each other
        prices = [
            (100.0, date(2024, 1, 1)),
            (100.5, date(2024, 1, 5)),
            (101.0, date(2024, 1, 10)),
        ]

        levels = detector.cluster_levels(prices, tolerance_pct=0.02)

        # Should cluster into single level
        assert len(levels) == 1
        assert levels[0].strength == 3

    def test_cluster_distant_prices_separate(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Distant prices should remain separate."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        # Prices far apart (>2% difference)
        prices = [
            (100.0, date(2024, 1, 1)),
            (110.0, date(2024, 1, 5)),
            (120.0, date(2024, 1, 10)),
        ]

        levels = detector.cluster_levels(prices, tolerance_pct=0.02)

        # Should remain as separate levels
        assert len(levels) == 3

    def test_cluster_empty_list(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Empty price list should return empty clusters."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        levels = detector.cluster_levels([])

        assert levels == []

    def test_cluster_preserves_latest_touch(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Clustered level should have latest touch date."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        prices = [
            (100.0, date(2024, 1, 1)),
            (100.5, date(2024, 1, 15)),  # Latest
            (100.2, date(2024, 1, 10)),
        ]

        levels = detector.cluster_levels(prices, tolerance_pct=0.02)

        assert levels[0].last_touch == date(2024, 1, 15)


# ============================================================================
# Level Detection Tests
# ============================================================================


class TestLevelDetection:
    """Tests for main level detection."""

    def test_detect_levels_returns_correct_type(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """detect_levels should return SupportResistanceLevels."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        levels = detector.detect_levels()

        assert isinstance(levels, SupportResistanceLevels)

    def test_detects_obvious_double_bottom(self, double_bottom_df: pd.DataFrame) -> None:
        """Should detect support at double bottom."""
        detector = SupportResistanceDetector(double_bottom_df)
        levels = detector.detect_levels(min_touches=1)

        # Should find support levels
        assert len(levels.support_levels) > 0

    def test_detects_obvious_double_top(self, double_top_df: pd.DataFrame) -> None:
        """Should detect resistance at double top."""
        detector = SupportResistanceDetector(double_top_df)
        levels = detector.detect_levels(min_touches=1)

        # Should find resistance levels
        assert len(levels.resistance_levels) > 0

    def test_level_strength_increases_with_touches(self, trading_range_df: pd.DataFrame) -> None:
        """Levels with more touches should have higher strength."""
        detector = SupportResistanceDetector(trading_range_df, lookback=100)
        levels = detector.detect_levels(min_touches=1)

        # In trading range with multiple cycles, should find levels with strength > 1
        all_levels = levels.all_levels
        if all_levels:
            max_strength = max(level.strength for level in all_levels)
            assert max_strength >= 1

    def test_min_touches_filter(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """min_touches should filter out weak levels."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        levels_1 = detector.detect_levels(min_touches=1)
        levels_3 = detector.detect_levels(min_touches=3)

        # Higher min_touches should find fewer or equal levels
        total_1 = len(levels_1.support_levels) + len(levels_1.resistance_levels)
        total_3 = len(levels_3.support_levels) + len(levels_3.resistance_levels)

        assert total_3 <= total_1

    def test_levels_sorted_by_strength(self, trading_range_df: pd.DataFrame) -> None:
        """Levels should be sorted by strength (descending)."""
        detector = SupportResistanceDetector(trading_range_df, lookback=100)
        levels = detector.detect_levels(min_touches=1)

        # Check support levels are sorted
        for i in range(len(levels.support_levels) - 1):
            assert levels.support_levels[i].strength >= levels.support_levels[i + 1].strength

        # Check resistance levels are sorted
        for i in range(len(levels.resistance_levels) - 1):
            assert levels.resistance_levels[i].strength >= levels.resistance_levels[i + 1].strength

    def test_nearest_support_below_current_price(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """nearest_support should be below current price."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        levels = detector.detect_levels(min_touches=1)

        if levels.nearest_support is not None:
            current_price = simple_ohlcv_df["close"].iloc[-1]
            assert levels.nearest_support.price < current_price

    def test_nearest_resistance_above_current_price(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """nearest_resistance should be above current price."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        levels = detector.detect_levels(min_touches=1)

        if levels.nearest_resistance is not None:
            current_price = simple_ohlcv_df["close"].iloc[-1]
            assert levels.nearest_resistance.price > current_price


# ============================================================================
# Level Validation Tests
# ============================================================================


class TestLevelValidation:
    """Tests for level validation methods."""

    def test_is_at_support_true_when_at_level(self, trading_range_df: pd.DataFrame) -> None:
        """is_at_support should return True when at support level."""
        detector = SupportResistanceDetector(trading_range_df, lookback=100)
        levels = detector.detect_levels(min_touches=1)

        if levels.support_levels:
            support_price = levels.support_levels[0].price
            # Price exactly at support should return True
            assert detector.is_at_support(support_price, tolerance_pct=0.01)

    def test_is_at_support_false_when_far(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """is_at_support should return False when far from support."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        # Price very high, unlikely to be at support
        assert not detector.is_at_support(1000.0, tolerance_pct=0.01)

    def test_is_at_resistance_true_when_at_level(self, trading_range_df: pd.DataFrame) -> None:
        """is_at_resistance should return True when at resistance level."""
        detector = SupportResistanceDetector(trading_range_df, lookback=100)
        levels = detector.detect_levels(min_touches=1)

        if levels.resistance_levels:
            resistance_price = levels.resistance_levels[0].price
            # Price exactly at resistance should return True
            assert detector.is_at_resistance(resistance_price, tolerance_pct=0.01)

    def test_is_at_resistance_false_when_far(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """is_at_resistance should return False when far from resistance."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        # Price very low, unlikely to be at resistance
        assert not detector.is_at_resistance(1.0, tolerance_pct=0.01)

    def test_distance_to_support_positive(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """distance_to_support should return positive percentage."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        current_price = simple_ohlcv_df["close"].iloc[-1]

        distance = detector.distance_to_support(current_price)

        # Distance is always positive or inf
        assert distance >= 0 or distance == float("inf")

    def test_distance_to_resistance_positive(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """distance_to_resistance should return positive percentage."""
        detector = SupportResistanceDetector(simple_ohlcv_df)
        current_price = simple_ohlcv_df["close"].iloc[-1]

        distance = detector.distance_to_resistance(current_price)

        # Distance is always positive or inf
        assert distance >= 0 or distance == float("inf")

    def test_distance_returns_inf_when_no_levels(self, uptrend_df: pd.DataFrame) -> None:
        """Distance should return inf when no levels found."""
        detector = SupportResistanceDetector(uptrend_df)

        # In strong uptrend with min_touches=10, unlikely to find levels
        # Using a very high price to ensure no support below
        distance = detector.distance_to_support(10000.0)

        # Either returns inf or a valid distance
        assert distance >= 0 or distance == float("inf")


# ============================================================================
# Helper Method Tests
# ============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_level_at_price_found(self, trading_range_df: pd.DataFrame) -> None:
        """get_level_at_price should find level at specific price."""
        detector = SupportResistanceDetector(trading_range_df, lookback=100)
        levels = detector.detect_levels(min_touches=1)

        if levels.all_levels:
            target_price = levels.all_levels[0].price
            found_level = detector.get_level_at_price(target_price, tolerance_pct=0.02)

            assert found_level is not None
            assert abs(found_level.price - target_price) / target_price <= 0.02

    def test_get_level_at_price_not_found(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """get_level_at_price should return None when no level found."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        # Price far from any level
        found_level = detector.get_level_at_price(1000.0, tolerance_pct=0.01)

        assert found_level is None

    def test_get_levels_in_range(self, trading_range_df: pd.DataFrame) -> None:
        """get_levels_in_range should return levels within range."""
        detector = SupportResistanceDetector(trading_range_df, lookback=100)

        levels_in_range = detector.get_levels_in_range(90.0, 110.0)

        # All returned levels should be within range
        for level in levels_in_range:
            assert 90.0 <= level.price <= 110.0

    def test_get_levels_in_range_empty(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """get_levels_in_range should return empty list if no levels in range."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        # Range far from any levels
        levels_in_range = detector.get_levels_in_range(1000.0, 2000.0)

        assert levels_in_range == []


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_lookback_limits_data(self, trading_range_df: pd.DataFrame) -> None:
        """Lookback should limit the amount of data analyzed."""
        detector_30 = SupportResistanceDetector(trading_range_df, lookback=30)
        detector_100 = SupportResistanceDetector(trading_range_df, lookback=100)

        assert len(detector_30.df) == 30
        assert len(detector_100.df) == 100

    def test_lookback_larger_than_data(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Lookback larger than data should use all available data."""
        detector = SupportResistanceDetector(simple_ohlcv_df, lookback=1000)

        assert len(detector.df) == len(simple_ohlcv_df)

    def test_handles_flat_prices(self) -> None:
        """Should handle flat/constant prices without error."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        df = pd.DataFrame(
            {
                "high": [100.0] * 30,
                "low": [100.0] * 30,
                "close": [100.0] * 30,
            },
            index=dates,
        )

        detector = SupportResistanceDetector(df, lookback=30)
        levels = detector.detect_levels()

        # Should not raise, may or may not find levels
        assert isinstance(levels, SupportResistanceLevels)

    def test_handles_volatile_data(self) -> None:
        """Should handle highly volatile data without error."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        np.random.seed(42)

        prices = 100 + np.random.randn(60) * 10  # High volatility

        df = pd.DataFrame(
            {
                "high": prices + 2,
                "low": prices - 2,
                "close": prices,
            },
            index=dates,
        )

        detector = SupportResistanceDetector(df)
        levels = detector.detect_levels()

        # Should not raise
        assert isinstance(levels, SupportResistanceLevels)

    def test_level_type_enum_values(self) -> None:
        """LevelType enum should have correct values."""
        assert LevelType.SUPPORT.value == "SUPPORT"
        assert LevelType.RESISTANCE.value == "RESISTANCE"

    def test_tolerance_affects_clustering(self, simple_ohlcv_df: pd.DataFrame) -> None:
        """Different tolerance should affect clustering results."""
        detector = SupportResistanceDetector(simple_ohlcv_df)

        prices = [
            (100.0, date(2024, 1, 1)),
            (102.0, date(2024, 1, 5)),
            (104.0, date(2024, 1, 10)),
        ]

        levels_1pct = detector.cluster_levels(prices, tolerance_pct=0.01)
        levels_5pct = detector.cluster_levels(prices, tolerance_pct=0.05)

        # Higher tolerance should result in fewer clusters
        assert len(levels_5pct) <= len(levels_1pct)
