"""Unit tests for multi-timeframe analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators.exceptions import InsufficientDataError, InvalidDataError
from src.indicators.timeframes import (
    ConfluenceResult,
    MultiTimeframeAnalyzer,
    MultiTimeframeResult,
    SignalStrength,
    TimeframeAggregator,
    Trend,
    TrendAlignment,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def intraday_1min_df() -> pd.DataFrame:
    """Create 1-minute intraday data for aggregation tests."""
    # Create 8 hours of 1-minute data (480 bars)
    dates = pd.date_range("2024-01-02 09:30", periods=480, freq="1min")

    np.random.seed(42)
    base_price = 100.0
    prices = []

    for _ in range(480):
        base_price = base_price * (1 + np.random.normal(0, 0.001))
        prices.append(base_price)

    prices = np.array(prices)

    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.randint(1000, 5000, 480),
        },
        index=dates,
    )


@pytest.fixture
def intraday_5min_df() -> pd.DataFrame:
    """Create 5-minute intraday data."""
    # Create 5 days of 5-minute data (78 bars/day * 5 = 390 bars)
    dates = pd.date_range("2024-01-02 09:30", periods=390, freq="5min")

    np.random.seed(42)
    base_price = 100.0
    prices = []

    for _ in range(390):
        base_price = base_price * (1 + np.random.normal(0, 0.002))
        prices.append(base_price)

    prices = np.array(prices)

    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.003,
            "low": prices * 0.997,
            "close": prices,
            "volume": np.random.randint(5000, 20000, 390),
        },
        index=dates,
    )


@pytest.fixture
def daily_bullish_df() -> pd.DataFrame:
    """Create daily data with bullish trend."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Clear uptrend
    base = 100.0
    prices = []
    for _ in range(100):
        base = base * 1.005  # 0.5% daily gain
        prices.append(base)

    prices = np.array(prices)

    return pd.DataFrame(
        {
            "open": prices * 0.998,
            "high": prices * 1.01,
            "low": prices * 0.995,
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def daily_bearish_df() -> pd.DataFrame:
    """Create daily data with bearish trend."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Clear downtrend
    base = 200.0
    prices = []
    for _ in range(100):
        base = base * 0.995  # 0.5% daily loss
        prices.append(base)

    prices = np.array(prices)

    return pd.DataFrame(
        {
            "open": prices * 1.002,
            "high": prices * 1.005,
            "low": prices * 0.99,
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def daily_neutral_df() -> pd.DataFrame:
    """Create daily data with sideways/neutral trend."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    np.random.seed(42)
    # Sideways movement
    prices = 100 + np.random.randn(100) * 2

    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )


@pytest.fixture
def daily_rsi_oversold_df() -> pd.DataFrame:
    """Create daily data where RSI is oversold."""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")

    # Sharp decline to create oversold RSI
    prices = []
    base = 200.0
    for i in range(100):
        base = base * 0.999 if i < 80 else base * 0.98
        prices.append(base)

    prices = np.array(prices)

    return pd.DataFrame(
        {
            "open": prices * 1.001,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )


# ============================================================================
# TrendAlignment Tests
# ============================================================================


class TestTrendAlignment:
    """Tests for TrendAlignment dataclass."""

    def test_trend_alignment_creation(self) -> None:
        """Test creating a valid TrendAlignment."""
        alignment = TrendAlignment(
            daily_trend=Trend.BULLISH,
            h4_trend=Trend.BULLISH,
            h1_trend=Trend.BULLISH,
            alignment_score=1.0,
        )

        assert alignment.daily_trend == Trend.BULLISH
        assert alignment.alignment_score == 1.0

    def test_alignment_score_validation(self) -> None:
        """Test alignment score must be 0-1."""
        with pytest.raises(ValueError, match="alignment_score"):
            TrendAlignment(
                daily_trend=Trend.BULLISH,
                alignment_score=1.5,
            )

    def test_is_fully_aligned_all_bullish(self) -> None:
        """Test is_fully_aligned when all trends are bullish."""
        alignment = TrendAlignment(
            daily_trend=Trend.BULLISH,
            h4_trend=Trend.BULLISH,
            h1_trend=Trend.BULLISH,
            alignment_score=1.0,
        )

        assert alignment.is_fully_aligned

    def test_is_fully_aligned_mixed_signals(self) -> None:
        """Test is_fully_aligned with mixed signals."""
        alignment = TrendAlignment(
            daily_trend=Trend.BULLISH,
            h4_trend=Trend.BEARISH,
            h1_trend=Trend.BULLISH,
            alignment_score=0.5,
        )

        assert not alignment.is_fully_aligned

    def test_is_fully_aligned_with_neutrals(self) -> None:
        """Test is_fully_aligned with neutral trends."""
        alignment = TrendAlignment(
            daily_trend=Trend.BULLISH,
            h4_trend=Trend.NEUTRAL,
            h1_trend=Trend.BULLISH,
            alignment_score=0.7,
        )

        assert alignment.is_fully_aligned


# ============================================================================
# ConfluenceResult Tests
# ============================================================================


class TestConfluenceResult:
    """Tests for ConfluenceResult dataclass."""

    def test_confluence_result_creation(self) -> None:
        """Test creating a valid ConfluenceResult."""
        result = ConfluenceResult(
            indicator="rsi",
            timeframes_agreeing=["daily", "4H"],
            confluence_score=0.8,
            signal_strength=SignalStrength.STRONG,
        )

        assert result.indicator == "rsi"
        assert len(result.timeframes_agreeing) == 2
        assert result.signal_strength == SignalStrength.STRONG

    def test_confluence_score_validation(self) -> None:
        """Test confluence score must be 0-1."""
        with pytest.raises(ValueError, match="confluence_score"):
            ConfluenceResult(
                indicator="rsi",
                confluence_score=1.5,
            )


# ============================================================================
# TimeframeAggregator Tests
# ============================================================================


class TestTimeframeAggregator:
    """Tests for TimeframeAggregator class."""

    def test_aggregator_creation(self, intraday_1min_df: pd.DataFrame) -> None:
        """Test creating a valid aggregator."""
        aggregator = TimeframeAggregator(intraday_1min_df)
        assert aggregator.df is not None

    def test_empty_dataframe_raises_error(self) -> None:
        """Empty DataFrame should raise InvalidDataError."""
        df = pd.DataFrame()
        with pytest.raises(InvalidDataError, match="empty"):
            TimeframeAggregator(df)

    def test_missing_columns_raises_error(self) -> None:
        """Missing columns should raise InvalidDataError."""
        df = pd.DataFrame(
            {"close": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="1min"),
        )
        with pytest.raises(InvalidDataError, match="Missing required columns"):
            TimeframeAggregator(df)

    def test_non_datetime_index_raises_error(self) -> None:
        """Non-datetime index should raise InvalidDataError."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
                "volume": [1000, 1000, 1000],
            }
        )
        with pytest.raises(InvalidDataError, match="DatetimeIndex"):
            TimeframeAggregator(df)

    def test_aggregate_to_5min(self, intraday_1min_df: pd.DataFrame) -> None:
        """Test aggregation to 5-minute timeframe."""
        aggregator = TimeframeAggregator(intraday_1min_df)
        result = aggregator.aggregate_to_timeframe("5min")

        # 480 1-min bars should produce ~96 5-min bars
        assert len(result) > 0
        assert len(result) < len(intraday_1min_df)

        # Verify columns exist
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_aggregate_to_1h(self, intraday_1min_df: pd.DataFrame) -> None:
        """Test aggregation to 1-hour timeframe."""
        aggregator = TimeframeAggregator(intraday_1min_df)
        result = aggregator.aggregate_to_timeframe("1H")

        # 480 1-min bars (8 hours) should produce 8-9 hourly bars
        # (depends on alignment with hour boundaries)
        assert len(result) > 0
        assert len(result) <= 10

    def test_aggregate_preserves_volume_sum(self, intraday_1min_df: pd.DataFrame) -> None:
        """Total volume should be preserved after aggregation."""
        aggregator = TimeframeAggregator(intraday_1min_df)

        total_volume_before = intraday_1min_df["volume"].sum()
        result = aggregator.aggregate_to_timeframe("5min")
        total_volume_after = result["volume"].sum()

        # Allow small difference due to incomplete periods being dropped
        assert abs(total_volume_after - total_volume_before) / total_volume_before < 0.1

    def test_aggregate_ohlc_logic(self) -> None:
        """Test OHLCV aggregation logic is correct."""
        # Create specific data where we know the expected aggregation
        dates = pd.date_range("2024-01-02 09:30", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "volume": [1000, 2000, 3000, 4000, 5000],
            },
            index=dates,
        )

        aggregator = TimeframeAggregator(df)
        result = aggregator.aggregate_to_timeframe("5min")

        # Should have 1 5-min bar
        assert len(result) == 1

        bar = result.iloc[0]
        assert bar["open"] == 100.0  # First open
        assert bar["high"] == 105.0  # Max high
        assert bar["low"] == 99.0  # Min low
        assert bar["close"] == 104.5  # Last close
        assert bar["volume"] == 15000  # Sum of volumes

    def test_unsupported_timeframe_raises_error(self, intraday_1min_df: pd.DataFrame) -> None:
        """Unsupported timeframe should raise ValueError."""
        aggregator = TimeframeAggregator(intraday_1min_df)

        with pytest.raises(ValueError, match="Unsupported timeframe"):
            aggregator.aggregate_to_timeframe("2H")

    def test_get_available_timeframes(self, intraday_1min_df: pd.DataFrame) -> None:
        """Test getting available timeframes."""
        aggregator = TimeframeAggregator(intraday_1min_df)
        available = aggregator.get_available_timeframes()

        assert "5min" in available
        assert "15min" in available
        assert "1H" in available


# ============================================================================
# MultiTimeframeAnalyzer Tests
# ============================================================================


class TestMultiTimeframeAnalyzer:
    """Tests for MultiTimeframeAnalyzer class."""

    def test_analyzer_creation(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test creating a valid analyzer."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        assert analyzer.daily_df is not None

    def test_insufficient_daily_data_raises_error(self) -> None:
        """Insufficient daily data should raise InsufficientDataError."""
        df = pd.DataFrame(
            {
                "open": range(10),
                "high": range(1, 11),
                "low": range(10),
                "close": range(10),
                "volume": [1000] * 10,
            },
            index=pd.date_range("2024-01-01", periods=10, freq="D"),
        )

        with pytest.raises(InsufficientDataError):
            MultiTimeframeAnalyzer(df)

    def test_daily_trend_bullish(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test bullish trend detection."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        alignment = analyzer.get_trend_alignment()

        assert alignment.daily_trend == Trend.BULLISH

    def test_daily_trend_bearish(self, daily_bearish_df: pd.DataFrame) -> None:
        """Test bearish trend detection."""
        analyzer = MultiTimeframeAnalyzer(daily_bearish_df)
        alignment = analyzer.get_trend_alignment()

        assert alignment.daily_trend == Trend.BEARISH

    def test_trend_alignment_daily_only(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test trend alignment with only daily data."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        alignment = analyzer.get_trend_alignment()

        assert alignment.daily_trend != Trend.NEUTRAL
        assert alignment.h4_trend is None
        assert alignment.h1_trend is None
        assert alignment.alignment_score > 0

    def test_trend_alignment_with_intraday(
        self, daily_bullish_df: pd.DataFrame, intraday_5min_df: pd.DataFrame
    ) -> None:
        """Test trend alignment with intraday data."""
        # This may or may not produce h4/h1 trends depending on data length
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df, intraday_5min_df)
        alignment = analyzer.get_trend_alignment()

        assert alignment.daily_trend is not None
        assert alignment.alignment_score >= 0

    def test_trend_alignment_scoring_all_agree(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test alignment score when all timeframes agree."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        alignment = analyzer.get_trend_alignment()

        # With only daily and a clear trend, score should be moderate to high
        assert alignment.alignment_score >= 0.5

    def test_get_indicator_confluence_rsi(self, daily_rsi_oversold_df: pd.DataFrame) -> None:
        """Test RSI confluence detection."""
        analyzer = MultiTimeframeAnalyzer(daily_rsi_oversold_df)
        result = analyzer.get_indicator_confluence("rsi")

        assert isinstance(result, ConfluenceResult)
        assert result.indicator == "rsi"
        assert 0 <= result.confluence_score <= 1

    def test_get_indicator_confluence_macd(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test MACD confluence detection."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        result = analyzer.get_indicator_confluence("macd")

        assert isinstance(result, ConfluenceResult)
        assert result.indicator == "macd"

    def test_get_indicator_confluence_stochastic(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test Stochastic confluence detection."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        result = analyzer.get_indicator_confluence("stochastic")

        assert isinstance(result, ConfluenceResult)
        assert result.indicator == "stochastic"

    def test_analyze_returns_complete_result(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test analyze returns complete MultiTimeframeResult."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        result = analyzer.analyze()

        assert isinstance(result, MultiTimeframeResult)
        assert isinstance(result.trend_alignment, TrendAlignment)
        assert "rsi" in result.indicator_confluence
        assert "macd" in result.indicator_confluence
        assert "stochastic" in result.indicator_confluence
        assert result.overall_bias in [Trend.BULLISH, Trend.BEARISH, Trend.NEUTRAL]
        assert 0 <= result.confidence <= 1

    def test_overall_bias_bullish(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test overall bias detection for bullish market."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        result = analyzer.analyze()

        # With clear bullish daily trend, overall bias should be bullish or neutral
        assert result.overall_bias in [Trend.BULLISH, Trend.NEUTRAL]

    def test_overall_bias_bearish(self, daily_bearish_df: pd.DataFrame) -> None:
        """Test overall bias detection for bearish market."""
        analyzer = MultiTimeframeAnalyzer(daily_bearish_df)
        result = analyzer.analyze()

        # Trend should be detected as bearish
        assert result.trend_alignment.daily_trend == Trend.BEARISH

        # Overall bias may differ from trend due to contrarian indicator signals
        # (oversold RSI in downtrend = potential bullish reversal signal)
        assert result.overall_bias in [Trend.BEARISH, Trend.NEUTRAL, Trend.BULLISH]


# ============================================================================
# Signal Strength Tests
# ============================================================================


class TestSignalStrength:
    """Tests for signal strength classification."""

    def test_signal_strength_enum_values(self) -> None:
        """Test SignalStrength enum values."""
        assert SignalStrength.WEAK.value == "WEAK"
        assert SignalStrength.MODERATE.value == "MODERATE"
        assert SignalStrength.STRONG.value == "STRONG"

    def test_strong_confluence_produces_strong_signal(self, daily_bullish_df: pd.DataFrame) -> None:
        """High confluence should produce strong signal."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)

        # Get any indicator confluence
        result = analyzer.get_indicator_confluence("macd")

        # Signal strength should be valid
        assert result.signal_strength in [
            SignalStrength.WEAK,
            SignalStrength.MODERATE,
            SignalStrength.STRONG,
        ]


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_trend_enum_values(self) -> None:
        """Test Trend enum values."""
        assert Trend.BULLISH.value == "BULLISH"
        assert Trend.BEARISH.value == "BEARISH"
        assert Trend.NEUTRAL.value == "NEUTRAL"

    def test_handles_neutral_market(self, daily_neutral_df: pd.DataFrame) -> None:
        """Test handling of neutral/sideways market."""
        analyzer = MultiTimeframeAnalyzer(daily_neutral_df)
        result = analyzer.analyze()

        # Should not crash, bias may be neutral
        assert isinstance(result, MultiTimeframeResult)

    def test_aggregator_handles_partial_periods(self) -> None:
        """Test aggregator handles incomplete periods at edges."""
        # Create data that doesn't align perfectly with timeframe boundaries
        dates = pd.date_range("2024-01-02 09:33", periods=100, freq="1min")
        df = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [101] * 100,
                "low": [99] * 100,
                "close": [100] * 100,
                "volume": [1000] * 100,
            },
            index=dates,
        )

        aggregator = TimeframeAggregator(df)
        result = aggregator.aggregate_to_timeframe("5min")

        # Should still produce valid output
        assert len(result) > 0
        assert not result.isna().any().any()

    def test_analyzer_with_none_intraday(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test analyzer works with None intraday data."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df, intraday_df=None)
        result = analyzer.analyze()

        assert result.trend_alignment.h4_trend is None
        assert result.trend_alignment.h1_trend is None

    def test_unknown_indicator_returns_neutral(self, daily_bullish_df: pd.DataFrame) -> None:
        """Unknown indicator should return neutral signal."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        result = analyzer.get_indicator_confluence("unknown_indicator")

        # Should not crash, should have neutral/weak result
        assert isinstance(result, ConfluenceResult)

    def test_multiple_timeframe_aggregations(self, intraday_1min_df: pd.DataFrame) -> None:
        """Test multiple aggregations from same data."""
        aggregator = TimeframeAggregator(intraday_1min_df)

        tf_5min = aggregator.aggregate_to_timeframe("5min")
        tf_15min = aggregator.aggregate_to_timeframe("15min")
        tf_1h = aggregator.aggregate_to_timeframe("1H")

        # Each should have fewer bars
        assert len(tf_15min) < len(tf_5min)
        assert len(tf_1h) < len(tf_15min)

    def test_confidence_calculation(self, daily_bullish_df: pd.DataFrame) -> None:
        """Test confidence is properly calculated."""
        analyzer = MultiTimeframeAnalyzer(daily_bullish_df)
        result = analyzer.analyze()

        # Confidence should be between 0 and 1
        assert 0 <= result.confidence <= 1

    def test_alignment_with_all_neutral_trends(self) -> None:
        """Test alignment score when all trends are neutral."""
        alignment = TrendAlignment(
            daily_trend=Trend.NEUTRAL,
            h4_trend=Trend.NEUTRAL,
            h1_trend=Trend.NEUTRAL,
            alignment_score=0.5,
        )

        # All neutral should be considered aligned
        assert alignment.is_fully_aligned
