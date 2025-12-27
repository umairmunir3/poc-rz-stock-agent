"""Multi-timeframe analysis module."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from src.indicators.exceptions import InsufficientDataError, InvalidDataError
from src.indicators.technical import TechnicalIndicators


class Trend(str, Enum):
    """Trend direction."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(str, Enum):
    """Signal strength level."""

    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


@dataclass
class TrendAlignment:
    """Trend alignment across multiple timeframes.

    Attributes:
        daily_trend: Trend direction on daily timeframe
        h4_trend: Trend direction on 4-hour timeframe (if available)
        h1_trend: Trend direction on 1-hour timeframe (if available)
        alignment_score: Score from 0-1 (higher = more aligned)
    """

    daily_trend: Trend
    h4_trend: Trend | None = None
    h1_trend: Trend | None = None
    alignment_score: float = 0.0

    def __post_init__(self) -> None:
        """Validate alignment score."""
        if not 0.0 <= self.alignment_score <= 1.0:
            raise ValueError("alignment_score must be between 0 and 1")

    @property
    def is_fully_aligned(self) -> bool:
        """Check if all available timeframes agree on trend direction."""
        trends = [self.daily_trend]
        if self.h4_trend is not None:
            trends.append(self.h4_trend)
        if self.h1_trend is not None:
            trends.append(self.h1_trend)

        # All non-neutral trends should be the same
        non_neutral = [t for t in trends if t != Trend.NEUTRAL]
        if len(non_neutral) <= 1:
            return True
        return all(t == non_neutral[0] for t in non_neutral)


@dataclass
class ConfluenceResult:
    """Result of indicator confluence analysis.

    Attributes:
        indicator: Name of the indicator analyzed
        timeframes_agreeing: List of timeframes with agreeing signals
        confluence_score: Score from 0-1 (higher = more confluence)
        signal_strength: Overall signal strength
    """

    indicator: str
    timeframes_agreeing: list[str] = field(default_factory=list)
    confluence_score: float = 0.0
    signal_strength: SignalStrength = SignalStrength.WEAK

    def __post_init__(self) -> None:
        """Validate confluence score."""
        if not 0.0 <= self.confluence_score <= 1.0:
            raise ValueError("confluence_score must be between 0 and 1")


@dataclass
class MultiTimeframeResult:
    """Overall multi-timeframe analysis result.

    Attributes:
        trend_alignment: Trend alignment across timeframes
        indicator_confluence: Dictionary of indicator confluence results
        overall_bias: Overall market bias
        confidence: Confidence level (0-1)
    """

    trend_alignment: TrendAlignment
    indicator_confluence: dict[str, ConfluenceResult] = field(default_factory=dict)
    overall_bias: Trend = Trend.NEUTRAL
    confidence: float = 0.0


class TimeframeAggregator:
    """Aggregates intraday data to higher timeframes.

    Converts fine-grained data (1-min, 5-min bars) to coarser timeframes
    while correctly handling OHLCV aggregation.

    Attributes:
        df: DataFrame with intraday OHLCV data
    """

    SUPPORTED_TIMEFRAMES = {"5min", "15min", "30min", "1H", "4H", "1D"}
    TIMEFRAME_MINUTES = {
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "1H": 60,
        "4H": 240,
        "1D": 1440,
    }
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with intraday data.

        Args:
            df: DataFrame with OHLCV columns and datetime index

        Raises:
            InvalidDataError: If required columns are missing or index is not datetime
        """
        self._validate_input(df)
        self.df = df.copy()

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if df.empty:
            raise InvalidDataError("DataFrame is empty")

        df_columns = {col.lower() for col in df.columns}
        missing = self.REQUIRED_COLUMNS - df_columns
        if missing:
            raise InvalidDataError(f"Missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise InvalidDataError("DataFrame index must be DatetimeIndex")

    def aggregate_to_timeframe(self, timeframe: str) -> pd.DataFrame:
        """Aggregate data to specified timeframe.

        Args:
            timeframe: Target timeframe (5min, 15min, 30min, 1H, 4H, 1D)

        Returns:
            DataFrame with aggregated OHLCV data

        Raises:
            ValueError: If timeframe is not supported
        """
        if timeframe not in self.SUPPORTED_TIMEFRAMES:
            raise ValueError(
                f"Unsupported timeframe: {timeframe}. Supported: {self.SUPPORTED_TIMEFRAMES}"
            )

        # Map timeframe string to pandas offset
        offset_map = {
            "5min": "5min",
            "15min": "15min",
            "30min": "30min",
            "1H": "1h",
            "4H": "4h",
            "1D": "1D",
        }

        offset = offset_map[timeframe]

        # Aggregate using proper OHLCV logic
        aggregated = self.df.resample(offset).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

        # Drop rows with NaN (incomplete periods at edges)
        aggregated = aggregated.dropna()

        return aggregated

    def get_available_timeframes(self) -> list[str]:
        """Get list of timeframes that can be created from this data.

        Returns:
            List of valid timeframe strings based on data frequency
        """
        if len(self.df) < 2:
            return []

        # Estimate the base frequency from the data
        time_diff = (self.df.index[1] - self.df.index[0]).total_seconds() / 60
        base_minutes = int(time_diff)

        available = []
        for tf, minutes in self.TIMEFRAME_MINUTES.items():
            if minutes >= base_minutes:
                available.append(tf)

        return available


class MultiTimeframeAnalyzer:
    """Analyzes trends and indicators across multiple timeframes.

    Combines daily and intraday data to provide confluence analysis
    and trend alignment scoring.

    Attributes:
        daily_df: DataFrame with daily OHLCV data
        intraday_df: Optional DataFrame with intraday data
    """

    MIN_DAILY_BARS = 50  # Need enough for EMA 50

    def __init__(
        self,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame | None = None,
    ) -> None:
        """Initialize with daily and optional intraday data.

        Args:
            daily_df: DataFrame with daily OHLCV data
            intraday_df: Optional DataFrame with intraday data

        Raises:
            InsufficientDataError: If not enough daily data
        """
        self._validate_daily(daily_df)
        self.daily_df = daily_df.copy()
        self.intraday_df = intraday_df.copy() if intraday_df is not None else None

        # Pre-compute indicators for daily
        self._daily_indicators = TechnicalIndicators(self.daily_df)

        # Aggregate intraday to different timeframes if available
        self._h4_df: pd.DataFrame | None = None
        self._h1_df: pd.DataFrame | None = None

        if self.intraday_df is not None:
            self._setup_intraday_timeframes()

    def _validate_daily(self, df: pd.DataFrame) -> None:
        """Validate daily DataFrame."""
        if df.empty:
            raise InvalidDataError("Daily DataFrame is empty")

        if len(df) < self.MIN_DAILY_BARS:
            raise InsufficientDataError(
                required=self.MIN_DAILY_BARS,
                available=len(df),
                indicator="MultiTimeframeAnalyzer",
            )

    def _setup_intraday_timeframes(self) -> None:
        """Set up aggregated intraday timeframes."""
        if self.intraday_df is None:
            return

        try:
            aggregator = TimeframeAggregator(self.intraday_df)

            # Try to create 4H and 1H aggregations
            if "4H" in aggregator.get_available_timeframes():
                h4 = aggregator.aggregate_to_timeframe("4H")
                if len(h4) >= 20:  # Need enough bars for indicators
                    self._h4_df = h4

            if "1H" in aggregator.get_available_timeframes():
                h1 = aggregator.aggregate_to_timeframe("1H")
                if len(h1) >= 20:
                    self._h1_df = h1
        except (InvalidDataError, ValueError):
            # If aggregation fails, continue without intraday
            pass

    def _determine_trend(self, df: pd.DataFrame) -> Trend:
        """Determine trend for a single timeframe.

        Trend is determined by:
        1. Price vs EMA 50
        2. EMA 9 vs EMA 21
        3. MACD histogram direction
        """
        if len(df) < 50:
            return Trend.NEUTRAL

        ti = TechnicalIndicators(df)

        try:
            ema_9 = ti.calculate_ema(9)
            ema_21 = ti.calculate_ema(21)
            ema_50 = ti.calculate_ema(50)
            _, _, macd_hist = ti.calculate_macd()
        except InsufficientDataError:
            return Trend.NEUTRAL

        current_close = df["close"].iloc[-1]
        current_ema_9 = ema_9.iloc[-1]
        current_ema_21 = ema_21.iloc[-1]
        current_ema_50 = ema_50.iloc[-1]
        current_macd_hist = macd_hist.iloc[-1]

        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0

        # Signal 1: Price vs EMA 50
        if current_close > current_ema_50:
            bullish_count += 1
        elif current_close < current_ema_50:
            bearish_count += 1

        # Signal 2: EMA 9 vs EMA 21
        if current_ema_9 > current_ema_21:
            bullish_count += 1
        elif current_ema_9 < current_ema_21:
            bearish_count += 1

        # Signal 3: MACD histogram direction
        if not pd.isna(current_macd_hist):
            if current_macd_hist > 0:
                bullish_count += 1
            elif current_macd_hist < 0:
                bearish_count += 1

        # Determine trend based on majority
        if bullish_count >= 2:
            return Trend.BULLISH
        elif bearish_count >= 2:
            return Trend.BEARISH
        else:
            return Trend.NEUTRAL

    def get_trend_alignment(self) -> TrendAlignment:
        """Get trend alignment across all available timeframes.

        Returns:
            TrendAlignment with trends and alignment score
        """
        daily_trend = self._determine_trend(self.daily_df)

        h4_trend = None
        h1_trend = None

        if self._h4_df is not None:
            h4_trend = self._determine_trend(self._h4_df)

        if self._h1_df is not None:
            h1_trend = self._determine_trend(self._h1_df)

        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(daily_trend, h4_trend, h1_trend)

        return TrendAlignment(
            daily_trend=daily_trend,
            h4_trend=h4_trend,
            h1_trend=h1_trend,
            alignment_score=alignment_score,
        )

    def _calculate_alignment_score(
        self,
        daily: Trend,
        h4: Trend | None,
        h1: Trend | None,
    ) -> float:
        """Calculate alignment score based on trend agreement.

        Score is 1.0 when all trends agree, 0.0 when completely opposed.
        """
        trends = [daily]
        if h4 is not None:
            trends.append(h4)
        if h1 is not None:
            trends.append(h1)

        if len(trends) == 1:
            # Only daily, score based on strength (non-neutral = higher)
            return 0.7 if daily != Trend.NEUTRAL else 0.5

        # Count trend occurrences
        bullish_count = sum(1 for t in trends if t == Trend.BULLISH)
        bearish_count = sum(1 for t in trends if t == Trend.BEARISH)
        neutral_count = sum(1 for t in trends if t == Trend.NEUTRAL)

        total = len(trends)

        # Perfect alignment (all same non-neutral)
        if bullish_count == total or bearish_count == total:
            return 1.0

        # Mostly aligned with some neutral
        max_directional = max(bullish_count, bearish_count)
        if max_directional > 0 and neutral_count > 0:
            # Partial alignment
            return 0.5 + (max_directional / total) * 0.4

        # Mixed signals (bullish and bearish present)
        if bullish_count > 0 and bearish_count > 0:
            # Conflict - lower score
            return 0.3 - (min(bullish_count, bearish_count) / total) * 0.2

        # All neutral
        return 0.5

    def get_indicator_confluence(self, indicator: str) -> ConfluenceResult:
        """Analyze confluence for a specific indicator across timeframes.

        Args:
            indicator: Indicator name (rsi, macd, stochastic)

        Returns:
            ConfluenceResult with agreement analysis
        """
        timeframes_agreeing: list[str] = []
        signals: list[str] = []

        # Analyze each available timeframe
        timeframe_data = [
            ("daily", self.daily_df),
            ("4H", self._h4_df),
            ("1H", self._h1_df),
        ]

        for tf_name, df in timeframe_data:
            if df is None or len(df) < 35:  # Need enough for MACD
                continue

            signal = self._get_indicator_signal(df, indicator)
            if signal:
                signals.append(signal)
                if signal != "NEUTRAL":
                    timeframes_agreeing.append(tf_name)

        # Calculate confluence score
        if not signals:
            return ConfluenceResult(
                indicator=indicator,
                timeframes_agreeing=[],
                confluence_score=0.0,
                signal_strength=SignalStrength.WEAK,
            )

        # Count agreeing signals
        bullish = sum(1 for s in signals if s == "BULLISH")
        bearish = sum(1 for s in signals if s == "BEARISH")

        total_signals = len(signals)
        max_agreement = max(bullish, bearish)

        confluence_score = max_agreement / total_signals if total_signals > 0 else 0.0

        # Determine signal strength
        if confluence_score >= 0.8:
            strength = SignalStrength.STRONG
        elif confluence_score >= 0.5:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK

        return ConfluenceResult(
            indicator=indicator,
            timeframes_agreeing=timeframes_agreeing,
            confluence_score=confluence_score,
            signal_strength=strength,
        )

    def _get_indicator_signal(self, df: pd.DataFrame, indicator: str) -> str:
        """Get signal from a specific indicator on a timeframe."""
        try:
            ti = TechnicalIndicators(df)

            if indicator.lower() == "rsi":
                rsi = ti.calculate_rsi(14)
                current_rsi = rsi.iloc[-1]
                if pd.isna(current_rsi):
                    return "NEUTRAL"
                if current_rsi < 30:
                    return "BULLISH"  # Oversold = potential buy
                elif current_rsi > 70:
                    return "BEARISH"  # Overbought = potential sell
                return "NEUTRAL"

            elif indicator.lower() == "macd":
                _, _, hist = ti.calculate_macd()
                current_hist = hist.iloc[-1]
                prev_hist = hist.iloc[-2] if len(hist) > 1 else 0

                if pd.isna(current_hist):
                    return "NEUTRAL"
                # Bullish if histogram positive and increasing
                if current_hist > 0 and current_hist > prev_hist:
                    return "BULLISH"
                elif current_hist < 0 and current_hist < prev_hist:
                    return "BEARISH"
                return "NEUTRAL"

            elif indicator.lower() == "stochastic":
                stoch_k, _ = ti.calculate_stochastic()
                current_k = stoch_k.iloc[-1]
                if pd.isna(current_k):
                    return "NEUTRAL"
                if current_k < 20:
                    return "BULLISH"  # Oversold
                elif current_k > 80:
                    return "BEARISH"  # Overbought
                return "NEUTRAL"

            else:
                return "NEUTRAL"

        except (InsufficientDataError, InvalidDataError):
            return "NEUTRAL"

    def analyze(self) -> MultiTimeframeResult:
        """Perform full multi-timeframe analysis.

        Returns:
            MultiTimeframeResult with overall bias and confidence
        """
        trend_alignment = self.get_trend_alignment()

        # Analyze key indicators
        indicator_confluence = {
            "rsi": self.get_indicator_confluence("rsi"),
            "macd": self.get_indicator_confluence("macd"),
            "stochastic": self.get_indicator_confluence("stochastic"),
        }

        # Determine overall bias
        overall_bias = self._determine_overall_bias(trend_alignment, indicator_confluence)

        # Calculate confidence
        confidence = self._calculate_confidence(trend_alignment, indicator_confluence)

        return MultiTimeframeResult(
            trend_alignment=trend_alignment,
            indicator_confluence=indicator_confluence,
            overall_bias=overall_bias,
            confidence=confidence,
        )

    def _determine_overall_bias(
        self,
        trend: TrendAlignment,
        confluence: dict[str, ConfluenceResult],
    ) -> Trend:
        """Determine overall market bias from all signals."""
        bullish_weight = 0.0
        bearish_weight = 0.0

        # Weight from trend alignment
        if trend.daily_trend == Trend.BULLISH:
            bullish_weight += 2.0 * trend.alignment_score
        elif trend.daily_trend == Trend.BEARISH:
            bearish_weight += 2.0 * trend.alignment_score

        # Weight from indicator confluence
        for result in confluence.values():
            if result.signal_strength == SignalStrength.STRONG:
                weight = 1.0
            elif result.signal_strength == SignalStrength.MODERATE:
                weight = 0.5
            else:
                weight = 0.2

            # Check if agreeing timeframes lean bullish or bearish
            if result.timeframes_agreeing:
                # Use the indicator signal on daily as primary
                daily_signal = self._get_indicator_signal(self.daily_df, result.indicator)
                if daily_signal == "BULLISH":
                    bullish_weight += weight * result.confluence_score
                elif daily_signal == "BEARISH":
                    bearish_weight += weight * result.confluence_score

        # Determine bias
        if bullish_weight > bearish_weight * 1.2:  # Need 20% more to confirm
            return Trend.BULLISH
        elif bearish_weight > bullish_weight * 1.2:
            return Trend.BEARISH
        else:
            return Trend.NEUTRAL

    def _calculate_confidence(
        self,
        trend: TrendAlignment,
        confluence: dict[str, ConfluenceResult],
    ) -> float:
        """Calculate confidence in the overall analysis."""
        scores = [trend.alignment_score]

        for result in confluence.values():
            scores.append(result.confluence_score)

        # Average of all scores
        if not scores:
            return 0.0

        return sum(scores) / len(scores)
