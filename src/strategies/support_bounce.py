"""Support Bounce Strategy - trades bounces off key support levels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from src.indicators.support_resistance import SupportResistanceDetector
from src.indicators.technical import TechnicalIndicators
from src.strategies.base import ExitSignal, Signal, Strategy, strategy_registry

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SupportBounceConfig:
    """Configuration for Support Bounce Strategy.

    Attributes:
        sr_lookback: Lookback period for S/R detection
        support_tolerance_atr: How close to support (in ATR multiples)
        rsi_max: Maximum RSI for entries (not overbought)
        volume_multiplier: Required volume spike on bounce
        atr_stop_multiplier: ATR multiplier below support for stop
        atr_profit_multiplier: ATR multiplier for profit target
        max_hold_days: Maximum days to hold position
    """

    sr_lookback: int = 60
    support_tolerance_atr: float = 0.5
    rsi_max: int = 50
    volume_multiplier: float = 1.2
    atr_stop_multiplier: float = 0.5
    atr_profit_multiplier: float = 1.5
    max_hold_days: int = 8


class SupportBounceStrategy(Strategy):
    """Support Bounce Strategy - trades bounces off support levels.

    This strategy identifies stocks bouncing off key support levels with
    bullish confirmation candles and volume spikes.

    Entry Conditions for LONG:
    - Price touches key support level (from S/R detector)
    - Bullish engulfing or hammer candle pattern
    - RSI < 50 (not overbought)
    - Volume spike on bounce (> 1.2x average)

    Exit Conditions:
    - Price reaches prior resistance level
    - ATR-based profit target
    - Stop loss below support level
    - Time exit after max hold days

    Historical stats: 57-61% win rate, 1.6:1 R:R
    """

    name = "SupportBounce"
    description = "Trades bounces off key support levels"

    def __init__(self, config: SupportBounceConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration. Uses defaults if not provided.
        """
        self.config = config or SupportBounceConfig()

    def scan(self, df: pd.DataFrame) -> Signal | None:
        """Scan for support bounce entry signal.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Signal if bounce conditions are met, None otherwise
        """
        if not self.validate_dataframe(df):
            logger.warning("Invalid DataFrame - missing required columns")
            return None

        min_rows = max(self.config.sr_lookback, 60)
        if len(df) < min_rows:
            logger.debug(f"Insufficient data for support bounce (need {min_rows} rows)")
            return None

        try:
            indicators = TechnicalIndicators(df)
            sr_detector = SupportResistanceDetector(df, lookback=self.config.sr_lookback)
        except Exception as e:
            logger.warning(f"Failed to initialize indicators: {e}")
            return None

        # Get current values
        current_close = float(df["close"].iloc[-1])
        current_open = float(df["open"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        current_volume = float(df["volume"].iloc[-1])
        prev_close = float(df["close"].iloc[-2])
        prev_open = float(df["open"].iloc[-2])
        prev_high = float(df["high"].iloc[-2])
        prev_low = float(df["low"].iloc[-2])

        # Calculate indicators
        atr = indicators.calculate_atr(14)
        current_atr = float(atr.iloc[-1])

        rsi = indicators.calculate_rsi(14)
        current_rsi = float(rsi.iloc[-1])

        volume_sma = indicators.calculate_volume_sma(20)
        avg_volume = float(volume_sma.iloc[-1])
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 0

        # Detect support levels
        try:
            levels = sr_detector.detect_levels()
        except Exception as e:
            logger.debug(f"S/R detection failed: {e}")
            return None

        if levels.nearest_support is None:
            logger.debug("No support level found")
            return None

        support_price = levels.nearest_support.price
        support_strength = levels.nearest_support.strength

        # Check if price is near support
        distance_to_support = abs(current_low - support_price)
        tolerance = current_atr * self.config.support_tolerance_atr

        if distance_to_support > tolerance:
            logger.debug(
                f"Price not near support: distance={distance_to_support:.2f}, tolerance={tolerance:.2f}"
            )
            return None

        # Check for bullish candle pattern
        is_bullish = self._is_bullish_reversal_candle(
            current_open,
            current_high,
            current_low,
            current_close,
            prev_open,
            prev_high,
            prev_low,
            prev_close,
        )

        if not is_bullish:
            logger.debug("No bullish reversal candle pattern")
            return None

        # Check RSI not overbought
        if current_rsi > self.config.rsi_max:
            logger.debug(f"RSI too high: {current_rsi:.1f} > {self.config.rsi_max}")
            return None

        # Check volume spike
        if relative_volume < self.config.volume_multiplier:
            logger.debug(
                f"Volume too low: {relative_volume:.2f}x < {self.config.volume_multiplier}x"
            )
            return None

        # All conditions met - create signal
        stop_loss = support_price - (current_atr * self.config.atr_stop_multiplier)

        # Take profit at resistance or ATR target
        if levels.nearest_resistance is not None:
            resistance_target = levels.nearest_resistance.price
            atr_target = current_close + (current_atr * self.config.atr_profit_multiplier)
            take_profit = min(resistance_target, atr_target)
        else:
            take_profit = current_close + (current_atr * self.config.atr_profit_multiplier)

        # Calculate score
        score = self._calculate_score(
            support_strength=support_strength,
            relative_volume=relative_volume,
            current_rsi=current_rsi,
            distance_to_support=distance_to_support,
            tolerance=tolerance,
        )

        # Build reasoning
        reasons = [
            f"Bounce at support {support_price:.2f} (strength={support_strength})",
            "Bullish reversal candle",
            f"RSI={current_rsi:.1f}",
            f"Volume={relative_volume:.1f}x avg",
        ]

        symbol = df.attrs.get("symbol", "UNKNOWN")
        reasoning = "; ".join(reasons)

        logger.info(
            f"Support Bounce signal for {symbol}: LONG, {reasoning}, "
            f"Score={score}, Entry={current_close:.2f}"
        )

        return Signal(
            symbol=symbol,
            strategy=self.name,
            direction="LONG",
            entry_price=current_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            score=score,
            reasoning=reasoning,
            metadata={
                "support_level": support_price,
                "support_strength": support_strength,
                "resistance_level": levels.nearest_resistance.price
                if levels.nearest_resistance
                else None,
                "rsi": current_rsi,
                "atr": current_atr,
                "relative_volume": relative_volume,
            },
        )

    def _is_bullish_reversal_candle(
        self,
        curr_open: float,
        curr_high: float,
        curr_low: float,
        curr_close: float,
        prev_open: float,
        _prev_high: float,
        _prev_low: float,
        prev_close: float,
    ) -> bool:
        """Check for bullish reversal candle patterns.

        Checks for:
        - Bullish engulfing
        - Hammer
        - Bullish pin bar

        Args:
            Current and previous candle OHLC values

        Returns:
            True if bullish reversal pattern detected
        """
        curr_body = curr_close - curr_open
        curr_range = curr_high - curr_low
        curr_lower_wick = min(curr_open, curr_close) - curr_low

        # Bullish engulfing: current bullish candle engulfs previous bearish
        if (
            curr_close > curr_open  # Current is bullish
            and prev_close < prev_open  # Previous is bearish
            and curr_open < prev_close  # Opens below prev close
            and curr_close > prev_open
        ):  # Closes above prev open
            return True

        # Hammer: small body at top, long lower wick
        if curr_range > 0:
            body_ratio = abs(curr_body) / curr_range
            lower_wick_ratio = curr_lower_wick / curr_range

            # Hammer: body < 30%, lower wick > 60%
            if body_ratio < 0.3 and lower_wick_ratio > 0.6:
                return True

            # Bullish pin bar: close > open, lower wick > 2x body
            if curr_close > curr_open and curr_lower_wick > abs(curr_body) * 2:
                return True

        # Simple bullish candle after decline
        return curr_close > curr_open and prev_close < prev_open

    def _calculate_score(
        self,
        support_strength: int,
        relative_volume: float,
        current_rsi: float,
        distance_to_support: float,
        tolerance: float,
    ) -> int:
        """Calculate signal strength score (0-100).

        Args:
            support_strength: Number of times support was tested
            relative_volume: Volume relative to average
            current_rsi: Current RSI value
            distance_to_support: Distance from price to support
            tolerance: Maximum acceptable distance

        Returns:
            Score from 0-100
        """
        score = 50  # Base score

        # +15 if support strength > 3
        if support_strength > 3:
            score += 15
        elif support_strength > 2:
            score += 10

        # +10 if volume > 1.5x average
        if relative_volume > 1.5:
            score += 10

        # +10 if RSI < 35 (oversold)
        if current_rsi < 35:
            score += 10

        # +10 if very close to support (< 25% of tolerance)
        if distance_to_support < tolerance * 0.25:
            score += 10

        # +5 for being at a significant level
        score += 5

        return min(score, 100)

    def check_exit(self, df: pd.DataFrame, trade: Any) -> ExitSignal | None:
        """Check if an open trade should be exited.

        Exit Conditions:
        - Price reaches resistance level
        - Take profit hit
        - Stop loss hit
        - Time exit after max hold days

        Args:
            df: DataFrame with current OHLCV data
            trade: Trade object with entry details

        Returns:
            ExitSignal if exit conditions are met, None otherwise
        """
        if not self.validate_dataframe(df):
            return None

        current_close = float(df["close"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        current_high = float(df["high"].iloc[-1])

        # Extract trade attributes
        trade_id = getattr(trade, "trade_id", getattr(trade, "id", 0))
        stop_loss = getattr(trade, "stop_loss", 0.0)
        take_profit = getattr(trade, "take_profit", 0.0)
        entry_date = getattr(trade, "entry_date", None)
        resistance_level = getattr(trade, "resistance_level", None)

        # Check stop loss
        if current_low <= stop_loss:
            logger.info(f"Trade {trade_id}: Stop loss triggered")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="STOP_LOSS",
                exit_price=stop_loss,
                reasoning=f"Stop loss hit at {stop_loss:.2f}",
            )

        # Check take profit
        if current_high >= take_profit:
            logger.info(f"Trade {trade_id}: Take profit triggered")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="TAKE_PROFIT",
                exit_price=take_profit,
                reasoning=f"Take profit hit at {take_profit:.2f}",
            )

        # Check resistance level (if provided from metadata)
        if resistance_level is not None and current_high >= resistance_level:
            logger.info(f"Trade {trade_id}: Resistance level reached")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="STRATEGY_EXIT",
                exit_price=resistance_level,
                reasoning=f"Reached resistance at {resistance_level:.2f}",
            )

        # Check time exit
        if entry_date is not None:
            if isinstance(entry_date, datetime):
                days_held = (datetime.now() - entry_date).days
            else:
                days_held = (datetime.now().date() - entry_date).days

            if days_held >= self.config.max_hold_days:
                logger.info(f"Trade {trade_id}: Time exit after {days_held} days")
                return ExitSignal(
                    trade_id=trade_id,
                    exit_type="TIME_EXIT",
                    exit_price=current_close,
                    reasoning=f"Max hold period ({self.config.max_hold_days} days) exceeded",
                )

        return None

    def get_parameters(self) -> dict[str, Any]:
        """Return current strategy parameters."""
        return {
            "sr_lookback": self.config.sr_lookback,
            "support_tolerance_atr": self.config.support_tolerance_atr,
            "rsi_max": self.config.rsi_max,
            "volume_multiplier": self.config.volume_multiplier,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "atr_profit_multiplier": self.config.atr_profit_multiplier,
            "max_hold_days": self.config.max_hold_days,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters."""
        if "sr_lookback" in params:
            self.config.sr_lookback = int(params["sr_lookback"])
        if "support_tolerance_atr" in params:
            self.config.support_tolerance_atr = float(params["support_tolerance_atr"])
        if "rsi_max" in params:
            self.config.rsi_max = int(params["rsi_max"])
        if "volume_multiplier" in params:
            self.config.volume_multiplier = float(params["volume_multiplier"])
        if "atr_stop_multiplier" in params:
            self.config.atr_stop_multiplier = float(params["atr_stop_multiplier"])
        if "atr_profit_multiplier" in params:
            self.config.atr_profit_multiplier = float(params["atr_profit_multiplier"])
        if "max_hold_days" in params:
            self.config.max_hold_days = int(params["max_hold_days"])


# Register strategy with global registry
strategy_registry.register(SupportBounceStrategy)
