"""Volume Breakout Strategy - trades volume-confirmed breakouts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from src.indicators.technical import TechnicalIndicators
from src.strategies.base import ExitSignal, Signal, Strategy, strategy_registry

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BreakoutConfig:
    """Configuration for Breakout Strategy.

    Attributes:
        lookback_period: Period for high/low detection
        volume_multiplier: Required volume vs average
        atr_stop_multiplier: ATR multiplier for stop loss distance
        trailing_stop_atr: ATR multiplier for trailing stop
        max_hold_days: Maximum days to hold position
        min_atr_percent: Minimum ATR as percentage of price
        close_range_threshold: Required close position in day's range (0.25 = top/bottom 25%)
    """

    lookback_period: int = 20
    volume_multiplier: float = 1.5
    atr_stop_multiplier: float = 2.0
    trailing_stop_atr: float = 2.0
    max_hold_days: int = 5
    min_atr_percent: float = 1.0
    close_range_threshold: float = 0.25


class BreakoutStrategy(Strategy):
    """Volume Breakout Strategy - trades volume-confirmed breakouts.

    This strategy identifies stocks breaking out of consolidation ranges
    with strong volume confirmation, betting on continuation momentum.

    Entry Conditions for LONG breakout:
    - Close > highest high of last N days
    - Volume > X times 20-day average volume
    - Close in upper 25% of day's range (strong close)
    - ATR > 1% of price (enough volatility)

    Entry Conditions for SHORT breakout:
    - Close < lowest low of last N days
    - Volume > X times 20-day average
    - Close in lower 25% of day's range

    Exit Conditions:
    - Trailing stop based on ATR
    - Time exit after max hold days
    - Move to breakeven if up > 1R
    """

    name = "VolumeBreakout"
    description = "Trades volume-confirmed price breakouts"

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration. Uses defaults if not provided.
        """
        self.config = config or BreakoutConfig()

    def scan(self, df: pd.DataFrame) -> Signal | None:
        """Scan for breakout entry signal.

        Args:
            df: DataFrame with OHLCV data (needs lookback_period + 20 rows)

        Returns:
            Signal if breakout conditions are met, None otherwise
        """
        if not self.validate_dataframe(df):
            logger.warning("Invalid DataFrame - missing required columns")
            return None

        min_rows = self.config.lookback_period + 20
        if len(df) < min_rows:
            logger.debug(f"Insufficient data for breakout strategy (need {min_rows} rows)")
            return None

        try:
            indicators = TechnicalIndicators(df)
        except Exception as e:
            logger.warning(f"Failed to initialize indicators: {e}")
            return None

        # Get current values
        current_close = float(df["close"].iloc[-1])
        current_high = float(df["high"].iloc[-1])
        current_low = float(df["low"].iloc[-1])
        current_open = float(df["open"].iloc[-1])
        current_volume = float(df["volume"].iloc[-1])

        # Calculate indicators
        atr = indicators.calculate_atr(14)
        current_atr = float(atr.iloc[-1])

        volume_sma = indicators.calculate_volume_sma(20)
        avg_volume = float(volume_sma.iloc[-1])

        # Calculate highest high and lowest low over lookback period (excluding today)
        lookback_df = df.iloc[-(self.config.lookback_period + 1) : -1]
        highest_high = float(lookback_df["high"].max())
        lowest_low = float(lookback_df["low"].min())

        # Calculate relative volume
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 0

        # Calculate close position in day's range
        day_range = current_high - current_low
        close_position = (current_close - current_low) / day_range if day_range > 0 else 0.5

        # Calculate ATR as percentage of price
        atr_percent = (current_atr / current_close) * 100 if current_close > 0 else 0

        # Check for LONG breakout
        long_signal = self._check_long_breakout(
            current_close=current_close,
            highest_high=highest_high,
            relative_volume=relative_volume,
            close_position=close_position,
            atr_percent=atr_percent,
        )

        # Check for SHORT breakout
        short_signal = self._check_short_breakout(
            current_close=current_close,
            lowest_low=lowest_low,
            relative_volume=relative_volume,
            close_position=close_position,
            atr_percent=atr_percent,
        )

        if long_signal:
            return self._create_signal(
                df=df,
                direction="LONG",
                entry_price=current_close,
                current_atr=current_atr,
                relative_volume=relative_volume,
                highest_high=highest_high,
                lowest_low=lowest_low,
                current_open=current_open,
            )
        elif short_signal:
            return self._create_signal(
                df=df,
                direction="SHORT",
                entry_price=current_close,
                current_atr=current_atr,
                relative_volume=relative_volume,
                highest_high=highest_high,
                lowest_low=lowest_low,
                current_open=current_open,
            )

        return None

    def _check_long_breakout(
        self,
        current_close: float,
        highest_high: float,
        relative_volume: float,
        close_position: float,
        atr_percent: float,
    ) -> bool:
        """Check if conditions are met for LONG breakout.

        Args:
            current_close: Current closing price
            highest_high: Highest high over lookback period
            relative_volume: Current volume relative to average
            close_position: Close position in day's range (0-1)
            atr_percent: ATR as percentage of price

        Returns:
            True if all LONG breakout conditions are met
        """
        # Condition 1: Close above highest high
        if current_close <= highest_high:
            return False

        # Condition 2: Volume confirmation
        if relative_volume < self.config.volume_multiplier:
            return False

        # Condition 3: Strong close (upper 25% of range)
        if close_position < (1 - self.config.close_range_threshold):
            return False

        # Condition 4: Sufficient volatility
        return atr_percent >= self.config.min_atr_percent

    def _check_short_breakout(
        self,
        current_close: float,
        lowest_low: float,
        relative_volume: float,
        close_position: float,
        atr_percent: float,
    ) -> bool:
        """Check if conditions are met for SHORT breakout.

        Args:
            current_close: Current closing price
            lowest_low: Lowest low over lookback period
            relative_volume: Current volume relative to average
            close_position: Close position in day's range (0-1)
            atr_percent: ATR as percentage of price

        Returns:
            True if all SHORT breakout conditions are met
        """
        # Condition 1: Close below lowest low
        if current_close >= lowest_low:
            return False

        # Condition 2: Volume confirmation
        if relative_volume < self.config.volume_multiplier:
            return False

        # Condition 3: Weak close (lower 25% of range)
        if close_position > self.config.close_range_threshold:
            return False

        # Condition 4: Sufficient volatility
        return atr_percent >= self.config.min_atr_percent

    def _create_signal(
        self,
        df: pd.DataFrame,
        direction: Literal["LONG", "SHORT"],
        entry_price: float,
        current_atr: float,
        relative_volume: float,
        highest_high: float,
        lowest_low: float,
        current_open: float,
    ) -> Signal:
        """Create a trading signal.

        Args:
            df: OHLCV DataFrame
            direction: Trade direction
            entry_price: Entry price
            current_atr: Current ATR value
            relative_volume: Volume relative to average
            highest_high: Highest high over lookback
            lowest_low: Lowest low over lookback
            current_open: Current open price

        Returns:
            Signal object
        """
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(
            entry_price=entry_price,
            atr=current_atr,
            direction=direction,
            multiplier=self.config.atr_stop_multiplier,
        )

        # For breakout, use 3:1 R:R for initial take profit target
        take_profit = self.calculate_take_profit(
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction,
            risk_reward=3.0,
        )

        # Calculate score
        score = self._calculate_score(
            df=df,
            direction=direction,
            relative_volume=relative_volume,
            highest_high=highest_high,
            lowest_low=lowest_low,
            entry_price=entry_price,
            current_open=current_open,
        )

        # Build reasoning
        reasons = []
        if direction == "LONG":
            reasons.append(f"Breakout above {highest_high:.2f}")
        else:
            reasons.append(f"Breakdown below {lowest_low:.2f}")
        reasons.append(f"Volume={relative_volume:.1f}x avg")
        reasons.append(f"ATR={current_atr:.2f}")

        symbol = df.attrs.get("symbol", "UNKNOWN")
        reasoning = "; ".join(reasons)

        logger.info(
            f"Breakout signal for {symbol}: {direction}, {reasoning}, "
            f"Score={score}, Entry={entry_price:.2f}, SL={stop_loss:.2f}"
        )

        return Signal(
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            score=score,
            reasoning=reasoning,
            metadata={
                "atr": current_atr,
                "relative_volume": relative_volume,
                "highest_high": highest_high,
                "lowest_low": lowest_low,
                "breakout_level": highest_high if direction == "LONG" else lowest_low,
            },
        )

    def _calculate_score(
        self,
        df: pd.DataFrame,
        direction: Literal["LONG", "SHORT"],
        relative_volume: float,
        highest_high: float,
        lowest_low: float,
        entry_price: float,
        current_open: float,
    ) -> int:
        """Calculate signal strength score (0-100).

        Args:
            df: OHLCV DataFrame
            direction: Trade direction
            relative_volume: Volume relative to average
            highest_high: Highest high over lookback
            lowest_low: Lowest low over lookback
            entry_price: Entry price
            current_open: Open price

        Returns:
            Score from 0-100
        """
        score = 50  # Base score

        # +15 if volume > 2x average
        if relative_volume > 2.0:
            score += 15

        # +10 if breaking multi-month high/low (check if beyond 60-day high/low)
        if len(df) >= 60:
            sixty_day_high = float(df["high"].iloc[-60:-1].max())
            sixty_day_low = float(df["low"].iloc[-60:-1].min())

            if (direction == "LONG" and entry_price > sixty_day_high) or (
                direction == "SHORT" and entry_price < sixty_day_low
            ):
                score += 10

        # +10 if gap up/down on breakout
        if (direction == "LONG" and current_open > highest_high) or (
            direction == "SHORT" and current_open < lowest_low
        ):
            score += 10

        # +5 for consolidation prior to breakout
        if self.detect_consolidation(df):
            score += 5

        return min(score, 100)

    def detect_consolidation(
        self, df: pd.DataFrame, min_days: int = 10, max_range_percent: float = 10.0
    ) -> bool:
        """Detect if price is in consolidation (compressed range).

        A good breakout setup often occurs after a period of consolidation
        where price trades in a tight range.

        Args:
            df: OHLCV DataFrame
            min_days: Minimum days to check for consolidation
            max_range_percent: Maximum range as percentage of price for consolidation

        Returns:
            True if price is in consolidation pattern
        """
        if len(df) < min_days + 1:
            return False

        # Look at the period before today
        consolidation_df = df.iloc[-(min_days + 1) : -1]

        high = float(consolidation_df["high"].max())
        low = float(consolidation_df["low"].min())
        mid_price = (high + low) / 2

        if mid_price <= 0:
            return False

        range_percent = ((high - low) / mid_price) * 100

        return range_percent <= max_range_percent

    def check_exit(self, df: pd.DataFrame, trade: Any) -> ExitSignal | None:
        """Check if an open trade should be exited.

        Exit Conditions:
        - Trailing stop based on ATR
        - Time exit after max hold days
        - Move to breakeven if up > 1R

        Args:
            df: DataFrame with current OHLCV data
            trade: Trade object with entry_price, stop_loss, take_profit,
                   trade_id, entry_date, direction, and highest_price_since_entry

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
        entry_price = getattr(trade, "entry_price", 0.0)
        initial_stop = getattr(trade, "stop_loss", 0.0)
        direction = getattr(trade, "direction", "LONG")
        entry_date = getattr(trade, "entry_date", None)
        highest_since_entry = getattr(trade, "highest_price_since_entry", entry_price)
        lowest_since_entry = getattr(trade, "lowest_price_since_entry", entry_price)

        # Calculate current ATR for trailing stop
        try:
            indicators = TechnicalIndicators(df)
            atr = float(indicators.calculate_atr(14).iloc[-1])
        except Exception:
            atr = abs(entry_price - initial_stop) / self.config.atr_stop_multiplier

        # Update highest/lowest since entry
        if direction == "LONG":
            highest_since_entry = max(highest_since_entry, current_high)
        else:
            lowest_since_entry = min(lowest_since_entry, current_low)

        # Calculate trailing stop
        if direction == "LONG":
            trailing_stop = highest_since_entry - (atr * self.config.trailing_stop_atr)
            # Never let winner turn into loser: if up > 1R, move to breakeven
            risk = entry_price - initial_stop
            if highest_since_entry - entry_price > risk:
                trailing_stop = max(trailing_stop, entry_price)
            current_stop = max(initial_stop, trailing_stop)

            # Check stop loss hit
            if current_low <= current_stop:
                exit_type = "TRAILING_STOP" if current_stop > initial_stop else "STOP_LOSS"
                logger.info(f"Trade {trade_id}: {exit_type} at {current_stop:.2f}")
                return ExitSignal(
                    trade_id=trade_id,
                    exit_type=exit_type,
                    exit_price=current_stop,
                    reasoning=f"{exit_type} triggered at {current_stop:.2f}",
                )
        else:  # SHORT
            trailing_stop = lowest_since_entry + (atr * self.config.trailing_stop_atr)
            # Never let winner turn into loser
            risk = initial_stop - entry_price
            if entry_price - lowest_since_entry > risk:
                trailing_stop = min(trailing_stop, entry_price)
            current_stop = min(initial_stop, trailing_stop)

            # Check stop loss hit
            if current_high >= current_stop:
                exit_type = "TRAILING_STOP" if current_stop < initial_stop else "STOP_LOSS"
                logger.info(f"Trade {trade_id}: {exit_type} at {current_stop:.2f}")
                return ExitSignal(
                    trade_id=trade_id,
                    exit_type=exit_type,
                    exit_price=current_stop,
                    reasoning=f"{exit_type} triggered at {current_stop:.2f}",
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
        """Return current strategy parameters.

        Returns:
            Dictionary of parameter names to values
        """
        return {
            "lookback_period": self.config.lookback_period,
            "volume_multiplier": self.config.volume_multiplier,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "trailing_stop_atr": self.config.trailing_stop_atr,
            "max_hold_days": self.config.max_hold_days,
            "min_atr_percent": self.config.min_atr_percent,
            "close_range_threshold": self.config.close_range_threshold,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters.

        Args:
            params: Dictionary of parameter names to new values
        """
        if "lookback_period" in params:
            self.config.lookback_period = int(params["lookback_period"])
        if "volume_multiplier" in params:
            self.config.volume_multiplier = float(params["volume_multiplier"])
        if "atr_stop_multiplier" in params:
            self.config.atr_stop_multiplier = float(params["atr_stop_multiplier"])
        if "trailing_stop_atr" in params:
            self.config.trailing_stop_atr = float(params["trailing_stop_atr"])
        if "max_hold_days" in params:
            self.config.max_hold_days = int(params["max_hold_days"])
        if "min_atr_percent" in params:
            self.config.min_atr_percent = float(params["min_atr_percent"])
        if "close_range_threshold" in params:
            self.config.close_range_threshold = float(params["close_range_threshold"])


# Register strategy with global registry
strategy_registry.register(BreakoutStrategy)
