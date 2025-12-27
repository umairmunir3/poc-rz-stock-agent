"""EMA Crossover Strategy - trades EMA 9/21 crossovers in trend."""

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
class EMACrossoverConfig:
    """Configuration for EMA Crossover Strategy.

    Attributes:
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        trend_ema: Trend filter EMA period
        atr_stop_multiplier: ATR multiplier for stop loss
        profit_target_percent: Profit target as percentage
        volume_threshold: Minimum relative volume
        max_hold_days: Maximum days to hold position
    """

    fast_ema: int = 9
    slow_ema: int = 21
    trend_ema: int = 50
    atr_stop_multiplier: float = 1.5
    profit_target_percent: float = 8.0
    volume_threshold: float = 0.8
    max_hold_days: int = 10


class EMACrossoverStrategy(Strategy):
    """EMA Crossover Strategy - trades EMA 9/21 crossovers.

    This strategy identifies trend-following opportunities when the fast EMA
    crosses above/below the slow EMA, filtered by the longer-term trend.

    Entry Conditions for LONG:
    - EMA 9 crosses above EMA 21
    - Price above EMA 50 (trend filter)
    - MACD histogram positive
    - Volume > 0.8x average

    Entry Conditions for SHORT:
    - EMA 9 crosses below EMA 21
    - Price below EMA 50 (trend filter)
    - MACD histogram negative
    - Volume > 0.8x average

    Exit Conditions:
    - EMA 9 crosses in opposite direction
    - Profit target reached (8%)
    - Stop loss hit (1.5 ATR)
    - Time exit after max hold days

    Historical stats: 55-58% win rate, 1.4:1 R:R
    """

    name = "EMACrossover"
    description = "Trades EMA 9/21 crossovers in trending markets"

    def __init__(self, config: EMACrossoverConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration. Uses defaults if not provided.
        """
        self.config = config or EMACrossoverConfig()

    def scan(self, df: pd.DataFrame) -> Signal | None:
        """Scan for EMA crossover entry signal.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Signal if crossover conditions are met, None otherwise
        """
        if not self.validate_dataframe(df):
            logger.warning("Invalid DataFrame - missing required columns")
            return None

        min_rows = max(self.config.trend_ema, 35) + 5  # Need extra for MACD
        if len(df) < min_rows:
            logger.debug(f"Insufficient data for EMA crossover (need {min_rows} rows)")
            return None

        try:
            indicators = TechnicalIndicators(df)
        except Exception as e:
            logger.warning(f"Failed to initialize indicators: {e}")
            return None

        # Get current and previous values
        current_close = float(df["close"].iloc[-1])
        current_volume = float(df["volume"].iloc[-1])

        # Calculate EMAs
        ema_fast = indicators.calculate_ema(self.config.fast_ema)
        ema_slow = indicators.calculate_ema(self.config.slow_ema)
        ema_trend = indicators.calculate_ema(self.config.trend_ema)

        current_fast = float(ema_fast.iloc[-1])
        current_slow = float(ema_slow.iloc[-1])
        current_trend = float(ema_trend.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])

        # Calculate MACD
        _, _, macd_hist = indicators.calculate_macd()
        current_hist = float(macd_hist.iloc[-1])

        # Calculate relative volume
        volume_sma = indicators.calculate_volume_sma(20)
        avg_volume = float(volume_sma.iloc[-1])
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 0

        # Check for bullish crossover (LONG)
        bullish_crossover = prev_fast <= prev_slow and current_fast > current_slow
        bearish_crossover = prev_fast >= prev_slow and current_fast < current_slow

        # Determine direction
        direction: Literal["LONG", "SHORT"] | None = None
        if bullish_crossover:
            # Additional LONG conditions
            if current_close > current_trend and current_hist > 0:
                if relative_volume >= self.config.volume_threshold:
                    direction = "LONG"
        elif bearish_crossover:
            # Additional SHORT conditions
            if current_close < current_trend and current_hist < 0:
                if relative_volume >= self.config.volume_threshold:
                    direction = "SHORT"

        if direction is None:
            return None

        # Calculate ATR for stop loss
        atr = indicators.calculate_atr(14)
        current_atr = float(atr.iloc[-1])

        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(
            entry_price=current_close,
            atr=current_atr,
            direction=direction,
            multiplier=self.config.atr_stop_multiplier,
        )

        # Take profit based on percentage target
        if direction == "LONG":
            take_profit = current_close * (1 + self.config.profit_target_percent / 100)
        else:
            take_profit = current_close * (1 - self.config.profit_target_percent / 100)

        # Calculate score
        score = self._calculate_score(
            _df=df,
            indicators=indicators,
            direction=direction,
            relative_volume=relative_volume,
            current_hist=current_hist,
        )

        # Build reasoning
        reasons = []
        if direction == "LONG":
            reasons.append(f"EMA {self.config.fast_ema} crossed above EMA {self.config.slow_ema}")
            reasons.append(f"Price above EMA {self.config.trend_ema}")
        else:
            reasons.append(f"EMA {self.config.fast_ema} crossed below EMA {self.config.slow_ema}")
            reasons.append(f"Price below EMA {self.config.trend_ema}")
        reasons.append(f"MACD hist={current_hist:.3f}")
        reasons.append(f"Volume={relative_volume:.1f}x avg")

        symbol = df.attrs.get("symbol", "UNKNOWN")
        reasoning = "; ".join(reasons)

        logger.info(
            f"EMA Crossover signal for {symbol}: {direction}, {reasoning}, "
            f"Score={score}, Entry={current_close:.2f}"
        )

        return Signal(
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            entry_price=current_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            score=score,
            reasoning=reasoning,
            metadata={
                "ema_fast": current_fast,
                "ema_slow": current_slow,
                "ema_trend": current_trend,
                "macd_hist": current_hist,
                "atr": current_atr,
                "relative_volume": relative_volume,
            },
        )

    def _calculate_score(
        self,
        _df: pd.DataFrame,
        indicators: TechnicalIndicators,
        direction: str,
        relative_volume: float,
        current_hist: float,
    ) -> int:
        """Calculate signal strength score (0-100).

        Args:
            df: OHLCV DataFrame
            indicators: TechnicalIndicators instance
            direction: Trade direction
            relative_volume: Volume relative to average
            current_hist: Current MACD histogram value

        Returns:
            Score from 0-100
        """
        score = 50  # Base score

        # +15 if volume > 1.5x average
        if relative_volume > 1.5:
            score += 15

        # +10 if MACD histogram is strongly positive/negative
        if (direction == "LONG" and current_hist > 0.5) or (
            direction == "SHORT" and current_hist < -0.5
        ):
            score += 10

        # +10 if ADX > 25 (strong trend)
        try:
            adx = indicators.calculate_adx(14)
            if float(adx.iloc[-1]) > 25:
                score += 10
        except Exception:
            pass

        # +10 if RSI is in favorable zone
        try:
            rsi = indicators.calculate_rsi(14)
            current_rsi = float(rsi.iloc[-1])
            if (direction == "LONG" and 40 < current_rsi < 70) or (
                direction == "SHORT" and 30 < current_rsi < 60
            ):
                score += 10
        except Exception:
            pass

        # +5 if all EMAs aligned
        try:
            ema_9 = float(indicators.calculate_ema(9).iloc[-1])
            ema_21 = float(indicators.calculate_ema(21).iloc[-1])
            ema_50 = float(indicators.calculate_ema(50).iloc[-1])
            if (direction == "LONG" and ema_9 > ema_21 > ema_50) or (
                direction == "SHORT" and ema_9 < ema_21 < ema_50
            ):
                score += 5
        except Exception:
            pass

        return min(score, 100)

    def check_exit(self, df: pd.DataFrame, trade: Any) -> ExitSignal | None:
        """Check if an open trade should be exited.

        Exit Conditions:
        - EMA 9 crosses in opposite direction
        - Profit target reached
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
        direction = getattr(trade, "direction", "LONG")
        entry_date = getattr(trade, "entry_date", None)

        # Check stop loss
        if (direction == "LONG" and current_low <= stop_loss) or (
            direction == "SHORT" and current_high >= stop_loss
        ):
            logger.info(f"Trade {trade_id}: Stop loss triggered at {stop_loss:.2f}")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="STOP_LOSS",
                exit_price=stop_loss,
                reasoning=f"Stop loss hit at {stop_loss:.2f}",
            )

        # Check take profit
        if (direction == "LONG" and current_high >= take_profit) or (
            direction == "SHORT" and current_low <= take_profit
        ):
            logger.info(f"Trade {trade_id}: Take profit triggered at {take_profit:.2f}")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="TAKE_PROFIT",
                exit_price=take_profit,
                reasoning=f"Profit target hit at {take_profit:.2f}",
            )

        # Check for EMA crossover exit
        try:
            indicators = TechnicalIndicators(df)
            ema_fast = indicators.calculate_ema(self.config.fast_ema)
            ema_slow = indicators.calculate_ema(self.config.slow_ema)

            current_fast = float(ema_fast.iloc[-1])
            current_slow = float(ema_slow.iloc[-1])
            prev_fast = float(ema_fast.iloc[-2])
            prev_slow = float(ema_slow.iloc[-2])

            # Check for opposite crossover
            if direction == "LONG":
                if prev_fast >= prev_slow and current_fast < current_slow:
                    logger.info(f"Trade {trade_id}: Bearish crossover exit")
                    return ExitSignal(
                        trade_id=trade_id,
                        exit_type="STRATEGY_EXIT",
                        exit_price=current_close,
                        reasoning="EMA bearish crossover",
                    )
            else:  # SHORT
                if prev_fast <= prev_slow and current_fast > current_slow:
                    logger.info(f"Trade {trade_id}: Bullish crossover exit")
                    return ExitSignal(
                        trade_id=trade_id,
                        exit_type="STRATEGY_EXIT",
                        exit_price=current_close,
                        reasoning="EMA bullish crossover",
                    )
        except Exception as e:
            logger.warning(f"Error calculating EMAs for exit: {e}")

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
            "fast_ema": self.config.fast_ema,
            "slow_ema": self.config.slow_ema,
            "trend_ema": self.config.trend_ema,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "profit_target_percent": self.config.profit_target_percent,
            "volume_threshold": self.config.volume_threshold,
            "max_hold_days": self.config.max_hold_days,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters."""
        if "fast_ema" in params:
            self.config.fast_ema = int(params["fast_ema"])
        if "slow_ema" in params:
            self.config.slow_ema = int(params["slow_ema"])
        if "trend_ema" in params:
            self.config.trend_ema = int(params["trend_ema"])
        if "atr_stop_multiplier" in params:
            self.config.atr_stop_multiplier = float(params["atr_stop_multiplier"])
        if "profit_target_percent" in params:
            self.config.profit_target_percent = float(params["profit_target_percent"])
        if "volume_threshold" in params:
            self.config.volume_threshold = float(params["volume_threshold"])
        if "max_hold_days" in params:
            self.config.max_hold_days = int(params["max_hold_days"])


# Register strategy with global registry
strategy_registry.register(EMACrossoverStrategy)
