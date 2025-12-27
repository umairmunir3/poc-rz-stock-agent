"""RSI Mean Reversion Strategy - trades oversold bounces in uptrends."""

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
class RSIStrategyConfig:
    """Configuration for RSI Mean Reversion Strategy.

    Attributes:
        rsi_period: RSI calculation period
        oversold_threshold: RSI level considered oversold (entry trigger)
        overbought_threshold: RSI level considered overbought (take profit)
        exit_rsi: RSI level to exit at (strategy exit)
        require_uptrend: Whether to require price > SMA(200)
        atr_stop_multiplier: ATR multiplier for stop loss distance
        min_rr_ratio: Minimum risk-reward ratio for take profit
        volume_threshold: Minimum relative volume (0.8 = 80% of average)
        max_hold_days: Maximum days to hold position
    """

    rsi_period: int = 14
    oversold_threshold: int = 30
    overbought_threshold: int = 70
    exit_rsi: int = 50
    require_uptrend: bool = True
    atr_stop_multiplier: float = 1.5
    min_rr_ratio: float = 1.5
    volume_threshold: float = 0.8
    max_hold_days: int = 5


class RSIMeanReversionStrategy(Strategy):
    """RSI Mean Reversion Strategy - trades oversold bounces.

    This strategy looks for stocks that are temporarily oversold (RSI < 30)
    within an overall uptrend (price > 200 SMA) and bets on mean reversion.

    Entry Conditions (ALL must be true for LONG):
    - RSI(14) < 30 (oversold)
    - Price > SMA(200) (in uptrend) if require_uptrend=True
    - Today's close > today's open (bullish candle preferred)
    - Volume > 0.8x 20-day average (not dead)

    Exit Conditions (ANY triggers exit):
    - RSI crosses above exit_rsi (50) - strategy exit
    - RSI > 70 (overbought) - take profit
    - Price hits stop loss
    - Price hits take profit
    - Time exit: 5 days max hold
    """

    name = "RSIMeanReversion"
    description = "Trades oversold bounces in uptrending stocks"

    def __init__(self, config: RSIStrategyConfig | None = None) -> None:
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration. Uses defaults if not provided.
        """
        self.config = config or RSIStrategyConfig()

    def scan(self, df: pd.DataFrame) -> Signal | None:
        """Scan for oversold bounce entry signal.

        Args:
            df: DataFrame with OHLCV data (needs 200+ rows for SMA)

        Returns:
            Signal if entry conditions are met, None otherwise
        """
        if not self.validate_dataframe(df):
            logger.warning("Invalid DataFrame - missing required columns")
            return None

        # Need enough data for indicators
        if len(df) < 200:
            logger.debug("Insufficient data for RSI strategy (need 200 rows)")
            return None

        try:
            indicators = TechnicalIndicators(df)
        except Exception as e:
            logger.warning(f"Failed to initialize indicators: {e}")
            return None

        # Get current values (last row)
        current_idx = -1
        current_close = float(df["close"].iloc[current_idx])
        current_open = float(df["open"].iloc[current_idx])
        current_volume = float(df["volume"].iloc[current_idx])

        # Calculate indicators
        rsi = indicators.calculate_rsi(self.config.rsi_period)
        current_rsi = float(rsi.iloc[current_idx])

        sma_200 = indicators.calculate_sma(200)
        current_sma_200 = float(sma_200.iloc[current_idx])

        volume_sma = indicators.calculate_volume_sma(20)
        avg_volume = float(volume_sma.iloc[current_idx])

        # Check entry conditions
        reasons: list[str] = []

        # Condition 1: RSI oversold
        if current_rsi >= self.config.oversold_threshold:
            logger.debug(
                f"RSI {current_rsi:.1f} not oversold (need < {self.config.oversold_threshold})"
            )
            return None
        reasons.append(f"RSI={current_rsi:.1f} (oversold)")

        # Condition 2: Uptrend check
        if self.config.require_uptrend and current_close <= current_sma_200:
            logger.debug(
                f"Price {current_close:.2f} below SMA200 {current_sma_200:.2f} - not in uptrend"
            )
            return None
        if self.config.require_uptrend:
            reasons.append(f"Price > SMA200 ({current_sma_200:.2f})")

        # Condition 3: Bullish candle (close > open)
        is_bullish_candle = current_close > current_open
        if not is_bullish_candle:
            logger.debug("Not a bullish candle (close <= open)")
            return None
        reasons.append("Bullish candle")

        # Condition 4: Volume check
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

        if relative_volume < self.config.volume_threshold:
            logger.debug(
                f"Volume too low: {relative_volume:.2f}x average (need {self.config.volume_threshold}x)"
            )
            return None
        reasons.append(f"Volume={relative_volume:.2f}x avg")

        # All conditions met - calculate stop loss and take profit
        atr = indicators.calculate_atr(14)
        current_atr = float(atr.iloc[current_idx])

        stop_loss = self.calculate_stop_loss(
            entry_price=current_close,
            atr=current_atr,
            direction="LONG",
            multiplier=self.config.atr_stop_multiplier,
        )

        take_profit = self.calculate_take_profit(
            entry_price=current_close,
            stop_loss=stop_loss,
            direction="LONG",
            risk_reward=self.config.min_rr_ratio * 1.5,
        )

        # Calculate score
        score = self._calculate_score(
            df=df,
            indicators=indicators,
            current_rsi=current_rsi,
            relative_volume=relative_volume,
            current_close=current_close,
        )

        # Get symbol from DataFrame if available
        symbol = df.attrs.get("symbol", "UNKNOWN")

        reasoning = "; ".join(reasons)
        logger.info(
            f"RSI Mean Reversion signal for {symbol}: {reasoning}, "
            f"Score={score}, Entry={current_close:.2f}, SL={stop_loss:.2f}, TP={take_profit:.2f}"
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
                "rsi": current_rsi,
                "sma_200": current_sma_200,
                "atr": current_atr,
                "relative_volume": relative_volume,
            },
        )

    def _calculate_score(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators,
        current_rsi: float,
        relative_volume: float,
        current_close: float,
    ) -> int:
        """Calculate signal strength score (0-100).

        Args:
            df: OHLCV DataFrame
            indicators: TechnicalIndicators instance
            current_rsi: Current RSI value
            relative_volume: Current relative volume
            current_close: Current closing price

        Returns:
            Score from 0-100
        """
        score = 50  # Base score if all conditions met

        # +10 if RSI < 25 (deeply oversold)
        if current_rsi < 25:
            score += 10

        # +10 if volume > 1.2x average
        if relative_volume > 1.2:
            score += 10

        # +10 if price at support level
        try:
            sr_detector = SupportResistanceDetector(df)
            levels = sr_detector.detect_levels()
            if levels.nearest_support is not None:
                support_distance = abs(current_close - levels.nearest_support.price)
                atr = float(indicators.calculate_atr(14).iloc[-1])
                if support_distance < atr * 0.5:  # Within half ATR of support
                    score += 10
        except Exception:
            pass  # Skip support check if it fails

        # +10 if EMA 9 > EMA 21 (short-term bullish)
        try:
            ema_9 = float(indicators.calculate_ema(9).iloc[-1])
            ema_21 = float(indicators.calculate_ema(21).iloc[-1])
            if ema_9 > ema_21:
                score += 10
        except Exception:
            pass

        # +10 if MACD histogram turning positive
        try:
            _, _, hist = indicators.calculate_macd()
            current_hist = float(hist.iloc[-1])
            prev_hist = float(hist.iloc[-2])
            if current_hist > prev_hist and current_hist > 0:
                score += 10
        except Exception:
            pass

        return min(score, 100)

    def check_exit(self, df: pd.DataFrame, trade: Any) -> ExitSignal | None:
        """Check if an open trade should be exited.

        Exit Conditions (ANY triggers exit):
        - RSI crosses above exit_rsi (50) - strategy exit
        - RSI > 70 (overbought) - take profit
        - Price hits stop loss
        - Price hits take profit
        - Time exit: max_hold_days exceeded

        Args:
            df: DataFrame with current OHLCV data
            trade: Trade object with entry_price, stop_loss, take_profit,
                   trade_id, and entry_date attributes

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

        # Check stop loss
        if stop_loss > 0 and current_low <= stop_loss:
            logger.info(f"Trade {trade_id}: Stop loss triggered at {stop_loss:.2f}")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="STOP_LOSS",
                exit_price=stop_loss,
                reasoning=f"Price hit stop loss at {stop_loss:.2f}",
            )

        # Check take profit
        if take_profit > 0 and current_high >= take_profit:
            logger.info(f"Trade {trade_id}: Take profit triggered at {take_profit:.2f}")
            return ExitSignal(
                trade_id=trade_id,
                exit_type="TAKE_PROFIT",
                exit_price=take_profit,
                reasoning=f"Price hit take profit at {take_profit:.2f}",
            )

        # Calculate RSI for strategy exits
        try:
            indicators = TechnicalIndicators(df)
            rsi = indicators.calculate_rsi(self.config.rsi_period)
            current_rsi = float(rsi.iloc[-1])

            # Check overbought exit
            if current_rsi >= self.config.overbought_threshold:
                logger.info(
                    f"Trade {trade_id}: RSI overbought exit at {current_rsi:.1f}"
                )
                return ExitSignal(
                    trade_id=trade_id,
                    exit_type="TAKE_PROFIT",
                    exit_price=current_close,
                    reasoning=f"RSI overbought at {current_rsi:.1f}",
                )

            # Check strategy exit (RSI crosses above exit level)
            if current_rsi >= self.config.exit_rsi:
                prev_rsi = float(rsi.iloc[-2])
                if prev_rsi < self.config.exit_rsi:
                    logger.info(
                        f"Trade {trade_id}: RSI crossed above {self.config.exit_rsi}"
                    )
                    return ExitSignal(
                        trade_id=trade_id,
                        exit_type="STRATEGY_EXIT",
                        exit_price=current_close,
                        reasoning=f"RSI crossed above {self.config.exit_rsi} (now {current_rsi:.1f})",
                    )
        except Exception as e:
            logger.warning(f"Error calculating RSI for exit check: {e}")

        # Check time exit
        if entry_date is not None:
            if isinstance(entry_date, datetime):
                days_held = (datetime.now() - entry_date).days
            else:
                days_held = (datetime.now().date() - entry_date).days

            if days_held >= self.config.max_hold_days:
                logger.info(
                    f"Trade {trade_id}: Time exit after {days_held} days"
                )
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
            "rsi_period": self.config.rsi_period,
            "oversold_threshold": self.config.oversold_threshold,
            "overbought_threshold": self.config.overbought_threshold,
            "exit_rsi": self.config.exit_rsi,
            "require_uptrend": self.config.require_uptrend,
            "atr_stop_multiplier": self.config.atr_stop_multiplier,
            "min_rr_ratio": self.config.min_rr_ratio,
            "volume_threshold": self.config.volume_threshold,
            "max_hold_days": self.config.max_hold_days,
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Update strategy parameters.

        Args:
            params: Dictionary of parameter names to new values
        """
        if "rsi_period" in params:
            self.config.rsi_period = int(params["rsi_period"])
        if "oversold_threshold" in params:
            self.config.oversold_threshold = int(params["oversold_threshold"])
        if "overbought_threshold" in params:
            self.config.overbought_threshold = int(params["overbought_threshold"])
        if "exit_rsi" in params:
            self.config.exit_rsi = int(params["exit_rsi"])
        if "require_uptrend" in params:
            self.config.require_uptrend = bool(params["require_uptrend"])
        if "atr_stop_multiplier" in params:
            self.config.atr_stop_multiplier = float(params["atr_stop_multiplier"])
        if "min_rr_ratio" in params:
            self.config.min_rr_ratio = float(params["min_rr_ratio"])
        if "volume_threshold" in params:
            self.config.volume_threshold = float(params["volume_threshold"])
        if "max_hold_days" in params:
            self.config.max_hold_days = int(params["max_hold_days"])


# Register strategy with global registry
strategy_registry.register(RSIMeanReversionStrategy)
