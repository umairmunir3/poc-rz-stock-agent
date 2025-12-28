"""Telegram Alert Bot - sends trading signals via Telegram."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Bot

    from src.risk.circuit_breakers import BreakerStatus
    from src.signals.realtime import ExitSignal
    from src.strategies.base import Signal

logger = logging.getLogger(__name__)


@dataclass
class TelegramConfig:
    """Configuration for Telegram bot.

    Attributes:
        token: Telegram bot token.
        chat_id: Chat ID to send messages to.
        max_messages_per_minute: Rate limit.
        retry_attempts: Number of retry attempts.
        retry_delay_seconds: Delay between retries.
    """

    token: str
    chat_id: str
    max_messages_per_minute: int = 20
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0


class TelegramAlertBot:
    """Telegram bot for sending trading alerts.

    Sends formatted trading signals, exit alerts, and risk notifications
    via Telegram with rate limiting and retry logic.

    Example:
        >>> bot = TelegramAlertBot(token="...", chat_id="...")
        >>> await bot.send_signal(my_signal)
    """

    def __init__(
        self,
        token: str,
        chat_id: str,
        config: TelegramConfig | None = None,
    ) -> None:
        """Initialize the Telegram bot.

        Args:
            token: Telegram bot token from BotFather.
            chat_id: Chat ID to send messages to.
            config: Optional full configuration.
        """
        if config:
            self.config = config
        else:
            self.config = TelegramConfig(token=token, chat_id=chat_id)

        self._bot: Bot | None = None
        self._message_times: deque[float] = deque(maxlen=self.config.max_messages_per_minute)
        self._priority_queue: list[tuple[int, str]] = []  # (priority, message)

    def _get_bot(self) -> Bot:
        """Get or create the Telegram bot instance."""
        if self._bot is None:
            from telegram import Bot

            self._bot = Bot(token=self.config.token)
        return self._bot

    # -------------------------------------------------------------------------
    # Message Formatting
    # -------------------------------------------------------------------------

    def format_signal(self, signal: Signal) -> str:
        """Format a trading signal for Telegram.

        Args:
            signal: The signal to format.

        Returns:
            Formatted message string.
        """
        direction_emoji = "🟢" if signal.direction == "LONG" else "🔴"
        stars = self._score_to_stars(signal.score)

        # Calculate percentages
        if signal.direction == "LONG":
            stop_pct = ((signal.stop_loss - signal.entry_price) / signal.entry_price) * 100
            target_pct = ((signal.take_profit - signal.entry_price) / signal.entry_price) * 100
        else:
            stop_pct = ((signal.entry_price - signal.stop_loss) / signal.entry_price) * 100
            target_pct = ((signal.entry_price - signal.take_profit) / signal.entry_price) * 100

        # Risk/Reward ratio
        risk = abs(signal.entry_price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        message = f"""🎯 NEW SIGNAL: {signal.symbol} — {signal.direction} {direction_emoji}

📈 Strategy: {signal.strategy}
⭐ Score: {signal.score}/100 {stars}

📍 Entry: ${signal.entry_price:.2f}
🛑 Stop: ${signal.stop_loss:.2f} ({stop_pct:+.1f}%)
🎯 Target: ${signal.take_profit:.2f} ({target_pct:+.1f}%)

📊 R:R: {rr_ratio:.1f}:1

📝 {signal.reasoning}"""

        return message

    def format_exit(self, exit_signal: ExitSignal) -> str:
        """Format an exit signal for Telegram.

        Args:
            exit_signal: The exit signal to format.

        Returns:
            Formatted message string.
        """
        reason_emoji = {
            "stop_loss": "🛑",
            "take_profit": "🎯",
            "signal_invalidated": "⚠️",
            "manual": "👤",
        }
        emoji = reason_emoji.get(exit_signal.exit_reason, "📤")

        pnl_emoji = "🟢" if exit_signal.pnl_percent >= 0 else "🔴"
        reason_text = exit_signal.exit_reason.replace("_", " ").title()

        message = f"""{emoji} EXIT SIGNAL: {exit_signal.symbol}

📊 Direction: {exit_signal.direction}
💰 Exit Price: ${exit_signal.exit_price:.2f}

{pnl_emoji} P&L: {exit_signal.pnl_percent:+.2f}%

📝 Reason: {reason_text}
🔗 Signal ID: {exit_signal.entry_signal_id}"""

        return message

    def format_daily_summary(
        self,
        signals: list[Signal],
        closed_trades: list[dict] | None = None,
        total_pnl: float = 0.0,
    ) -> str:
        """Format a daily trading summary.

        Args:
            signals: Signals generated today.
            closed_trades: Trades closed today.
            total_pnl: Total P&L for the day.

        Returns:
            Formatted message string.
        """
        if closed_trades is None:
            closed_trades = []

        today = datetime.now().strftime("%Y-%m-%d")
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"

        # Count by direction
        long_signals = sum(1 for s in signals if s.direction == "LONG")
        short_signals = sum(1 for s in signals if s.direction == "SHORT")

        # Top signals by score
        top_signals = sorted(signals, key=lambda s: s.score, reverse=True)[:5]
        top_signals_text = "\n".join(
            f"  • {s.symbol}: {s.direction} (Score: {s.score})" for s in top_signals
        )

        message = f"""📊 DAILY SUMMARY — {today}

📈 Signals Generated: {len(signals)}
   🟢 Long: {long_signals}
   🔴 Short: {short_signals}

💼 Trades Closed: {len(closed_trades)}
{pnl_emoji} Total P&L: {total_pnl:+.2f}%

🏆 Top Signals:
{top_signals_text}"""

        return message

    def format_risk_alert(self, status: BreakerStatus) -> str:
        """Format a circuit breaker risk alert.

        Args:
            status: The breaker status.

        Returns:
            Formatted message string.
        """
        status_emoji = {
            "OK": "🟢",
            "WARNING": "🟡",
            "TRIGGERED": "🔴",
        }
        emoji = status_emoji.get(status.status, "⚠️")

        can_trade_text = "Yes" if status.can_trade else "No"

        message = f"""⚠️ RISK ALERT: {status.breaker_name}

{emoji} Status: {status.status}

📊 Can Trade: {can_trade_text}
📏 Position Size: {status.position_size_multiplier:.0%}

📝 {status.message}
🚨 Action: {status.action_required}"""

        return message

    def _score_to_stars(self, score: int) -> str:
        """Convert a score to star rating.

        Args:
            score: Score from 0-100.

        Returns:
            Star emoji string.
        """
        if score >= 90:
            return "⭐⭐⭐⭐⭐"
        elif score >= 80:
            return "⭐⭐⭐⭐"
        elif score >= 70:
            return "⭐⭐⭐"
        elif score >= 60:
            return "⭐⭐"
        else:
            return "⭐"

    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------

    def _can_send_message(self) -> bool:
        """Check if we can send a message without exceeding rate limit.

        Returns:
            True if we can send.
        """
        current_time = time.time()

        # Remove messages older than 60 seconds
        while self._message_times and current_time - self._message_times[0] > 60:
            self._message_times.popleft()

        return len(self._message_times) < self.config.max_messages_per_minute

    def _record_message_sent(self) -> None:
        """Record that a message was sent."""
        self._message_times.append(time.time())

    async def _wait_for_rate_limit(self) -> None:
        """Wait until we can send a message."""
        while not self._can_send_message():
            await asyncio.sleep(0.5)

    # -------------------------------------------------------------------------
    # Sending Messages
    # -------------------------------------------------------------------------

    async def _send_message(
        self,
        text: str,
        _priority: int = 0,
        skip_rate_limit: bool = False,
    ) -> bool:
        """Send a message with retry logic.

        Args:
            text: Message text to send.
            priority: Higher priority = sent first (for risk alerts).
            skip_rate_limit: If True, skip rate limiting (for urgent alerts).

        Returns:
            True if message was sent successfully.
        """
        # Wait for rate limit unless skipping
        if not skip_rate_limit:
            await self._wait_for_rate_limit()

        bot = self._get_bot()

        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                await bot.send_message(
                    chat_id=self.config.chat_id,
                    text=text,
                    parse_mode="Markdown",
                )
                self._record_message_sent()
                logger.info(f"Telegram message sent successfully (attempt {attempt})")
                return True

            except Exception as e:
                logger.warning(f"Telegram send failed (attempt {attempt}): {e}")
                if attempt < self.config.retry_attempts:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        logger.error(f"Failed to send Telegram message after {self.config.retry_attempts} attempts")
        return False

    async def send_signal(self, signal: Signal) -> bool:
        """Send a trading signal alert.

        Args:
            signal: The signal to send.

        Returns:
            True if sent successfully.
        """
        message = self.format_signal(signal)
        return await self._send_message(message, _priority=1)

    async def send_exit(self, exit_signal: ExitSignal) -> bool:
        """Send an exit signal alert.

        Args:
            exit_signal: The exit signal to send.

        Returns:
            True if sent successfully.
        """
        message = self.format_exit(exit_signal)
        return await self._send_message(message, _priority=2)

    async def send_risk_alert(self, status: BreakerStatus) -> bool:
        """Send a risk/circuit breaker alert.

        Risk alerts have highest priority and skip rate limiting.

        Args:
            status: The breaker status.

        Returns:
            True if sent successfully.
        """
        message = self.format_risk_alert(status)
        # Risk alerts skip rate limiting - they're critical
        return await self._send_message(message, _priority=10, skip_rate_limit=True)

    async def send_daily_summary(
        self,
        signals: list[Signal],
        closed_trades: list[dict] | None = None,
        total_pnl: float = 0.0,
    ) -> bool:
        """Send a daily summary.

        Args:
            signals: Signals generated today.
            closed_trades: Trades closed today.
            total_pnl: Total P&L.

        Returns:
            True if sent successfully.
        """
        message = self.format_daily_summary(signals, closed_trades, total_pnl)
        return await self._send_message(message, _priority=0)

    async def send_custom_message(self, text: str) -> bool:
        """Send a custom message.

        Args:
            text: Message text.

        Returns:
            True if sent successfully.
        """
        return await self._send_message(text)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_rate_limit_status(self) -> dict[str, int]:
        """Get current rate limit status.

        Returns:
            Dictionary with messages_sent and remaining capacity.
        """
        current_time = time.time()

        # Clean old messages
        while self._message_times and current_time - self._message_times[0] > 60:
            self._message_times.popleft()

        messages_in_window = len(self._message_times)
        remaining = self.config.max_messages_per_minute - messages_in_window

        return {
            "messages_sent_last_minute": messages_in_window,
            "remaining_capacity": max(0, remaining),
        }
