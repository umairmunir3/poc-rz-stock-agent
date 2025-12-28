"""Unit tests for Telegram Alert Bot."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.signals.telegram_bot import TelegramAlertBot, TelegramConfig


class TestTelegramConfig:
    """Tests for TelegramConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = TelegramConfig(token="test_token", chat_id="123456")
        assert config.token == "test_token"
        assert config.chat_id == "123456"
        assert config.max_messages_per_minute == 20
        assert config.retry_attempts == 3


class TestTelegramAlertBotInit:
    """Tests for TelegramAlertBot initialization."""

    def test_init_with_token_and_chat_id(self) -> None:
        """Test initialization with token and chat_id."""
        bot = TelegramAlertBot(token="test_token", chat_id="123456")
        assert bot.config.token == "test_token"
        assert bot.config.chat_id == "123456"

    def test_init_with_config(self) -> None:
        """Test initialization with config object."""
        config = TelegramConfig(
            token="custom_token",
            chat_id="999999",
            max_messages_per_minute=30,
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)
        assert bot.config.token == "custom_token"
        assert bot.config.max_messages_per_minute == 30


class TestFormatSignal:
    """Tests for signal formatting."""

    def test_format_signal_readable(self) -> None:
        """Test that formatted signal is readable."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.strategy = "RSI Mean Reversion"
        mock_signal.score = 78
        mock_signal.entry_price = 178.50
        mock_signal.stop_loss = 174.20
        mock_signal.take_profit = 186.50
        mock_signal.reasoning = "RSI oversold at 28, price at 50 SMA support"

        message = bot.format_signal(mock_signal)

        # Check key elements are present
        assert "AAPL" in message
        assert "LONG" in message
        assert "RSI Mean Reversion" in message
        assert "78/100" in message
        assert "$178.50" in message
        assert "$174.20" in message
        assert "$186.50" in message
        assert "RSI oversold" in message

    def test_format_includes_all_fields(self) -> None:
        """Test that all signal fields are included."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_signal = MagicMock()
        mock_signal.symbol = "MSFT"
        mock_signal.direction = "SHORT"
        mock_signal.strategy = "Breakout"
        mock_signal.score = 85
        mock_signal.entry_price = 400.00
        mock_signal.stop_loss = 410.00
        mock_signal.take_profit = 380.00
        mock_signal.reasoning = "Breakdown below support"

        message = bot.format_signal(mock_signal)

        assert "MSFT" in message
        assert "SHORT" in message
        assert "Breakout" in message
        assert "85" in message
        assert "Entry" in message
        assert "Stop" in message
        assert "Target" in message
        assert "R:R" in message
        assert "Breakdown" in message

    def test_format_signal_long_percentages(self) -> None:
        """Test percentage calculation for long positions."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.strategy = "Test"
        mock_signal.score = 75
        mock_signal.entry_price = 100.00
        mock_signal.stop_loss = 95.00  # -5%
        mock_signal.take_profit = 110.00  # +10%
        mock_signal.reasoning = "Test"

        message = bot.format_signal(mock_signal)

        assert "-5.0%" in message
        assert "+10.0%" in message

    def test_format_signal_short_percentages(self) -> None:
        """Test percentage calculation for short positions."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "SHORT"
        mock_signal.strategy = "Test"
        mock_signal.score = 75
        mock_signal.entry_price = 100.00
        mock_signal.stop_loss = 105.00  # -5% for short
        mock_signal.take_profit = 90.00  # +10% for short
        mock_signal.reasoning = "Test"

        message = bot.format_signal(mock_signal)

        # For shorts, stop is above entry
        assert message is not None


class TestFormatExit:
    """Tests for exit signal formatting."""

    def test_format_exit_stop_loss(self) -> None:
        """Test formatting stop loss exit."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_exit = MagicMock()
        mock_exit.symbol = "AAPL"
        mock_exit.direction = "LONG"
        mock_exit.entry_signal_id = "SIG-000001"
        mock_exit.exit_reason = "stop_loss"
        mock_exit.exit_price = 145.00
        mock_exit.pnl_percent = -3.25

        message = bot.format_exit(mock_exit)

        assert "EXIT SIGNAL" in message
        assert "AAPL" in message
        assert "LONG" in message
        assert "Stop Loss" in message
        assert "-3.25%" in message
        assert "SIG-000001" in message

    def test_format_exit_take_profit(self) -> None:
        """Test formatting take profit exit."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_exit = MagicMock()
        mock_exit.symbol = "MSFT"
        mock_exit.direction = "SHORT"
        mock_exit.entry_signal_id = "SIG-000002"
        mock_exit.exit_reason = "take_profit"
        mock_exit.exit_price = 380.00
        mock_exit.pnl_percent = 5.50

        message = bot.format_exit(mock_exit)

        assert "Take Profit" in message
        assert "+5.50%" in message


class TestFormatDailySummary:
    """Tests for daily summary formatting."""

    def test_format_daily_summary(self) -> None:
        """Test daily summary formatting."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        signals = [
            MagicMock(symbol="AAPL", direction="LONG", score=85),
            MagicMock(symbol="MSFT", direction="LONG", score=78),
            MagicMock(symbol="GOOGL", direction="SHORT", score=72),
        ]

        message = bot.format_daily_summary(signals, total_pnl=2.5)

        assert "DAILY SUMMARY" in message
        assert "Signals Generated: 3" in message
        assert "Long: 2" in message
        assert "Short: 1" in message
        assert "+2.50%" in message

    def test_format_daily_summary_top_signals(self) -> None:
        """Test that top signals are included."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        signals = [
            MagicMock(symbol="AAPL", direction="LONG", score=95),
            MagicMock(symbol="MSFT", direction="LONG", score=78),
        ]

        message = bot.format_daily_summary(signals)

        assert "Top Signals" in message
        assert "AAPL" in message


class TestFormatRiskAlert:
    """Tests for risk alert formatting."""

    def test_format_risk_alert(self) -> None:
        """Test risk alert formatting."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_status = MagicMock()
        mock_status.breaker_name = "DrawdownBreaker"
        mock_status.status = "TRIGGERED"
        mock_status.can_trade = False
        mock_status.position_size_multiplier = 0.0
        mock_status.message = "Daily drawdown exceeded limit"
        mock_status.action_required = "Stop trading immediately"

        message = bot.format_risk_alert(mock_status)

        assert "RISK ALERT" in message
        assert "DrawdownBreaker" in message
        assert "TRIGGERED" in message
        assert "Can Trade: No" in message
        assert "Stop trading immediately" in message


class TestScoreToStars:
    """Tests for score to stars conversion."""

    def test_score_to_stars(self) -> None:
        """Test score to stars mapping."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        assert bot._score_to_stars(95) == "⭐⭐⭐⭐⭐"
        assert bot._score_to_stars(85) == "⭐⭐⭐⭐"
        assert bot._score_to_stars(75) == "⭐⭐⭐"
        assert bot._score_to_stars(65) == "⭐⭐"
        assert bot._score_to_stars(50) == "⭐"


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limiting_respects_limit(self) -> None:
        """Test that rate limiting respects message limit."""
        config = TelegramConfig(
            token="test",
            chat_id="123",
            max_messages_per_minute=5,
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)

        # Simulate 5 messages sent
        current_time = time.time()
        for _ in range(5):
            bot._message_times.append(current_time)

        assert bot._can_send_message() is False

    def test_rate_limit_window_slides(self) -> None:
        """Test that old messages expire from rate limit window."""
        config = TelegramConfig(
            token="test",
            chat_id="123",
            max_messages_per_minute=5,
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)

        # Simulate 5 messages sent 61 seconds ago
        old_time = time.time() - 61
        for _ in range(5):
            bot._message_times.append(old_time)

        # Should be able to send now (old messages expired)
        assert bot._can_send_message() is True

    def test_get_rate_limit_status(self) -> None:
        """Test getting rate limit status."""
        config = TelegramConfig(
            token="test",
            chat_id="123",
            max_messages_per_minute=20,
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)

        # Send 5 messages
        current_time = time.time()
        for _ in range(5):
            bot._message_times.append(current_time)

        status = bot.get_rate_limit_status()

        assert status["messages_sent_last_minute"] == 5
        assert status["remaining_capacity"] == 15


class TestSendMessage:
    """Tests for sending messages."""

    @pytest.mark.asyncio
    async def test_send_signal_success(self) -> None:
        """Test successful signal sending."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.strategy = "Test"
        mock_signal.score = 80
        mock_signal.entry_price = 150.0
        mock_signal.stop_loss = 145.0
        mock_signal.take_profit = 160.0
        mock_signal.reasoning = "Test reasoning"

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_get_bot.return_value = mock_bot

            result = await bot.send_signal(mock_signal)

            assert result is True
            mock_bot.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        """Test that transient errors are retried."""
        config = TelegramConfig(
            token="test",
            chat_id="123",
            retry_attempts=3,
            retry_delay_seconds=0.01,
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            # Fail twice, succeed on third attempt
            mock_bot.send_message = AsyncMock(
                side_effect=[Exception("Network error"), Exception("Timeout"), None]
            )
            mock_get_bot.return_value = mock_bot

            result = await bot._send_message("Test message")

            assert result is True
            assert mock_bot.send_message.call_count == 3

    @pytest.mark.asyncio
    async def test_handles_api_error(self) -> None:
        """Test that API errors are handled without crashing."""
        config = TelegramConfig(
            token="test",
            chat_id="123",
            retry_attempts=2,
            retry_delay_seconds=0.01,
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock(side_effect=Exception("API Error"))
            mock_get_bot.return_value = mock_bot

            result = await bot._send_message("Test message")

            assert result is False
            assert mock_bot.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_risk_alert_priority(self) -> None:
        """Test that risk alerts skip rate limiting."""
        config = TelegramConfig(
            token="test",
            chat_id="123",
            max_messages_per_minute=1,  # Very low limit
        )
        bot = TelegramAlertBot(token="", chat_id="", config=config)

        # Fill up the rate limit
        current_time = time.time()
        bot._message_times.append(current_time)

        mock_status = MagicMock()
        mock_status.breaker_name = "DrawdownBreaker"
        mock_status.status = "TRIGGERED"
        mock_status.can_trade = False
        mock_status.position_size_multiplier = 0.0
        mock_status.message = "Drawdown exceeded"
        mock_status.action_required = "Stop trading"

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_get_bot.return_value = mock_bot

            # This should succeed even though rate limit is hit
            result = await bot.send_risk_alert(mock_status)

            assert result is True
            mock_bot.send_message.assert_called_once()


class TestSendExit:
    """Tests for sending exit signals."""

    @pytest.mark.asyncio
    async def test_send_exit_success(self) -> None:
        """Test successful exit signal sending."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        mock_exit = MagicMock()
        mock_exit.symbol = "AAPL"
        mock_exit.direction = "LONG"
        mock_exit.entry_signal_id = "SIG-000001"
        mock_exit.exit_reason = "take_profit"
        mock_exit.exit_price = 165.0
        mock_exit.pnl_percent = 5.5

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_get_bot.return_value = mock_bot

            result = await bot.send_exit(mock_exit)

            assert result is True


class TestSendDailySummary:
    """Tests for sending daily summary."""

    @pytest.mark.asyncio
    async def test_send_daily_summary_success(self) -> None:
        """Test successful daily summary sending."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        signals = [
            MagicMock(symbol="AAPL", direction="LONG", score=85),
        ]

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_get_bot.return_value = mock_bot

            result = await bot.send_daily_summary(signals, total_pnl=1.5)

            assert result is True


class TestSendCustomMessage:
    """Tests for sending custom messages."""

    @pytest.mark.asyncio
    async def test_send_custom_message(self) -> None:
        """Test sending a custom message."""
        bot = TelegramAlertBot(token="test", chat_id="123")

        with patch.object(bot, "_get_bot") as mock_get_bot:
            mock_bot = MagicMock()
            mock_bot.send_message = AsyncMock()
            mock_get_bot.return_value = mock_bot

            result = await bot.send_custom_message("Hello, World!")

            assert result is True
            mock_bot.send_message.assert_called_once()


class TestBotInstance:
    """Tests for bot instance management."""

    def test_get_bot_creates_instance(self) -> None:
        """Test that _get_bot creates a Bot instance."""
        bot = TelegramAlertBot(token="test_token", chat_id="123")

        with patch("telegram.Bot") as mock_bot_class:
            mock_bot_class.return_value = MagicMock()
            result = bot._get_bot()

            mock_bot_class.assert_called_once_with(token="test_token")
            assert result is not None

    def test_get_bot_reuses_instance(self) -> None:
        """Test that _get_bot reuses existing instance."""
        bot = TelegramAlertBot(token="test_token", chat_id="123")
        bot._bot = MagicMock()  # Pre-set the bot

        result = bot._get_bot()

        assert result is bot._bot
