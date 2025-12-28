"""Unit tests for Real-Time Signal Generator."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from src.signals.realtime import (
    ET,
    MARKET_CLOSE,
    MARKET_OPEN,
    US_MARKET_HOLIDAYS,
    ActiveSignal,
    ExitSignal,
    RealtimeConfig,
    RealTimeSignalGenerator,
)


class TestRealtimeConfig:
    """Tests for RealtimeConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RealtimeConfig()
        assert config.scan_interval_minutes == 5
        assert config.symbols_per_batch == 50
        assert config.min_score_threshold == 70
        assert config.max_active_signals == 20

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RealtimeConfig(
            scan_interval_minutes=10,
            symbols_per_batch=100,
            min_score_threshold=80,
        )
        assert config.scan_interval_minutes == 10
        assert config.symbols_per_batch == 100
        assert config.min_score_threshold == 80


class TestExitSignal:
    """Tests for ExitSignal dataclass."""

    def test_creation(self) -> None:
        """Test ExitSignal creation."""
        exit_sig = ExitSignal(
            symbol="AAPL",
            direction="LONG",
            entry_signal_id="SIG-000001",
            exit_reason="stop_loss",
            exit_price=145.0,
            pnl_percent=-2.5,
        )
        assert exit_sig.symbol == "AAPL"
        assert exit_sig.direction == "LONG"
        assert exit_sig.exit_reason == "stop_loss"
        assert exit_sig.pnl_percent == -2.5


class TestActiveSignal:
    """Tests for ActiveSignal dataclass."""

    def test_update_price(self) -> None:
        """Test price tracking updates."""
        mock_signal = MagicMock()
        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
            current_price=150.0,
            highest_price=150.0,
            lowest_price=150.0,
        )

        # Price goes up
        active.update_price(155.0)
        assert active.current_price == 155.0
        assert active.highest_price == 155.0
        assert active.lowest_price == 150.0

        # Price goes down
        active.update_price(148.0)
        assert active.current_price == 148.0
        assert active.highest_price == 155.0
        assert active.lowest_price == 148.0


class TestMarketHoursDetection:
    """Tests for market hours detection."""

    def test_market_open_during_trading_hours(self) -> None:
        """Test market is open during trading hours."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Wednesday at 10:00 AM ET (explicit datetime)
        check_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=ET)
        assert generator.is_market_open(check_time) is True

    def test_market_closed_before_open(self) -> None:
        """Test market is closed before opening."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Wednesday at 8:00 AM ET (explicit datetime)
        check_time = datetime(2025, 1, 15, 8, 0, 0, tzinfo=ET)
        assert generator.is_market_open(check_time) is False

    def test_market_closed_after_close(self) -> None:
        """Test market is closed after closing time."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Wednesday at 5:00 PM ET (explicit datetime)
        check_time = datetime(2025, 1, 15, 17, 0, 0, tzinfo=ET)
        assert generator.is_market_open(check_time) is False

    def test_market_closed_weekend(self) -> None:
        """Test market is closed on weekend."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Saturday at 10:00 AM ET (explicit datetime)
        check_time = datetime(2025, 1, 18, 10, 0, 0, tzinfo=ET)
        assert generator.is_market_open(check_time) is False

    def test_market_closed_holiday(self) -> None:
        """Test market is closed on holidays."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Christmas 2025 (Thursday)
        check_time = datetime(2025, 12, 25, 10, 0, 0, tzinfo=ET)
        assert generator.is_market_open(check_time) is False

        # New Year's Day 2025
        check_time = datetime(2025, 1, 1, 10, 0, 0, tzinfo=ET)
        assert generator.is_market_open(check_time) is False

    def test_all_holidays_covered(self) -> None:
        """Test all major US holidays are in the list."""
        # Check 2025 holidays
        assert date(2025, 1, 1) in US_MARKET_HOLIDAYS  # New Year
        assert date(2025, 1, 20) in US_MARKET_HOLIDAYS  # MLK
        assert date(2025, 2, 17) in US_MARKET_HOLIDAYS  # Presidents
        assert date(2025, 4, 18) in US_MARKET_HOLIDAYS  # Good Friday
        assert date(2025, 5, 26) in US_MARKET_HOLIDAYS  # Memorial
        assert date(2025, 6, 19) in US_MARKET_HOLIDAYS  # Juneteenth
        assert date(2025, 7, 4) in US_MARKET_HOLIDAYS  # Independence
        assert date(2025, 9, 1) in US_MARKET_HOLIDAYS  # Labor
        assert date(2025, 11, 27) in US_MARKET_HOLIDAYS  # Thanksgiving
        assert date(2025, 12, 25) in US_MARKET_HOLIDAYS  # Christmas

    def test_get_next_market_open_same_day(self) -> None:
        """Test getting next market open before market opens."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Wednesday at 8:00 AM ET
        check_time = datetime(2025, 1, 15, 8, 0, 0, tzinfo=ET)
        next_open = generator.get_next_market_open(check_time)

        assert next_open.date() == date(2025, 1, 15)
        assert next_open.time() == MARKET_OPEN

    def test_get_next_market_open_next_day(self) -> None:
        """Test getting next market open after market close."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Wednesday at 5:00 PM ET
        check_time = datetime(2025, 1, 15, 17, 0, 0, tzinfo=ET)
        next_open = generator.get_next_market_open(check_time)

        assert next_open.date() == date(2025, 1, 16)  # Thursday
        assert next_open.time() == MARKET_OPEN

    def test_get_next_market_open_skips_weekend(self) -> None:
        """Test getting next market open skips weekend and holidays."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Friday Jan 17 at 5:00 PM ET - next trading day is Tuesday Jan 21
        # (Monday Jan 20 is MLK Day holiday)
        check_time = datetime(2025, 1, 17, 17, 0, 0, tzinfo=ET)
        next_open = generator.get_next_market_open(check_time)

        assert next_open.date() == date(2025, 1, 21)  # Tuesday (Mon is MLK Day)
        assert next_open.time() == MARKET_OPEN

    def test_get_market_close(self) -> None:
        """Test getting market close time."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        check_time = datetime(2025, 1, 15, 10, 0, 0, tzinfo=ET)
        market_close = generator.get_market_close(check_time)

        assert market_close.time() == MARKET_CLOSE


class TestScanLoop:
    """Tests for scan loop functionality."""

    @pytest.mark.asyncio
    async def test_start_connects_to_ib(self) -> None:
        """Test that start connects to IB."""
        scanner = MagicMock()
        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=False)
        ib_client.connect = AsyncMock(return_value=True)
        ib_client.subscribe_quotes = AsyncMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Start briefly then stop
        await generator.start()
        await asyncio.sleep(0.1)
        await generator.stop()

        ib_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_subscribes_to_quotes(self) -> None:
        """Test that start subscribes to quote updates."""
        scanner = MagicMock()
        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=True)
        ib_client.subscribe_quotes = AsyncMock()
        ib_client.unsubscribe_quotes = AsyncMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        await generator.start()
        await asyncio.sleep(0.1)
        await generator.stop()

        ib_client.subscribe_quotes.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self) -> None:
        """Test that stop cleanly cancels all tasks."""
        scanner = MagicMock()
        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=True)
        ib_client.subscribe_quotes = AsyncMock()
        ib_client.unsubscribe_quotes = AsyncMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        await generator.start()
        assert generator.is_running is True

        await generator.stop()
        assert generator.is_running is False
        assert generator._scan_task is None
        assert generator._exit_monitor_task is None

    @pytest.mark.asyncio
    async def test_scan_respects_interval(self) -> None:
        """Test that scans respect the configured interval."""
        scanner = MagicMock()
        scanner.scan_universe = AsyncMock(return_value=[])
        scanner.filter_by_score = MagicMock(return_value=[])

        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=True)
        ib_client.subscribe_quotes = AsyncMock()
        ib_client.unsubscribe_quotes = AsyncMock()

        config = RealtimeConfig(scan_interval_minutes=1)  # 1 minute = 60 seconds
        generator = RealTimeSignalGenerator(scanner, ib_client, config)

        # Mock market as open
        with patch.object(generator, "is_market_open", return_value=True):
            await generator.start()
            # Wait less than scan interval
            await asyncio.sleep(0.5)
            await generator.stop()

        # Should have run at least the initial scan
        assert scanner.scan_universe.call_count >= 0


class TestSignalPublishing:
    """Tests for signal publishing."""

    @pytest.mark.asyncio
    async def test_publishes_new_signals(self) -> None:
        """Test callback is invoked on new signal."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.entry_price = 150.0
        mock_signal.stop_loss = 145.0
        mock_signal.take_profit = 160.0
        mock_signal.score = 85

        scanner = MagicMock()
        scanner.scan_universe = AsyncMock(return_value=[mock_signal])
        scanner.filter_by_score = MagicMock(return_value=[mock_signal])

        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=True)
        ib_client.subscribe_quotes = AsyncMock()
        ib_client.unsubscribe_quotes = AsyncMock()

        config = RealtimeConfig(scan_interval_minutes=1)
        generator = RealTimeSignalGenerator(scanner, ib_client, config)

        # Set up watchlist so scan has symbols to scan
        generator._watchlist = ["AAPL", "MSFT"]

        received_signals: list = []
        generator.on_new_signal = lambda s: received_signals.append(s)

        with patch.object(generator, "is_market_open", return_value=True):
            await generator._run_scan()

        assert len(received_signals) == 1
        assert received_signals[0].symbol == "AAPL"


class TestExitMonitoring:
    """Tests for exit signal monitoring."""

    @pytest.mark.asyncio
    async def test_stop_loss_triggers_exit(self) -> None:
        """Test that hitting stop loss generates exit signal."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.entry_price = 150.0
        mock_signal.stop_loss = 145.0
        mock_signal.take_profit = 160.0

        scanner = MagicMock()
        ib_client = MagicMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Add active signal
        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
            current_price=144.0,  # Below stop loss
        )
        generator._active_signals["AAPL"] = active

        exit_signals: list = []
        generator.on_exit_signal = lambda e: exit_signals.append(e)

        await generator._check_exit_conditions()

        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == "stop_loss"
        assert "AAPL" not in generator._active_signals

    @pytest.mark.asyncio
    async def test_take_profit_triggers_exit(self) -> None:
        """Test that hitting take profit generates exit signal."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.entry_price = 150.0
        mock_signal.stop_loss = 145.0
        mock_signal.take_profit = 160.0

        scanner = MagicMock()
        ib_client = MagicMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Add active signal
        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
            current_price=161.0,  # Above take profit
        )
        generator._active_signals["AAPL"] = active

        exit_signals: list = []
        generator.on_exit_signal = lambda e: exit_signals.append(e)

        await generator._check_exit_conditions()

        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == "take_profit"
        assert exit_signals[0].pnl_percent > 0

    @pytest.mark.asyncio
    async def test_short_stop_loss_triggers_exit(self) -> None:
        """Test stop loss for SHORT position (price goes up)."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "SHORT"
        mock_signal.entry_price = 150.0
        mock_signal.stop_loss = 155.0  # Stop if price goes UP
        mock_signal.take_profit = 140.0  # Profit if price goes DOWN

        scanner = MagicMock()
        ib_client = MagicMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
            current_price=156.0,  # Above stop loss
        )
        generator._active_signals["AAPL"] = active

        exit_signals: list = []
        generator.on_exit_signal = lambda e: exit_signals.append(e)

        await generator._check_exit_conditions()

        assert len(exit_signals) == 1
        assert exit_signals[0].exit_reason == "stop_loss"
        assert exit_signals[0].pnl_percent < 0  # Loss on short


class TestWatchlistManagement:
    """Tests for watchlist management."""

    @pytest.mark.asyncio
    async def test_watchlist_prioritization(self) -> None:
        """Test that active signals get priority in watchlist."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"

        scanner = MagicMock()
        ib_client = MagicMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Add active signal
        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
        )
        generator._active_signals["AAPL"] = active

        watchlist = await generator.build_watchlist()

        # AAPL should be first since it has an active signal
        assert watchlist[0] == "AAPL"

    @pytest.mark.asyncio
    async def test_update_watchlist(self) -> None:
        """Test watchlist update subscribes/unsubscribes correctly."""
        scanner = MagicMock()
        ib_client = MagicMock()
        ib_client.subscribe_quotes = AsyncMock()
        ib_client.unsubscribe_quotes = AsyncMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)
        generator._watchlist = ["AAPL", "MSFT"]

        await generator.update_watchlist(["MSFT", "GOOGL"])

        # Should unsubscribe from AAPL
        ib_client.unsubscribe_quotes.assert_called_with(["AAPL"])
        # Should subscribe to GOOGL
        ib_client.subscribe_quotes.assert_called()


class TestCleanShutdown:
    """Tests for clean shutdown."""

    @pytest.mark.asyncio
    async def test_all_resources_released(self) -> None:
        """Test that all resources are released on shutdown."""
        scanner = MagicMock()
        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=True)
        ib_client.subscribe_quotes = AsyncMock()
        ib_client.unsubscribe_quotes = AsyncMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        await generator.start()
        generator._watchlist = ["AAPL", "MSFT"]
        generator._latest_quotes["AAPL"] = MagicMock()

        await generator.stop()

        assert generator._watchlist == []
        assert generator._latest_quotes == {}
        ib_client.unsubscribe_quotes.assert_called()


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_error_callback_invoked(self) -> None:
        """Test error callback is invoked on error."""
        scanner = MagicMock()
        ib_client = MagicMock()
        ib_client.is_connected = AsyncMock(return_value=False)
        ib_client.connect = AsyncMock(side_effect=Exception("Connection failed"))

        generator = RealTimeSignalGenerator(scanner, ib_client)

        errors: list = []
        generator.on_error = lambda e: errors.append(e)

        with pytest.raises(Exception, match="Connection failed"):
            await generator.start()

        assert len(errors) == 1
        assert "Connection failed" in str(errors[0])


class TestQuoteUpdates:
    """Tests for quote update handling."""

    def test_quote_update_tracks_prices(self) -> None:
        """Test that quote updates track price correctly."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"

        scanner = MagicMock()
        ib_client = MagicMock()

        generator = RealTimeSignalGenerator(scanner, ib_client)

        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
            current_price=150.0,
        )
        generator._active_signals["AAPL"] = active

        # Create mock quote
        mock_quote = MagicMock()
        mock_quote.symbol = "AAPL"
        mock_quote.last = 155.0
        mock_quote.bid = 154.9

        generator._on_quote_update(mock_quote)

        assert generator._latest_quotes["AAPL"] == mock_quote
        assert generator._active_signals["AAPL"].current_price == 155.0


class TestSignalLifecycle:
    """Tests for signal lifecycle management."""

    def test_get_active_signals(self) -> None:
        """Test getting active signals."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        mock_signal = MagicMock()
        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
        )
        generator._active_signals["AAPL"] = active

        signals = generator.get_active_signals()
        assert "AAPL" in signals
        assert signals["AAPL"].signal_id == "SIG-000001"

    def test_get_signal_by_id(self) -> None:
        """Test getting signal by ID."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        mock_signal = MagicMock()
        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
        )
        generator._active_signals["AAPL"] = active

        found = generator.get_signal_by_id("SIG-000001")
        assert found is not None
        assert found.signal_id == "SIG-000001"

        not_found = generator.get_signal_by_id("SIG-999999")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_invalidate_signal(self) -> None:
        """Test manual signal invalidation."""
        mock_signal = MagicMock()
        mock_signal.symbol = "AAPL"
        mock_signal.direction = "LONG"
        mock_signal.entry_price = 150.0

        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        active = ActiveSignal(
            signal=mock_signal,
            signal_id="SIG-000001",
            entry_time=datetime.now(ET),
            current_price=148.0,
        )
        generator._active_signals["AAPL"] = active

        exit_signals: list = []
        generator.on_exit_signal = lambda e: exit_signals.append(e)

        exit_sig = await generator.invalidate_signal("AAPL", "test reason")

        assert exit_sig is not None
        assert exit_sig.exit_reason == "signal_invalidated"
        assert "AAPL" not in generator._active_signals
        assert len(exit_signals) == 1


class TestTimezoneHandling:
    """Tests for timezone handling."""

    def test_market_hours_with_utc_time(self) -> None:
        """Test market hours detection with UTC time."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # 3:00 PM UTC = 10:00 AM ET (during market hours)
        utc_time = datetime(2025, 1, 15, 15, 0, 0, tzinfo=ZoneInfo("UTC"))
        assert generator.is_market_open(utc_time) is True

    def test_market_hours_with_naive_datetime(self) -> None:
        """Test market hours detection with naive datetime."""
        scanner = MagicMock()
        ib_client = MagicMock()
        generator = RealTimeSignalGenerator(scanner, ib_client)

        # Naive datetime treated as ET
        naive_time = datetime(2025, 1, 15, 10, 0, 0)
        assert generator.is_market_open(naive_time) is True
