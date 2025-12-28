"""Real-Time Signal Generator - generates signals during market hours."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Literal
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from src.data.ib_client import IBClient, Quote
    from src.strategies.base import Signal
    from src.strategies.scanner import StrategyScanner

logger = logging.getLogger(__name__)

# US Eastern timezone for market hours
ET = ZoneInfo("America/New_York")

# US Market holidays for 2024-2025
US_MARKET_HOLIDAYS: set[date] = {
    # 2024 holidays
    date(2024, 1, 1),  # New Year's Day
    date(2024, 1, 15),  # MLK Day
    date(2024, 2, 19),  # Presidents Day
    date(2024, 3, 29),  # Good Friday
    date(2024, 5, 27),  # Memorial Day
    date(2024, 6, 19),  # Juneteenth
    date(2024, 7, 4),  # Independence Day
    date(2024, 9, 2),  # Labor Day
    date(2024, 11, 28),  # Thanksgiving
    date(2024, 12, 25),  # Christmas
    # 2025 holidays
    date(2025, 1, 1),  # New Year's Day
    date(2025, 1, 20),  # MLK Day
    date(2025, 2, 17),  # Presidents Day
    date(2025, 4, 18),  # Good Friday
    date(2025, 5, 26),  # Memorial Day
    date(2025, 6, 19),  # Juneteenth
    date(2025, 7, 4),  # Independence Day
    date(2025, 9, 1),  # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
}

# Market hours
MARKET_OPEN = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET


@dataclass
class RealtimeConfig:
    """Configuration for real-time signal generator.

    Attributes:
        scan_interval_minutes: Minutes between scans.
        symbols_per_batch: Maximum symbols to scan per batch.
        min_score_threshold: Minimum signal score to publish.
        max_active_signals: Maximum concurrent active signals.
        exit_check_interval_seconds: Seconds between exit condition checks.
    """

    scan_interval_minutes: int = 5
    symbols_per_batch: int = 50
    min_score_threshold: int = 70
    max_active_signals: int = 20
    exit_check_interval_seconds: float = 10.0


@dataclass
class ExitSignal:
    """Exit signal for an existing position.

    Attributes:
        symbol: Stock ticker symbol.
        direction: Original trade direction.
        entry_signal_id: ID of the entry signal.
        exit_reason: Reason for exit (stop_loss, take_profit, signal_invalidated).
        exit_price: Price at which to exit.
        pnl_percent: Estimated P&L percentage.
        timestamp: When the exit signal was generated.
    """

    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_signal_id: str
    exit_reason: Literal["stop_loss", "take_profit", "signal_invalidated", "manual"]
    exit_price: float
    pnl_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ActiveSignal:
    """Tracks an active signal being monitored.

    Attributes:
        signal: The original entry signal.
        signal_id: Unique identifier for this signal.
        entry_time: When the signal was generated.
        current_price: Latest price for the symbol.
        highest_price: Highest price since entry (for trailing stops).
        lowest_price: Lowest price since entry (for trailing stops).
    """

    signal: Signal
    signal_id: str
    entry_time: datetime
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = float("inf")

    def update_price(self, price: float) -> None:
        """Update price tracking."""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)


class RealTimeSignalGenerator:
    """Generates trading signals during market hours.

    This generator runs a continuous loop during market hours,
    scanning for new signals and monitoring existing ones for exits.

    Example:
        >>> generator = RealTimeSignalGenerator(
        ...     scanner=my_scanner,
        ...     ib_client=my_ib_client,
        ...     config=RealtimeConfig(),
        ... )
        >>> generator.on_new_signal = lambda sig: print(f"New: {sig}")
        >>> generator.on_exit_signal = lambda exit: print(f"Exit: {exit}")
        >>> await generator.start()
    """

    def __init__(
        self,
        scanner: StrategyScanner,
        ib_client: IBClient,
        config: RealtimeConfig | None = None,
    ) -> None:
        """Initialize the real-time signal generator.

        Args:
            scanner: StrategyScanner for running strategy scans.
            ib_client: IBClient for market data.
            config: Configuration options.
        """
        self.scanner = scanner
        self.ib_client = ib_client
        self.config = config or RealtimeConfig()

        # State
        self._running = False
        self._scan_task: asyncio.Task[None] | None = None
        self._exit_monitor_task: asyncio.Task[None] | None = None
        self._watchlist: list[str] = []
        self._active_signals: dict[str, ActiveSignal] = {}
        self._signal_counter = 0

        # Callbacks
        self.on_new_signal: Callable[[Signal], None] | None = None
        self.on_exit_signal: Callable[[ExitSignal], None] | None = None
        self.on_error: Callable[[Exception], None] | None = None

        # Quote tracking
        self._latest_quotes: dict[str, Quote] = {}

    # -------------------------------------------------------------------------
    # Market Hours Awareness
    # -------------------------------------------------------------------------

    def is_market_open(self, check_time: datetime | None = None) -> bool:
        """Check if the market is currently open.

        Args:
            check_time: Time to check (defaults to now).

        Returns:
            True if market is open.
        """
        if check_time is None:
            check_time = datetime.now(ET)
        elif check_time.tzinfo is None:
            check_time = check_time.replace(tzinfo=ET)
        else:
            check_time = check_time.astimezone(ET)

        # Check if weekend
        if check_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if holiday
        if check_time.date() in US_MARKET_HOLIDAYS:
            return False

        # Check market hours
        current_time = check_time.time()
        return MARKET_OPEN <= current_time < MARKET_CLOSE

    def get_next_market_open(self, from_time: datetime | None = None) -> datetime:
        """Get the next market open time.

        Args:
            from_time: Starting time (defaults to now).

        Returns:
            Datetime of next market open.
        """
        if from_time is None:
            from_time = datetime.now(ET)
        elif from_time.tzinfo is None:
            from_time = from_time.replace(tzinfo=ET)
        else:
            from_time = from_time.astimezone(ET)

        current_date = from_time.date()
        current_time = from_time.time()

        # If before market open today and not holiday/weekend
        if (
            current_time < MARKET_OPEN
            and current_date.weekday() < 5
            and current_date not in US_MARKET_HOLIDAYS
        ):
            return datetime.combine(current_date, MARKET_OPEN, tzinfo=ET)

        # Find next trading day
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5 or next_date in US_MARKET_HOLIDAYS:
            next_date += timedelta(days=1)

        return datetime.combine(next_date, MARKET_OPEN, tzinfo=ET)

    def get_market_close(self, for_date: datetime | None = None) -> datetime:
        """Get the market close time for a given date.

        Args:
            for_date: Date to get close time for (defaults to today).

        Returns:
            Datetime of market close.
        """
        if for_date is None:
            for_date = datetime.now(ET)
        elif for_date.tzinfo is None:
            for_date = for_date.replace(tzinfo=ET)
        else:
            for_date = for_date.astimezone(ET)

        return datetime.combine(for_date.date(), MARKET_CLOSE, tzinfo=ET)

    # -------------------------------------------------------------------------
    # Scanning Loop
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the real-time signal generator.

        Connects to IB, subscribes to quotes, and begins the scanning loop.
        """
        if self._running:
            logger.warning("RealTimeSignalGenerator already running")
            return

        logger.info("Starting RealTimeSignalGenerator")
        self._running = True

        try:
            # Connect to IB
            if not await self.ib_client.is_connected():
                await self.ib_client.connect()

            # Build initial watchlist
            self._watchlist = await self.build_watchlist()

            # Subscribe to quotes
            if self._watchlist:
                await self.ib_client.subscribe_quotes(
                    self._watchlist,
                    self._on_quote_update,
                )

            # Start scan loop
            self._scan_task = asyncio.create_task(self._scan_loop())

            # Start exit monitor
            self._exit_monitor_task = asyncio.create_task(self._exit_monitor_loop())

            logger.info(f"RealTimeSignalGenerator started with {len(self._watchlist)} symbols")

        except Exception as e:
            self._running = False
            logger.error(f"Failed to start RealTimeSignalGenerator: {e}")
            if self.on_error:
                self.on_error(e)
            raise

    async def stop(self) -> None:
        """Stop the real-time signal generator.

        Cleanly shuts down all tasks and releases resources.
        """
        if not self._running:
            return

        logger.info("Stopping RealTimeSignalGenerator")
        self._running = False

        # Cancel tasks
        if self._scan_task:
            self._scan_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scan_task
            self._scan_task = None

        if self._exit_monitor_task:
            self._exit_monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._exit_monitor_task
            self._exit_monitor_task = None

        # Unsubscribe from quotes
        try:
            await self.ib_client.unsubscribe_quotes()
        except Exception as e:
            logger.warning(f"Error unsubscribing from quotes: {e}")

        # Clear state
        self._watchlist = []
        self._latest_quotes = {}

        logger.info("RealTimeSignalGenerator stopped")

    async def _scan_loop(self) -> None:
        """Main scanning loop that runs during market hours."""
        scan_interval = self.config.scan_interval_minutes * 60

        while self._running:
            try:
                if self.is_market_open():
                    await self._run_scan()
                else:
                    logger.debug("Market closed, skipping scan")

                # Wait for next scan interval
                await asyncio.sleep(scan_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scan loop: {e}")
                if self.on_error:
                    self.on_error(e)
                await asyncio.sleep(scan_interval)

    async def _run_scan(self) -> None:
        """Run a single scan iteration."""
        logger.info(f"Running scan on {len(self._watchlist)} symbols")

        if not self._watchlist:
            return

        # Scan for new signals
        signals = await self.scanner.scan_universe(
            symbols=self._watchlist[: self.config.symbols_per_batch],
        )

        # Filter by minimum score
        signals = self.scanner.filter_by_score(
            signals,
            min_score=self.config.min_score_threshold,
        )

        # Process new signals
        for signal in signals:
            # Skip if we already have an active signal for this symbol
            if signal.symbol in self._active_signals:
                continue

            # Skip if we've hit max active signals
            if len(self._active_signals) >= self.config.max_active_signals:
                break

            # Create and track active signal
            self._signal_counter += 1
            signal_id = f"SIG-{self._signal_counter:06d}"
            active = ActiveSignal(
                signal=signal,
                signal_id=signal_id,
                entry_time=datetime.now(ET),
                current_price=signal.entry_price,
                highest_price=signal.entry_price,
                lowest_price=signal.entry_price,
            )
            self._active_signals[signal.symbol] = active

            # Publish signal
            logger.info(f"New signal: {signal.symbol} {signal.direction} (ID: {signal_id})")
            if self.on_new_signal:
                self.on_new_signal(signal)

    async def _exit_monitor_loop(self) -> None:
        """Monitor active signals for exit conditions."""
        check_interval = self.config.exit_check_interval_seconds

        while self._running:
            try:
                if self.is_market_open():
                    await self._check_exit_conditions()

                await asyncio.sleep(check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in exit monitor: {e}")
                await asyncio.sleep(check_interval)

    async def _check_exit_conditions(self) -> None:
        """Check all active signals for exit conditions."""
        signals_to_remove: list[str] = []

        for symbol, active in self._active_signals.items():
            signal = active.signal
            current_price = active.current_price

            if current_price <= 0:
                continue

            exit_signal: ExitSignal | None = None

            if signal.direction == "LONG":
                # Check stop loss
                if current_price <= signal.stop_loss:
                    pnl = ((current_price - signal.entry_price) / signal.entry_price) * 100
                    exit_signal = ExitSignal(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_signal_id=active.signal_id,
                        exit_reason="stop_loss",
                        exit_price=current_price,
                        pnl_percent=pnl,
                    )
                # Check take profit
                elif current_price >= signal.take_profit:
                    pnl = ((current_price - signal.entry_price) / signal.entry_price) * 100
                    exit_signal = ExitSignal(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_signal_id=active.signal_id,
                        exit_reason="take_profit",
                        exit_price=current_price,
                        pnl_percent=pnl,
                    )
            else:  # SHORT
                # Check stop loss (price goes up)
                if current_price >= signal.stop_loss:
                    pnl = ((signal.entry_price - current_price) / signal.entry_price) * 100
                    exit_signal = ExitSignal(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_signal_id=active.signal_id,
                        exit_reason="stop_loss",
                        exit_price=current_price,
                        pnl_percent=pnl,
                    )
                # Check take profit (price goes down)
                elif current_price <= signal.take_profit:
                    pnl = ((signal.entry_price - current_price) / signal.entry_price) * 100
                    exit_signal = ExitSignal(
                        symbol=symbol,
                        direction=signal.direction,
                        entry_signal_id=active.signal_id,
                        exit_reason="take_profit",
                        exit_price=current_price,
                        pnl_percent=pnl,
                    )

            if exit_signal:
                logger.info(
                    f"Exit signal: {symbol} {exit_signal.exit_reason} "
                    f"(PnL: {exit_signal.pnl_percent:.2f}%)"
                )
                signals_to_remove.append(symbol)
                if self.on_exit_signal:
                    self.on_exit_signal(exit_signal)

        # Remove closed signals
        for symbol in signals_to_remove:
            del self._active_signals[symbol]

    def _on_quote_update(self, quote: Quote) -> None:
        """Handle incoming quote updates.

        Args:
            quote: The updated quote.
        """
        self._latest_quotes[quote.symbol] = quote

        # Update active signal price tracking
        if quote.symbol in self._active_signals:
            active = self._active_signals[quote.symbol]
            price = quote.last if quote.last > 0 else quote.bid
            if price > 0:
                active.update_price(price)

    # -------------------------------------------------------------------------
    # Watchlist Management
    # -------------------------------------------------------------------------

    async def build_watchlist(self) -> list[str]:
        """Build the initial watchlist for monitoring.

        Returns:
            List of symbols to monitor.
        """
        symbols: list[str] = []

        # Start with symbols that have active signals (highest priority)
        symbols.extend(self._active_signals.keys())

        # Add universe symbols from storage if available
        try:
            # Get universe from scanner's storage
            universe_symbols = await self._get_universe_symbols()
            for sym in universe_symbols:
                if sym not in symbols:
                    symbols.append(sym)
                if len(symbols) >= self.config.symbols_per_batch:
                    break
        except Exception as e:
            logger.warning(f"Could not load universe: {e}")

        return symbols[: self.config.symbols_per_batch]

    async def _get_universe_symbols(self) -> list[str]:
        """Get symbols from the scanner's storage.

        Returns:
            List of symbols.
        """
        # Default universe for swing trading
        return [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "AMD",
            "NFLX",
            "CRM",
            "ADBE",
            "ORCL",
            "CSCO",
            "INTC",
            "QCOM",
            "AVGO",
            "TXN",
            "SHOP",
            "SQ",
            "PYPL",
            "V",
            "MA",
            "JPM",
            "BAC",
            "GS",
            "MS",
            "C",
            "WFC",
            "BRK.B",
            "JNJ",
            "UNH",
            "PFE",
            "MRK",
            "ABBV",
            "LLY",
            "TMO",
            "ABT",
            "DHR",
            "XOM",
            "CVX",
            "COP",
            "SLB",
            "HD",
            "LOW",
            "TGT",
            "COST",
            "WMT",
            "NKE",
            "SBUX",
            "MCD",
        ]

    async def update_watchlist(self, new_symbols: list[str]) -> None:
        """Update the watchlist with new symbols.

        Args:
            new_symbols: New symbols to watch.
        """
        # Unsubscribe from old symbols
        old_only = [s for s in self._watchlist if s not in new_symbols]
        if old_only:
            await self.ib_client.unsubscribe_quotes(old_only)

        # Subscribe to new symbols
        new_only = [s for s in new_symbols if s not in self._watchlist]
        if new_only:
            await self.ib_client.subscribe_quotes(new_only, self._on_quote_update)

        self._watchlist = new_symbols
        logger.info(f"Watchlist updated: {len(self._watchlist)} symbols")

    # -------------------------------------------------------------------------
    # Signal Lifecycle
    # -------------------------------------------------------------------------

    def get_active_signals(self) -> dict[str, ActiveSignal]:
        """Get all currently active signals.

        Returns:
            Dictionary of symbol to active signal.
        """
        return self._active_signals.copy()

    def get_signal_by_id(self, signal_id: str) -> ActiveSignal | None:
        """Get an active signal by its ID.

        Args:
            signal_id: Signal ID to look up.

        Returns:
            The active signal, or None if not found.
        """
        for active in self._active_signals.values():
            if active.signal_id == signal_id:
                return active
        return None

    async def invalidate_signal(self, symbol: str, reason: str = "manual") -> ExitSignal | None:
        """Manually invalidate an active signal.

        Args:
            symbol: Symbol to invalidate.
            reason: Reason for invalidation.

        Returns:
            Exit signal if signal was active, None otherwise.
        """
        if symbol not in self._active_signals:
            return None

        active = self._active_signals[symbol]
        signal = active.signal
        current_price = active.current_price or signal.entry_price

        if signal.direction == "LONG":
            pnl = ((current_price - signal.entry_price) / signal.entry_price) * 100
        else:
            pnl = ((signal.entry_price - current_price) / signal.entry_price) * 100

        exit_signal = ExitSignal(
            symbol=symbol,
            direction=signal.direction,
            entry_signal_id=active.signal_id,
            exit_reason="signal_invalidated",
            exit_price=current_price,
            pnl_percent=pnl,
        )

        del self._active_signals[symbol]

        logger.info(f"Signal invalidated: {symbol} ({reason})")
        if self.on_exit_signal:
            self.on_exit_signal(exit_signal)

        return exit_signal

    @property
    def is_running(self) -> bool:
        """Check if the generator is running."""
        return self._running
