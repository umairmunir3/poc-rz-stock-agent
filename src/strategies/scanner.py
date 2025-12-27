"""Strategy Scanner - batch processing for scanning multiple stocks with multiple strategies."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal

import pandas as pd

from src.strategies.base import Signal, Strategy

if TYPE_CHECKING:
    from src.data.storage import StorageManager

logger = logging.getLogger(__name__)


@dataclass
class ScanMetrics:
    """Performance metrics for a scan operation.

    Attributes:
        total_duration_seconds: Total scan duration
        symbols_scanned: Number of symbols processed
        signals_generated: Total signals generated
        errors_count: Number of errors encountered
        strategy_durations: Duration per strategy in seconds
        strategy_signals: Signals generated per strategy
    """

    total_duration_seconds: float = 0.0
    symbols_scanned: int = 0
    signals_generated: int = 0
    errors_count: int = 0
    strategy_durations: dict[str, float] = field(default_factory=dict)
    strategy_signals: dict[str, int] = field(default_factory=dict)


@dataclass
class DailyScanResult:
    """Result of a daily scan operation.

    Attributes:
        date: Scan date
        signals: All signals generated
        metrics: Performance metrics
        errors: List of error messages
    """

    date: date
    signals: list[Signal]
    metrics: ScanMetrics
    errors: list[str] = field(default_factory=list)


class StrategyScanner:
    """Scans multiple stocks with multiple strategies.

    This scanner efficiently processes a universe of stocks using
    multiple trading strategies, supporting parallel execution
    for both I/O and CPU-bound operations.

    Example:
        >>> scanner = StrategyScanner(
        ...     strategies=[RSIMeanReversionStrategy(), BreakoutStrategy()],
        ...     storage=storage_manager,
        ... )
        >>> signals = await scanner.scan_universe(["AAPL", "MSFT", "GOOGL"])
    """

    def __init__(
        self,
        strategies: list[Strategy],
        storage: StorageManager,
        max_concurrent_fetches: int = 10,
        max_workers: int | None = None,
    ) -> None:
        """Initialize the scanner.

        Args:
            strategies: List of strategies to run on each symbol.
            storage: StorageManager for fetching OHLCV data.
            max_concurrent_fetches: Maximum concurrent data fetches.
            max_workers: Maximum workers for CPU-bound operations.
        """
        self.strategies = strategies
        self.storage = storage
        self.max_concurrent_fetches = max_concurrent_fetches
        self.max_workers = max_workers
        self._semaphore: asyncio.Semaphore | None = None

    async def scan_universe(
        self,
        symbols: list[str],
        scan_date: date | None = None,
        lookback_days: int = 100,
    ) -> list[Signal]:
        """Scan a universe of symbols with all strategies.

        Args:
            symbols: List of stock symbols to scan.
            scan_date: Date to scan (defaults to today).
            lookback_days: Number of historical days to fetch.

        Returns:
            List of signals sorted by score (highest first).
        """
        start_time = time.time()
        self._semaphore = asyncio.Semaphore(self.max_concurrent_fetches)

        if scan_date is None:
            scan_date = date.today()

        # Calculate date range
        from datetime import timedelta

        start_date = scan_date - timedelta(days=lookback_days)

        logger.info(
            f"Starting universe scan: {len(symbols)} symbols, {len(self.strategies)} strategies"
        )

        # Fetch data for all symbols concurrently
        data_tasks = [self._fetch_symbol_data(symbol, start_date, scan_date) for symbol in symbols]
        data_results = await asyncio.gather(*data_tasks, return_exceptions=True)

        # Build symbol -> DataFrame mapping, filtering out errors
        symbol_data: dict[str, pd.DataFrame] = {}
        errors: list[str] = []

        for symbol, result in zip(symbols, data_results, strict=False):
            if isinstance(result, Exception):
                errors.append(f"{symbol}: {result}")
                logger.warning(f"Failed to fetch data for {symbol}: {result}")
            elif isinstance(result, pd.DataFrame) and not result.empty:
                result.attrs["symbol"] = symbol
                symbol_data[symbol] = result

        logger.info(f"Fetched data for {len(symbol_data)}/{len(symbols)} symbols")

        # Run strategies on all symbols
        all_signals: list[Signal] = []
        strategy_durations: dict[str, float] = {}
        strategy_signals: dict[str, int] = {}

        for strategy in self.strategies:
            strategy_start = time.time()
            strategy_name = strategy.name

            signals = await self._run_strategy_on_symbols(strategy, symbol_data)
            all_signals.extend(signals)

            strategy_durations[strategy_name] = time.time() - strategy_start
            strategy_signals[strategy_name] = len(signals)

            logger.info(
                f"Strategy {strategy_name}: {len(signals)} signals "
                f"in {strategy_durations[strategy_name]:.2f}s"
            )

        # Sort by score (highest first)
        all_signals.sort(key=lambda s: s.score, reverse=True)

        total_duration = time.time() - start_time
        logger.info(
            f"Universe scan complete: {len(all_signals)} signals "
            f"in {total_duration:.2f}s, {len(errors)} errors"
        )

        return all_signals

    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV data for a single symbol with rate limiting.

        Args:
            symbol: Stock ticker symbol.
            start_date: Start of date range.
            end_date: End of date range.

        Returns:
            DataFrame with OHLCV data, or None if fetch failed.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_fetches)

        async with self._semaphore:
            try:
                df = await self.storage.get_daily_bars(symbol, start_date, end_date)
                return df
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                raise

    async def _run_strategy_on_symbols(
        self,
        strategy: Strategy,
        symbol_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """Run a single strategy on all symbols.

        Args:
            strategy: Strategy to run.
            symbol_data: Mapping of symbol to DataFrame.

        Returns:
            List of signals generated.
        """
        signals: list[Signal] = []
        errors: list[str] = []

        for symbol, df in symbol_data.items():
            try:
                signal = strategy.scan(df)
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                errors.append(f"{strategy.name}/{symbol}: {e}")
                logger.debug(f"Strategy {strategy.name} error on {symbol}: {e}")

        return signals

    def deduplicate_signals(
        self,
        signals: list[Signal],
        mode: str = "highest_score",
    ) -> list[Signal]:
        """Remove duplicate signals for the same symbol.

        Args:
            signals: List of signals to deduplicate.
            mode: Deduplication mode:
                - "highest_score": Keep signal with highest score
                - "confluence": Create combined signal with bonus points

        Returns:
            Deduplicated list of signals.
        """
        if mode == "confluence":
            return self._deduplicate_confluence(signals)
        return self._deduplicate_highest_score(signals)

    def _deduplicate_highest_score(self, signals: list[Signal]) -> list[Signal]:
        """Keep only the highest scoring signal per symbol.

        Args:
            signals: List of signals.

        Returns:
            Deduplicated signals.
        """
        symbol_signals: dict[str, Signal] = {}

        for signal in signals:
            existing = symbol_signals.get(signal.symbol)
            if existing is None or signal.score > existing.score:
                symbol_signals[signal.symbol] = signal

        return list(symbol_signals.values())

    def _deduplicate_confluence(self, signals: list[Signal]) -> list[Signal]:
        """Create confluence signals for symbols with multiple strategies.

        When multiple strategies signal the same symbol and direction,
        create a combined signal with bonus points.

        Args:
            signals: List of signals.

        Returns:
            Deduplicated signals with confluence bonuses.
        """
        # Group by symbol and direction
        symbol_direction_signals: dict[tuple[str, Literal["LONG", "SHORT"]], list[Signal]] = {}

        for signal in signals:
            key = (signal.symbol, signal.direction)
            if key not in symbol_direction_signals:
                symbol_direction_signals[key] = []
            symbol_direction_signals[key].append(signal)

        result: list[Signal] = []

        for (symbol, direction), group in symbol_direction_signals.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Multiple strategies agree - create confluence signal
                base_signal = max(group, key=lambda s: s.score)
                confluence_bonus = min(10 * (len(group) - 1), 20)  # Max +20 bonus
                new_score = min(base_signal.score + confluence_bonus, 100)

                strategies = [s.strategy for s in group]
                confluence_signal = Signal(
                    symbol=symbol,
                    strategy=f"Confluence({','.join(strategies)})",
                    direction=direction,
                    entry_price=base_signal.entry_price,
                    stop_loss=base_signal.stop_loss,
                    take_profit=base_signal.take_profit,
                    score=new_score,
                    reasoning=f"Confluence: {len(group)} strategies agree. "
                    + base_signal.reasoning,
                    timestamp=base_signal.timestamp,
                    metadata={
                        "confluence_count": len(group),
                        "strategies": strategies,
                        "original_score": base_signal.score,
                        "bonus": confluence_bonus,
                    },
                )
                result.append(confluence_signal)

        return result

    @staticmethod
    def filter_by_score(signals: list[Signal], min_score: int = 70) -> list[Signal]:
        """Filter signals by minimum score.

        Args:
            signals: List of signals to filter.
            min_score: Minimum score threshold.

        Returns:
            Filtered signals.
        """
        return [s for s in signals if s.score >= min_score]

    @staticmethod
    def filter_by_strategy(signals: list[Signal], strategy: str) -> list[Signal]:
        """Filter signals by strategy name.

        Args:
            signals: List of signals to filter.
            strategy: Strategy name to filter for.

        Returns:
            Filtered signals.
        """
        return [s for s in signals if s.strategy == strategy]

    @staticmethod
    def filter_by_direction(
        signals: list[Signal],
        direction: str,
    ) -> list[Signal]:
        """Filter signals by direction.

        Args:
            signals: List of signals to filter.
            direction: Direction to filter for ("LONG" or "SHORT").

        Returns:
            Filtered signals.
        """
        return [s for s in signals if s.direction == direction]

    async def run_daily_scan(
        self,
        symbols: list[str],
        scan_date: date | None = None,
        lookback_days: int = 100,
        deduplicate: bool = True,
        dedup_mode: str = "highest_score",
    ) -> DailyScanResult:
        """Run a complete daily scan with performance tracking.

        Args:
            symbols: List of symbols to scan.
            scan_date: Date to scan (defaults to today).
            lookback_days: Number of historical days to fetch.
            deduplicate: Whether to deduplicate signals.
            dedup_mode: Deduplication mode.

        Returns:
            DailyScanResult with signals, metrics, and errors.
        """
        start_time = time.time()
        if scan_date is None:
            scan_date = date.today()

        self._semaphore = asyncio.Semaphore(self.max_concurrent_fetches)

        from datetime import timedelta

        start_date = scan_date - timedelta(days=lookback_days)

        logger.info(
            f"Starting daily scan for {scan_date}: {len(symbols)} symbols, "
            f"{len(self.strategies)} strategies"
        )

        # Fetch data for all symbols concurrently
        data_tasks = [self._fetch_symbol_data(symbol, start_date, scan_date) for symbol in symbols]
        data_results = await asyncio.gather(*data_tasks, return_exceptions=True)

        # Build symbol -> DataFrame mapping
        symbol_data: dict[str, pd.DataFrame] = {}
        errors: list[str] = []
        errors_count = 0

        for symbol, result in zip(symbols, data_results, strict=False):
            if isinstance(result, Exception):
                errors.append(f"Fetch {symbol}: {result}")
                errors_count += 1
            elif isinstance(result, pd.DataFrame) and not result.empty:
                result.attrs["symbol"] = symbol
                symbol_data[symbol] = result

        # Run strategies
        all_signals: list[Signal] = []
        strategy_durations: dict[str, float] = {}
        strategy_signals: dict[str, int] = {}

        for strategy in self.strategies:
            strategy_start = time.time()
            strategy_name = strategy.name

            try:
                signals = await self._run_strategy_on_symbols(strategy, symbol_data)
                all_signals.extend(signals)
                strategy_signals[strategy_name] = len(signals)
            except Exception as e:
                errors.append(f"Strategy {strategy_name}: {e}")
                errors_count += 1
                strategy_signals[strategy_name] = 0

            strategy_durations[strategy_name] = time.time() - strategy_start

        # Deduplicate if requested
        if deduplicate:
            all_signals = self.deduplicate_signals(all_signals, mode=dedup_mode)

        # Sort by score
        all_signals.sort(key=lambda s: s.score, reverse=True)

        # Build metrics
        total_duration = time.time() - start_time
        metrics = ScanMetrics(
            total_duration_seconds=total_duration,
            symbols_scanned=len(symbol_data),
            signals_generated=len(all_signals),
            errors_count=errors_count,
            strategy_durations=strategy_durations,
            strategy_signals=strategy_signals,
        )

        logger.info(f"Daily scan complete: {len(all_signals)} signals in {total_duration:.2f}s")

        return DailyScanResult(
            date=scan_date,
            signals=all_signals,
            metrics=metrics,
            errors=errors,
        )
