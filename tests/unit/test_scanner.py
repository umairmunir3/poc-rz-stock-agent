"""Unit tests for Strategy Scanner."""

from dataclasses import dataclass
from datetime import date, datetime
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Signal, Strategy
from src.strategies.scanner import (
    DailyScanResult,
    ScanMetrics,
    StrategyScanner,
)


@dataclass
class MockStrategy(Strategy):
    """Mock strategy for testing."""

    name: str = "MockStrategy"
    description: str = "Mock strategy for testing"

    def __init__(
        self,
        name: str = "MockStrategy",
        signal_symbols: list[str] | None = None,
        raise_error: bool = False,
    ) -> None:
        self.name = name
        self.description = f"Mock {name}"
        self.signal_symbols = signal_symbols or []
        self.raise_error = raise_error

    def scan(self, df: pd.DataFrame) -> Signal | None:
        if self.raise_error:
            raise ValueError("Mock strategy error")

        symbol = df.attrs.get("symbol", "UNKNOWN")
        if symbol in self.signal_symbols:
            return Signal(
                symbol=symbol,
                strategy=self.name,
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=75,
                reasoning=f"Mock signal from {self.name}",
            )
        return None

    def check_exit(self, df: pd.DataFrame, trade) -> None:
        return None

    def get_parameters(self) -> dict:
        return {}

    def set_parameters(self, params: dict) -> None:
        pass


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create mock storage manager."""
    storage = AsyncMock()

    async def get_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.random.randint(900000, 1100000, n),
            },
            index=pd.date_range(end=datetime.now(), periods=n, freq="D"),
        )
        df.attrs["symbol"] = symbol
        return df

    storage.get_daily_bars = get_bars
    return storage


@pytest.fixture
def mock_storage_with_errors() -> AsyncMock:
    """Create mock storage that errors on some symbols."""
    storage = AsyncMock()
    error_symbols = {"BAD1", "BAD2"}

    async def get_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
        if symbol in error_symbols:
            raise ValueError(f"Failed to fetch {symbol}")

        np.random.seed(42)
        n = 100
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.random.randint(900000, 1100000, n),
            },
            index=pd.date_range(end=datetime.now(), periods=n, freq="D"),
        )
        df.attrs["symbol"] = symbol
        return df

    storage.get_daily_bars = get_bars
    return storage


class TestScanMetrics:
    """Tests for ScanMetrics dataclass."""

    def test_default_values(self) -> None:
        """Test default metric values."""
        metrics = ScanMetrics()
        assert metrics.total_duration_seconds == 0.0
        assert metrics.symbols_scanned == 0
        assert metrics.signals_generated == 0
        assert metrics.errors_count == 0
        assert metrics.strategy_durations == {}
        assert metrics.strategy_signals == {}

    def test_custom_values(self) -> None:
        """Test custom metric values."""
        metrics = ScanMetrics(
            total_duration_seconds=5.5,
            symbols_scanned=100,
            signals_generated=10,
            errors_count=2,
            strategy_durations={"RSI": 1.5, "MACD": 2.0},
            strategy_signals={"RSI": 5, "MACD": 5},
        )
        assert metrics.total_duration_seconds == 5.5
        assert metrics.symbols_scanned == 100
        assert metrics.signals_generated == 10
        assert metrics.errors_count == 2


class TestDailyScanResult:
    """Tests for DailyScanResult dataclass."""

    def test_default_values(self) -> None:
        """Test default result values."""
        result = DailyScanResult(
            date=date.today(),
            signals=[],
            metrics=ScanMetrics(),
        )
        assert result.date == date.today()
        assert result.signals == []
        assert result.errors == []


class TestStrategyScannerInit:
    """Tests for StrategyScanner initialization."""

    def test_init_with_defaults(self, mock_storage: AsyncMock) -> None:
        """Test scanner initialization with default values."""
        strategies = [MockStrategy()]
        scanner = StrategyScanner(strategies=strategies, storage=mock_storage)

        assert scanner.strategies == strategies
        assert scanner.storage == mock_storage
        assert scanner.max_concurrent_fetches == 10
        assert scanner.max_workers is None

    def test_init_with_custom_values(self, mock_storage: AsyncMock) -> None:
        """Test scanner initialization with custom values."""
        strategies = [MockStrategy()]
        scanner = StrategyScanner(
            strategies=strategies,
            storage=mock_storage,
            max_concurrent_fetches=5,
            max_workers=4,
        )

        assert scanner.max_concurrent_fetches == 5
        assert scanner.max_workers == 4


class TestScanUniverse:
    """Tests for scan_universe method."""

    @pytest.mark.asyncio
    async def test_scans_all_symbols(self, mock_storage: AsyncMock) -> None:
        """Test that all symbols are processed."""
        strategy = MockStrategy(signal_symbols=["AAPL", "GOOGL"])
        scanner = StrategyScanner(strategies=[strategy], storage=mock_storage)

        symbols = ["AAPL", "MSFT", "GOOGL"]
        signals = await scanner.scan_universe(symbols)

        # Should have signals for AAPL and GOOGL
        assert len(signals) == 2
        signal_symbols = {s.symbol for s in signals}
        assert signal_symbols == {"AAPL", "GOOGL"}

    @pytest.mark.asyncio
    async def test_runs_all_strategies(self, mock_storage: AsyncMock) -> None:
        """Test that each strategy is executed."""
        strategy1 = MockStrategy(name="Strategy1", signal_symbols=["AAPL"])
        strategy2 = MockStrategy(name="Strategy2", signal_symbols=["MSFT"])
        scanner = StrategyScanner(
            strategies=[strategy1, strategy2],
            storage=mock_storage,
        )

        symbols = ["AAPL", "MSFT"]
        signals = await scanner.scan_universe(symbols)

        # Both strategies should produce signals
        assert len(signals) == 2
        strategies_used = {s.strategy for s in signals}
        assert strategies_used == {"Strategy1", "Strategy2"}

    @pytest.mark.asyncio
    async def test_sorts_by_score(self, mock_storage: AsyncMock) -> None:
        """Test that signals are sorted by score (highest first)."""
        # Create strategies that produce different scores
        strategy = MockStrategy(signal_symbols=["AAPL", "MSFT", "GOOGL"])
        scanner = StrategyScanner(strategies=[strategy], storage=mock_storage)

        signals = await scanner.scan_universe(["AAPL", "MSFT", "GOOGL"])

        # Verify sorted by score descending
        scores = [s.score for s in signals]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_handles_individual_failures(self, mock_storage_with_errors: AsyncMock) -> None:
        """Test that one failure doesn't stop the scan."""
        strategy = MockStrategy(signal_symbols=["AAPL", "GOOGL"])
        scanner = StrategyScanner(
            strategies=[strategy],
            storage=mock_storage_with_errors,
        )

        # BAD1 and BAD2 will fail, but AAPL and GOOGL should succeed
        symbols = ["AAPL", "BAD1", "GOOGL", "BAD2"]
        signals = await scanner.scan_universe(symbols)

        # Should still get signals for successful symbols
        assert len(signals) == 2
        signal_symbols = {s.symbol for s in signals}
        assert signal_symbols == {"AAPL", "GOOGL"}

    @pytest.mark.asyncio
    async def test_handles_strategy_errors(self, mock_storage: AsyncMock) -> None:
        """Test that strategy errors are handled gracefully."""
        good_strategy = MockStrategy(name="Good", signal_symbols=["AAPL"])
        bad_strategy = MockStrategy(name="Bad", raise_error=True)
        scanner = StrategyScanner(
            strategies=[good_strategy, bad_strategy],
            storage=mock_storage,
        )

        signals = await scanner.scan_universe(["AAPL", "MSFT"])

        # Should get signal from good strategy
        assert len(signals) == 1
        assert signals[0].strategy == "Good"


class TestDeduplication:
    """Tests for signal deduplication."""

    def test_deduplicates_signals_highest_score(self, mock_storage: AsyncMock) -> None:
        """Test that same stock only appears once (highest score mode)."""
        scanner = StrategyScanner(strategies=[], storage=mock_storage)

        signals = [
            Signal(
                symbol="AAPL",
                strategy="Strategy1",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=70,
                reasoning="Low score",
            ),
            Signal(
                symbol="AAPL",
                strategy="Strategy2",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=85,
                reasoning="High score",
            ),
            Signal(
                symbol="MSFT",
                strategy="Strategy1",
                direction="LONG",
                entry_price=200.0,
                stop_loss=190.0,
                take_profit=220.0,
                score=75,
                reasoning="MSFT signal",
            ),
        ]

        deduped = scanner.deduplicate_signals(signals, mode="highest_score")

        assert len(deduped) == 2
        aapl_signal = next(s for s in deduped if s.symbol == "AAPL")
        assert aapl_signal.score == 85
        assert aapl_signal.strategy == "Strategy2"

    def test_deduplicates_signals_confluence(self, mock_storage: AsyncMock) -> None:
        """Test confluence mode creates combined signals."""
        scanner = StrategyScanner(strategies=[], storage=mock_storage)

        signals = [
            Signal(
                symbol="AAPL",
                strategy="Strategy1",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=70,
                reasoning="Signal 1",
            ),
            Signal(
                symbol="AAPL",
                strategy="Strategy2",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=75,
                reasoning="Signal 2",
            ),
        ]

        deduped = scanner.deduplicate_signals(signals, mode="confluence")

        assert len(deduped) == 1
        assert "Confluence" in deduped[0].strategy
        assert deduped[0].score > 75  # Should have bonus
        assert deduped[0].metadata["confluence_count"] == 2


class TestFiltering:
    """Tests for signal filtering."""

    @pytest.fixture
    def sample_signals(self) -> list[Signal]:
        """Create sample signals for filtering tests."""
        return [
            Signal(
                symbol="AAPL",
                strategy="RSI",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=80,
                reasoning="RSI oversold",
            ),
            Signal(
                symbol="MSFT",
                strategy="MACD",
                direction="SHORT",
                entry_price=200.0,
                stop_loss=210.0,
                take_profit=180.0,
                score=65,
                reasoning="MACD divergence",
            ),
            Signal(
                symbol="GOOGL",
                strategy="RSI",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=165.0,
                score=75,
                reasoning="RSI bounce",
            ),
        ]

    def test_filter_by_score_works(self, sample_signals: list[Signal]) -> None:
        """Test score filtering."""
        filtered = StrategyScanner.filter_by_score(sample_signals, min_score=70)

        assert len(filtered) == 2
        assert all(s.score >= 70 for s in filtered)
        assert "MSFT" not in [s.symbol for s in filtered]

    def test_filter_by_strategy_works(self, sample_signals: list[Signal]) -> None:
        """Test strategy filtering."""
        filtered = StrategyScanner.filter_by_strategy(sample_signals, strategy="RSI")

        assert len(filtered) == 2
        assert all(s.strategy == "RSI" for s in filtered)

    def test_filter_by_direction_works(self, sample_signals: list[Signal]) -> None:
        """Test direction filtering."""
        longs = StrategyScanner.filter_by_direction(sample_signals, direction="LONG")
        shorts = StrategyScanner.filter_by_direction(sample_signals, direction="SHORT")

        assert len(longs) == 2
        assert len(shorts) == 1
        assert all(s.direction == "LONG" for s in longs)
        assert all(s.direction == "SHORT" for s in shorts)


class TestDailyScan:
    """Tests for run_daily_scan method."""

    @pytest.mark.asyncio
    async def test_run_daily_scan_returns_result(self, mock_storage: AsyncMock) -> None:
        """Test that daily scan returns proper result."""
        strategy = MockStrategy(signal_symbols=["AAPL", "MSFT"])
        scanner = StrategyScanner(strategies=[strategy], storage=mock_storage)

        result = await scanner.run_daily_scan(["AAPL", "MSFT", "GOOGL"])

        assert isinstance(result, DailyScanResult)
        assert result.date == date.today()
        assert len(result.signals) == 2

    @pytest.mark.asyncio
    async def test_logs_performance_metrics(self, mock_storage: AsyncMock) -> None:
        """Test that performance metrics are captured."""
        strategy = MockStrategy(name="TestStrategy", signal_symbols=["AAPL"])
        scanner = StrategyScanner(strategies=[strategy], storage=mock_storage)

        result = await scanner.run_daily_scan(["AAPL", "MSFT"])

        assert result.metrics.total_duration_seconds > 0
        assert result.metrics.symbols_scanned == 2
        assert result.metrics.signals_generated == 1
        assert "TestStrategy" in result.metrics.strategy_durations
        assert result.metrics.strategy_signals["TestStrategy"] == 1

    @pytest.mark.asyncio
    async def test_tracks_errors(self, mock_storage_with_errors: AsyncMock) -> None:
        """Test that errors are tracked in result."""
        strategy = MockStrategy(signal_symbols=["AAPL"])
        scanner = StrategyScanner(
            strategies=[strategy],
            storage=mock_storage_with_errors,
        )

        result = await scanner.run_daily_scan(["AAPL", "BAD1"])

        assert result.metrics.errors_count > 0
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_deduplicates_when_requested(self, mock_storage: AsyncMock) -> None:
        """Test that deduplication is applied when requested."""
        strategy1 = MockStrategy(name="Strategy1", signal_symbols=["AAPL"])
        strategy2 = MockStrategy(name="Strategy2", signal_symbols=["AAPL"])
        scanner = StrategyScanner(
            strategies=[strategy1, strategy2],
            storage=mock_storage,
        )

        # With deduplication
        result_dedup = await scanner.run_daily_scan(
            ["AAPL"],
            deduplicate=True,
            dedup_mode="highest_score",
        )
        assert len(result_dedup.signals) == 1

        # Without deduplication
        result_no_dedup = await scanner.run_daily_scan(
            ["AAPL"],
            deduplicate=False,
        )
        assert len(result_no_dedup.signals) == 2


class TestIntegration:
    """Integration tests for scanner."""

    @pytest.mark.asyncio
    async def test_full_universe_scan(self, mock_storage: AsyncMock) -> None:
        """Test scanning 50 mock stocks."""
        symbols = [f"STOCK{i}" for i in range(50)]
        # Signal for every 5th stock
        signal_symbols = symbols[::5]

        strategy = MockStrategy(signal_symbols=signal_symbols)
        scanner = StrategyScanner(
            strategies=[strategy],
            storage=mock_storage,
            max_concurrent_fetches=20,
        )

        result = await scanner.run_daily_scan(symbols)

        assert result.metrics.symbols_scanned == 50
        assert len(result.signals) == 10  # Every 5th stock
        assert result.metrics.total_duration_seconds < 30  # Performance requirement

    @pytest.mark.asyncio
    async def test_parallel_execution_is_faster(self, mock_storage: AsyncMock) -> None:
        """Test that parallel execution is faster than sequential would be."""
        symbols = [f"STOCK{i}" for i in range(20)]
        strategy = MockStrategy(signal_symbols=symbols[:5])

        # With high concurrency
        scanner_parallel = StrategyScanner(
            strategies=[strategy],
            storage=mock_storage,
            max_concurrent_fetches=20,
        )

        result = await scanner_parallel.run_daily_scan(symbols)

        # Should complete quickly with parallel execution
        assert result.metrics.total_duration_seconds < 5
        assert result.metrics.symbols_scanned == 20
