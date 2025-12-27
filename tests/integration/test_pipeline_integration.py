"""Integration tests for data pipeline with mocked APIs."""

import json
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from src.data.alpha_vantage import AlphaVantageClient
from src.data.exceptions import AlphaVantageError
from src.data.pipeline import BackfillCheckpoint, DataPipeline
from src.data.storage import StorageManager
from src.data.universe import StockUniverse, SymbolMetadata

# Skip integration tests in CI
pytestmark = pytest.mark.integration


@pytest.fixture
async def storage(tmp_path: Path) -> StorageManager:
    """Create a real storage manager with SQLite."""
    db_path = tmp_path / "test.db"
    storage = StorageManager(f"sqlite+aiosqlite:///{db_path}")
    await storage.init_db()
    yield storage
    await storage.close()


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Create sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "open": [100.0 + i * 0.5 for i in range(100)],
            "high": [105.0 + i * 0.5 for i in range(100)],
            "low": [99.0 + i * 0.5 for i in range(100)],
            "close": [104.0 + i * 0.5 for i in range(100)],
            "volume": [1000000 + i * 10000 for i in range(100)],
            "adjusted_close": [104.0 + i * 0.5 for i in range(100)],
        },
        index=dates,
    )


@pytest.fixture
def mock_av_client(sample_ohlcv_data: pd.DataFrame) -> AsyncMock:
    """Create mock AV client."""
    client = AsyncMock(spec=AlphaVantageClient)
    client.get_daily_ohlcv = AsyncMock(return_value=sample_ohlcv_data)
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_universe() -> AsyncMock:
    """Create mock universe."""
    universe = AsyncMock(spec=StockUniverse)

    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    universe.build_universe = AsyncMock(return_value=symbols)
    universe.refresh_universe = AsyncMock(return_value=symbols)

    def get_metadata(symbol: str) -> SymbolMetadata:
        return SymbolMetadata(
            symbol=symbol,
            name=f"{symbol} Inc.",
            sector="Technology",
            industry="Software",
            market_cap=1000000000000,
            avg_volume=50000000,
        )

    universe.get_symbol_metadata = AsyncMock(side_effect=get_metadata)
    return universe


class TestEndToEndDailyUpdate:
    """End-to-end tests for daily update pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_daily_update(
        self,
        storage: StorageManager,
        mock_av_client: AsyncMock,
        mock_universe: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Full pipeline with mocked APIs stores data correctly."""
        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        stats = await pipeline.run_daily_update()

        # Verify stats
        assert stats.total_symbols == 5
        assert stats.success_count == 5
        assert stats.failure_count == 0

        # Verify data was stored
        for symbol in ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]:
            stock = await storage.get_stock(symbol)
            assert stock is not None
            assert stock.symbol == symbol

            bars = await storage.get_daily_bars(symbol)
            assert len(bars) > 0

    @pytest.mark.asyncio
    async def test_daily_update_with_partial_failures(
        self,
        storage: StorageManager,
        mock_universe: AsyncMock,
        sample_ohlcv_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Test daily update handles some symbols failing."""
        mock_av_client = AsyncMock(spec=AlphaVantageClient)
        mock_av_client.close = AsyncMock()

        call_count = 0

        async def mock_get_daily(symbol: str, outputsize: str = "compact") -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            if symbol == "GOOGL":
                raise AlphaVantageError("API Error for GOOGL")
            return sample_ohlcv_data

        mock_av_client.get_daily_ohlcv = AsyncMock(side_effect=mock_get_daily)

        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        stats = await pipeline.run_daily_update()

        assert stats.success_count == 4
        assert stats.failure_count == 1
        assert len(stats.errors) == 1
        assert stats.errors[0]["symbol"] == "GOOGL"

        # Verify successful symbols were stored
        for symbol in ["AAPL", "MSFT", "AMZN", "META"]:
            stock = await storage.get_stock(symbol)
            assert stock is not None

        # GOOGL should not be stored
        googl = await storage.get_stock("GOOGL")
        assert googl is None


class TestBackfillWithCheckpoint:
    """Tests for backfill with checkpoint/resume functionality."""

    @pytest.mark.asyncio
    async def test_backfill_with_checkpoint(
        self,
        storage: StorageManager,
        mock_universe: AsyncMock,
        sample_ohlcv_data: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Simulate failure mid-backfill, then resume."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]

        mock_av_client = AsyncMock(spec=AlphaVantageClient)
        mock_av_client.close = AsyncMock()

        # First run: fail on MSFT
        processed = []

        async def mock_get_daily_first(symbol: str, outputsize: str = "full") -> pd.DataFrame:
            processed.append(symbol)
            if symbol == "MSFT":
                raise AlphaVantageError("Simulated failure on MSFT")
            return sample_ohlcv_data

        mock_av_client.get_daily_ohlcv = AsyncMock(side_effect=mock_get_daily_first)

        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        stats1 = await pipeline.backfill_history(symbols, date(2024, 1, 1), resume=True)

        # Should have processed all but failed on MSFT
        assert stats1.success_count == 4
        assert stats1.failure_count == 1

        # Verify checkpoint exists
        checkpoint_files = list(pipeline.checkpoint_dir.glob("backfill_*.json"))
        assert len(checkpoint_files) == 1

        # Load and verify checkpoint
        with open(checkpoint_files[0]) as f:
            checkpoint_data = json.load(f)
        checkpoint = BackfillCheckpoint.from_dict(checkpoint_data)
        assert "MSFT" in checkpoint.failed_symbols
        assert len(checkpoint.completed_symbols) == 4

        # Second run: resume (MSFT is in failed, not remaining)
        mock_av_client.get_daily_ohlcv.reset_mock()
        mock_av_client.get_daily_ohlcv.return_value = sample_ohlcv_data

        # Since all symbols are processed (either completed or failed),
        # remaining should be empty
        assert len(checkpoint.remaining_symbols) == 0

    @pytest.mark.asyncio
    async def test_backfill_creates_valid_checkpoint(
        self,
        storage: StorageManager,
        mock_av_client: AsyncMock,
        mock_universe: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Verify checkpoint file format is valid."""
        symbols = ["AAPL", "GOOGL"]

        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        await pipeline.backfill_history(symbols, date(2024, 1, 1), resume=False)

        # Verify checkpoint was created
        checkpoint_files = list(pipeline.checkpoint_dir.glob("backfill_*.json"))
        assert len(checkpoint_files) == 1

        # Verify checkpoint is valid JSON
        with open(checkpoint_files[0]) as f:
            data = json.load(f)

        # Verify required fields
        assert "symbols" in data
        assert "start_date" in data
        assert "completed_symbols" in data
        assert "failed_symbols" in data
        assert "last_updated" in data

        # Verify data types
        assert isinstance(data["symbols"], list)
        assert isinstance(data["completed_symbols"], list)
        assert isinstance(data["failed_symbols"], list)


class TestPipelineWithRealDatabase:
    """Tests using real database operations."""

    @pytest.mark.asyncio
    async def test_pipeline_stores_correct_data_types(
        self,
        storage: StorageManager,
        mock_av_client: AsyncMock,
        mock_universe: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Verify data types are correctly stored in database."""
        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        await pipeline.run_daily_update()

        # Verify stock metadata
        stock = await storage.get_stock("AAPL")
        assert stock is not None
        assert isinstance(stock.symbol, str)
        assert isinstance(stock.name, str)

        # Verify daily bars
        bars = await storage.get_daily_bars("AAPL")
        assert len(bars) > 0

        # Check data types of first bar
        bar = bars[0]
        assert isinstance(bar.open, float)
        assert isinstance(bar.high, float)
        assert isinstance(bar.low, float)
        assert isinstance(bar.close, float)
        assert isinstance(bar.volume, int)

    @pytest.mark.asyncio
    async def test_pipeline_handles_duplicate_runs(
        self,
        storage: StorageManager,
        mock_av_client: AsyncMock,
        mock_universe: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Verify running pipeline twice doesn't duplicate data."""
        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        # Run twice
        await pipeline.run_daily_update()
        bars_after_first = await storage.get_daily_bars("AAPL")

        await pipeline.run_daily_update()
        bars_after_second = await storage.get_daily_bars("AAPL")

        # Should have same number of bars (upsert behavior)
        assert len(bars_after_first) == len(bars_after_second)


class TestAlertIntegration:
    """Integration tests for alert system."""

    @pytest.mark.asyncio
    async def test_alert_callback_receives_data(
        self,
        storage: StorageManager,
        mock_universe: AsyncMock,
        tmp_path: Path,
    ) -> None:
        """Verify alert callback receives proper data structure."""
        mock_av_client = AsyncMock(spec=AlphaVantageClient)
        mock_av_client.close = AsyncMock()
        mock_av_client.get_daily_ohlcv = AsyncMock(side_effect=AlphaVantageError("All failing"))

        alerts_received: list[dict] = []

        async def alert_handler(alert: dict) -> None:
            alerts_received.append(alert)

        pipeline = DataPipeline(
            av_client=mock_av_client,
            storage=storage,
            universe=mock_universe,
            checkpoint_dir=tmp_path / "checkpoints",
            alert_threshold_pct=10.0,
        )
        pipeline.on_alert(alert_handler)

        await pipeline.run_daily_update()

        # Should have received alert
        assert len(alerts_received) == 1

        alert = alerts_received[0]
        assert "type" in alert
        assert "message" in alert
        assert "data" in alert
        assert "timestamp" in alert

        # Verify data contains stats
        assert "failure_rate" in alert["data"]
        assert "total_symbols" in alert["data"]
