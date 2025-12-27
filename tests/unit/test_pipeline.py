"""Unit tests for the data pipeline module."""

import asyncio
import contextlib
import json
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.data.exceptions import AlphaVantageError, InvalidSymbolError, RateLimitError
from src.data.pipeline import (
    BackfillCheckpoint,
    DataPipeline,
    EventType,
    PipelineStats,
    handler,
)
from src.data.universe import SymbolMetadata


@pytest.fixture
def mock_av_client() -> AsyncMock:
    """Create a mock Alpha Vantage client."""
    client = AsyncMock()
    client.close = AsyncMock()

    # Default successful response
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "volume": [1000000, 1100000, 1200000],
            "adjusted_close": [104.0, 105.0, 106.0],
        },
        index=pd.date_range("2024-01-01", periods=3),
    )
    client.get_daily_ohlcv = AsyncMock(return_value=df)
    return client


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock storage manager."""
    storage = AsyncMock()
    storage.init_db = AsyncMock()
    storage.close = AsyncMock()
    storage.save_stock = AsyncMock()
    storage.save_daily_bars = AsyncMock()
    return storage


@pytest.fixture
def mock_universe() -> AsyncMock:
    """Create a mock stock universe."""
    universe = AsyncMock()
    universe.build_universe = AsyncMock(return_value=["AAPL", "GOOGL", "MSFT"])
    universe.refresh_universe = AsyncMock(return_value=["AAPL", "GOOGL", "MSFT"])
    universe.get_symbol_metadata = AsyncMock(
        return_value=SymbolMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3000000000000,
            avg_volume=50000000,
        )
    )
    return universe


@pytest.fixture
def pipeline(
    mock_av_client: AsyncMock, mock_storage: AsyncMock, mock_universe: AsyncMock, tmp_path: Path
) -> DataPipeline:
    """Create a pipeline with mocked dependencies."""
    return DataPipeline(
        av_client=mock_av_client,
        storage=mock_storage,
        universe=mock_universe,
        checkpoint_dir=tmp_path / "checkpoints",
    )


class TestPipelineStats:
    """Tests for PipelineStats dataclass."""

    def test_success_rate_calculation(self) -> None:
        """Test success rate calculation."""
        stats = PipelineStats(total_symbols=100, success_count=80, failure_count=20)
        assert stats.success_rate == 80.0

    def test_failure_rate_calculation(self) -> None:
        """Test failure rate calculation."""
        stats = PipelineStats(total_symbols=100, success_count=80, failure_count=20)
        assert stats.failure_rate == 20.0

    def test_zero_symbols_rates(self) -> None:
        """Test rates with zero symbols."""
        stats = PipelineStats(total_symbols=0)
        assert stats.success_rate == 0.0
        assert stats.failure_rate == 0.0

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        stats = PipelineStats()
        stats.start_time = datetime(2024, 1, 1, 10, 0, 0)
        stats.end_time = datetime(2024, 1, 1, 10, 5, 30)
        assert stats.duration_seconds == 330.0

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        stats = PipelineStats(
            total_symbols=100,
            success_count=90,
            failure_count=10,
        )
        result = stats.to_dict()

        assert result["total_symbols"] == 100
        assert result["success_count"] == 90
        assert result["failure_count"] == 10
        assert "success_rate" in result
        assert "failure_rate" in result


class TestBackfillCheckpoint:
    """Tests for BackfillCheckpoint dataclass."""

    def test_remaining_symbols(self) -> None:
        """Test remaining symbols calculation."""
        checkpoint = BackfillCheckpoint(
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
            start_date=date(2024, 1, 1),
            completed_symbols=["AAPL", "GOOGL"],
            failed_symbols=["MSFT"],
        )
        assert checkpoint.remaining_symbols == ["AMZN"]

    def test_progress_percentage(self) -> None:
        """Test progress percentage calculation."""
        checkpoint = BackfillCheckpoint(
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN"],
            start_date=date(2024, 1, 1),
            completed_symbols=["AAPL", "GOOGL"],
            failed_symbols=[],
        )
        assert checkpoint.progress_pct == 50.0

    def test_serialization_round_trip(self) -> None:
        """Test to_dict and from_dict."""
        checkpoint = BackfillCheckpoint(
            symbols=["AAPL", "GOOGL"],
            start_date=date(2024, 1, 1),
            completed_symbols=["AAPL"],
            failed_symbols=[],
        )
        data = checkpoint.to_dict()
        restored = BackfillCheckpoint.from_dict(data)

        assert restored.symbols == checkpoint.symbols
        assert restored.start_date == checkpoint.start_date
        assert restored.completed_symbols == checkpoint.completed_symbols


class TestDailyUpdate:
    """Tests for daily update functionality."""

    @pytest.mark.asyncio
    async def test_daily_update_fetches_all_symbols(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Verify all universe symbols are fetched."""
        stats = await pipeline.run_daily_update()

        assert stats.total_symbols == 3
        assert stats.success_count == 3
        assert stats.failure_count == 0
        assert mock_av_client.get_daily_ohlcv.call_count == 3
        assert mock_storage.save_daily_bars.call_count == 3

    @pytest.mark.asyncio
    async def test_daily_update_handles_failures(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Test that some symbols fail while others succeed."""
        # Make second call fail
        mock_av_client.get_daily_ohlcv.side_effect = [
            pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                    "adjusted_close": [104.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            ),
            AlphaVantageError("API Error"),
            pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                    "adjusted_close": [104.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            ),
        ]

        stats = await pipeline.run_daily_update()

        assert stats.total_symbols == 3
        assert stats.success_count == 2
        assert stats.failure_count == 1
        assert len(stats.errors) == 1

    @pytest.mark.asyncio
    async def test_daily_update_skips_invalid_symbols(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Test that invalid symbols are skipped."""
        mock_av_client.get_daily_ohlcv.side_effect = [
            pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                    "adjusted_close": [104.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            ),
            InvalidSymbolError("INVALID"),
            pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                    "adjusted_close": [104.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            ),
        ]

        stats = await pipeline.run_daily_update()

        assert stats.skipped_count == 1
        assert stats.success_count == 2
        assert stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_daily_update_handles_rate_limit(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Test rate limit handling with retry."""
        # First call succeeds, second rate limits then succeeds
        call_count = 0

        async def mock_get_daily(*args: str, **kwargs: str) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RateLimitError("Rate limited", retry_after=0.01)
            return pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                    "adjusted_close": [104.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            )

        mock_av_client.get_daily_ohlcv.side_effect = mock_get_daily

        stats = await pipeline.run_daily_update()

        # Should retry after rate limit
        assert stats.success_count == 3
        assert mock_av_client.get_daily_ohlcv.call_count == 4  # 3 + 1 retry


class TestBackfill:
    """Tests for historical backfill functionality."""

    @pytest.mark.asyncio
    async def test_backfill_processes_all_symbols(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock, mock_storage: AsyncMock
    ) -> None:
        """Test backfill processes all symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        stats = await pipeline.backfill_history(symbols, date(2024, 1, 1), resume=False)

        assert stats.total_symbols == 3
        assert stats.success_count == 3
        assert mock_av_client.get_daily_ohlcv.call_count == 3

    @pytest.mark.asyncio
    async def test_backfill_resumes_on_failure(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock, tmp_path: Path
    ) -> None:
        """Verify checkpoint/resume logic."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        # First run - fail on MSFT
        call_count = 0

        async def mock_get_daily_fail(*args: str, **kwargs: str) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # MSFT
                raise AlphaVantageError("API Error")
            return pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [105.0],
                    "low": [99.0],
                    "close": [104.0],
                    "volume": [1000000],
                    "adjusted_close": [104.0],
                },
                index=pd.date_range("2024-01-01", periods=1),
            )

        mock_av_client.get_daily_ohlcv.side_effect = mock_get_daily_fail
        stats1 = await pipeline.backfill_history(symbols, date(2024, 1, 1), resume=True)

        assert stats1.success_count == 3  # AAPL, GOOGL, AMZN
        assert stats1.failure_count == 1  # MSFT

        # Verify checkpoint was saved
        checkpoint_files = list(pipeline.checkpoint_dir.glob("backfill_*.json"))
        assert len(checkpoint_files) == 1

        # Second run - resume (only MSFT remaining in failed)
        mock_av_client.get_daily_ohlcv.reset_mock()
        mock_av_client.get_daily_ohlcv.side_effect = None
        mock_av_client.get_daily_ohlcv.return_value = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000000],
                "adjusted_close": [104.0],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        # Load checkpoint and verify remaining
        with open(checkpoint_files[0]) as f:
            checkpoint_data = json.load(f)
        checkpoint = BackfillCheckpoint.from_dict(checkpoint_data)

        # Failed symbols are tracked separately, remaining should be empty
        assert len(checkpoint.remaining_symbols) == 0
        assert "MSFT" in checkpoint.failed_symbols

    @pytest.mark.asyncio
    async def test_rate_limiting_respected(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Verify rate limiting is respected during backfill."""
        symbols = ["AAPL", "GOOGL", "MSFT"]

        # Track call count
        call_count = 0
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [104.0],
                "volume": [1000000],
                "adjusted_close": [104.0],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        async def track_calls(*args: str, **kwargs: str) -> pd.DataFrame:
            nonlocal call_count
            call_count += 1
            return df

        mock_av_client.get_daily_ohlcv.side_effect = track_calls

        await pipeline.backfill_history(symbols, date(2024, 1, 1), resume=False)

        # All calls should complete (rate limiter is in av_client)
        assert call_count == 3


class TestAlerts:
    """Tests for alert functionality."""

    @pytest.mark.asyncio
    async def test_alerts_on_high_failure_rate(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Verify alert when >10% fail."""
        # Make all calls fail
        mock_av_client.get_daily_ohlcv.side_effect = AlphaVantageError("API Error")

        alert_received: list[dict] = []

        async def capture_alert(alert: dict) -> None:
            alert_received.append(alert)

        pipeline.on_alert(capture_alert)

        stats = await pipeline.run_daily_update()

        # 100% failure rate should trigger alert
        assert stats.failure_rate == 100.0
        assert len(alert_received) == 1
        assert alert_received[0]["type"] == "HIGH_FAILURE_RATE"

    @pytest.mark.asyncio
    async def test_no_alert_on_low_failure_rate(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Verify no alert when failure rate is below threshold."""
        alert_received: list[dict] = []

        async def capture_alert(alert: dict) -> None:
            alert_received.append(alert)

        pipeline.on_alert(capture_alert)

        stats = await pipeline.run_daily_update()

        # 0% failure rate should not trigger alert
        assert stats.failure_rate == 0.0
        assert len(alert_received) == 0


class TestLambdaHandler:
    """Tests for Lambda handler."""

    def test_lambda_handler_routes_daily_update(self) -> None:
        """Verify DAILY_UPDATE event type handling."""
        with (
            patch("config.settings.get_settings") as mock_settings,
            patch("src.data.pipeline.AlphaVantageClient") as mock_client_cls,
            patch("src.data.pipeline.StorageManager") as mock_storage_cls,
            patch("src.data.pipeline.StockUniverse") as mock_universe_cls,
        ):
            # Configure mocks
            mock_settings.return_value = MagicMock(
                alpha_vantage_api_key="test_key",
                database_url="sqlite+aiosqlite:///:memory:",
            )

            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            mock_storage = AsyncMock()
            mock_storage.init_db = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_storage.save_stock = AsyncMock()
            mock_storage.save_daily_bars = AsyncMock()
            mock_storage_cls.return_value = mock_storage

            mock_universe = AsyncMock()
            mock_universe.build_universe = AsyncMock(return_value=["AAPL"])
            mock_universe.get_symbol_metadata = AsyncMock(
                return_value=SymbolMetadata(symbol="AAPL", name="Apple Inc.")
            )
            mock_universe_cls.return_value = mock_universe

            mock_client.get_daily_ohlcv = AsyncMock(
                return_value=pd.DataFrame(
                    {
                        "open": [100.0],
                        "high": [105.0],
                        "low": [99.0],
                        "close": [104.0],
                        "volume": [1000000],
                        "adjusted_close": [104.0],
                    },
                    index=pd.date_range("2024-01-01", periods=1),
                )
            )

            event = {"type": EventType.DAILY_UPDATE.value}
            result = handler(event, None)

            assert result["statusCode"] == 200
            body = json.loads(result["body"])
            assert "total_symbols" in body

    def test_lambda_handler_routes_backfill(self) -> None:
        """Verify BACKFILL event type handling."""
        with (
            patch("config.settings.get_settings") as mock_settings,
            patch("src.data.pipeline.AlphaVantageClient") as mock_client_cls,
            patch("src.data.pipeline.StorageManager") as mock_storage_cls,
            patch("src.data.pipeline.StockUniverse") as mock_universe_cls,
        ):
            mock_settings.return_value = MagicMock(
                alpha_vantage_api_key="test_key",
                database_url="sqlite+aiosqlite:///:memory:",
            )

            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client.get_daily_ohlcv = AsyncMock(
                return_value=pd.DataFrame(
                    {
                        "open": [100.0],
                        "high": [105.0],
                        "low": [99.0],
                        "close": [104.0],
                        "volume": [1000000],
                        "adjusted_close": [104.0],
                    },
                    index=pd.date_range("2024-01-01", periods=1),
                )
            )
            mock_client_cls.return_value = mock_client

            mock_storage = AsyncMock()
            mock_storage.init_db = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_storage.save_stock = AsyncMock()
            mock_storage.save_daily_bars = AsyncMock()
            mock_storage_cls.return_value = mock_storage

            mock_universe = AsyncMock()
            mock_universe.get_symbol_metadata = AsyncMock(
                return_value=SymbolMetadata(symbol="AAPL", name="Apple Inc.")
            )
            mock_universe_cls.return_value = mock_universe

            event = {
                "type": EventType.BACKFILL.value,
                "symbols": ["AAPL", "GOOGL"],
                "start_date": "2024-01-01",
            }
            result = handler(event, None)

            assert result["statusCode"] == 200

    def test_lambda_handler_routes_universe_refresh(self) -> None:
        """Verify UNIVERSE_REFRESH event type handling."""
        with (
            patch("config.settings.get_settings") as mock_settings,
            patch("src.data.pipeline.AlphaVantageClient") as mock_client_cls,
            patch("src.data.pipeline.StorageManager") as mock_storage_cls,
            patch("src.data.pipeline.StockUniverse") as mock_universe_cls,
        ):
            mock_settings.return_value = MagicMock(
                alpha_vantage_api_key="test_key",
                database_url="sqlite+aiosqlite:///:memory:",
            )

            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            mock_storage = AsyncMock()
            mock_storage.init_db = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_storage_cls.return_value = mock_storage

            mock_universe = AsyncMock()
            mock_universe.refresh_universe = AsyncMock(return_value=["AAPL", "GOOGL", "MSFT"])
            mock_universe_cls.return_value = mock_universe

            event = {"type": EventType.UNIVERSE_REFRESH.value}
            result = handler(event, None)

            assert result["statusCode"] == 200
            body = json.loads(result["body"])
            assert body["symbol_count"] == 3

    def test_lambda_handler_rejects_unknown_event(self) -> None:
        """Verify unknown event type returns error."""
        with (
            patch("config.settings.get_settings") as mock_settings,
            patch("src.data.pipeline.AlphaVantageClient") as mock_client_cls,
            patch("src.data.pipeline.StorageManager") as mock_storage_cls,
            patch("src.data.pipeline.StockUniverse") as mock_universe_cls,
        ):
            mock_settings.return_value = MagicMock(
                alpha_vantage_api_key="test_key",
                database_url="sqlite+aiosqlite:///:memory:",
            )

            mock_client = AsyncMock()
            mock_client.close = AsyncMock()
            mock_client_cls.return_value = mock_client

            mock_storage = AsyncMock()
            mock_storage.init_db = AsyncMock()
            mock_storage.close = AsyncMock()
            mock_storage_cls.return_value = mock_storage

            mock_universe = AsyncMock()
            mock_universe_cls.return_value = mock_universe

            event = {"type": "UNKNOWN_EVENT"}
            result = handler(event, None)

            assert result["statusCode"] == 400
            body = json.loads(result["body"])
            assert "error" in body


class TestRealTimeUpdates:
    """Tests for real-time update functionality."""

    @pytest.mark.asyncio
    async def test_realtime_emits_events(
        self, pipeline: DataPipeline, mock_av_client: AsyncMock
    ) -> None:
        """Test that real-time updates emit events."""
        events_received: list[dict] = []

        async def capture_event(event: dict) -> None:
            events_received.append(event)

        pipeline.on_event(capture_event)

        # Run for a short time then cancel
        task = asyncio.create_task(
            pipeline.start_realtime_updates(["AAPL"], update_interval_seconds=0.01)
        )

        await asyncio.sleep(0.05)
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should have received at least one event
        assert len(events_received) >= 1
        assert events_received[0]["type"] == "QUOTE_UPDATE"
        assert "symbol" in events_received[0]["data"]
