"""Data pipeline orchestration for fetching, processing, and storage."""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.alpha_vantage import AlphaVantageClient
from src.data.exceptions import InvalidSymbolError, RateLimitError
from src.data.storage import StorageManager
from src.data.universe import StockUniverse

logger = logging.getLogger(__name__)

# Checkpoint directory for backfill resume
CHECKPOINT_DIR = Path(".checkpoints")


class EventType(str, Enum):
    """Lambda event types for EventBridge."""

    DAILY_UPDATE = "DAILY_UPDATE"
    BACKFILL = "BACKFILL"
    UNIVERSE_REFRESH = "UNIVERSE_REFRESH"


@dataclass
class PipelineStats:
    """Statistics for pipeline operations."""

    total_symbols: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.success_count / self.total_symbols) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.failure_count / self.total_symbols) * 100

    @property
    def duration_seconds(self) -> float:
        """Calculate operation duration in seconds."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "total_symbols": self.total_symbols,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "skipped_count": self.skipped_count,
            "success_rate": f"{self.success_rate:.2f}%",
            "failure_rate": f"{self.failure_rate:.2f}%",
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_count": len(self.errors),
        }


@dataclass
class BackfillCheckpoint:
    """Checkpoint for resuming backfill operations."""

    symbols: list[str]
    start_date: date
    completed_symbols: list[str] = field(default_factory=list)
    failed_symbols: list[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat(),
            "completed_symbols": self.completed_symbols,
            "failed_symbols": self.failed_symbols,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackfillCheckpoint":
        """Create from dictionary."""
        return cls(
            symbols=data["symbols"],
            start_date=date.fromisoformat(data["start_date"]),
            completed_symbols=data.get("completed_symbols", []),
            failed_symbols=data.get("failed_symbols", []),
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )

    @property
    def remaining_symbols(self) -> list[str]:
        """Get symbols that haven't been processed yet."""
        processed = set(self.completed_symbols) | set(self.failed_symbols)
        return [s for s in self.symbols if s not in processed]

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if not self.symbols:
            return 100.0
        processed = len(self.completed_symbols) + len(self.failed_symbols)
        return (processed / len(self.symbols)) * 100


class DataPipeline:
    """Orchestrates data fetching, processing, and storage.

    Manages daily updates, historical backfill, and real-time streaming.
    """

    def __init__(
        self,
        av_client: AlphaVantageClient,
        storage: StorageManager,
        universe: StockUniverse,
        alert_threshold_pct: float = 10.0,
        checkpoint_dir: Path | None = None,
    ) -> None:
        """Initialize the data pipeline.

        Args:
            av_client: Alpha Vantage API client.
            storage: Database storage manager.
            universe: Stock universe for filtering.
            alert_threshold_pct: Failure rate threshold for alerts (default 10%).
            checkpoint_dir: Directory for checkpoint files.
        """
        self.av_client = av_client
        self.storage = storage
        self.universe = universe
        self.alert_threshold_pct = alert_threshold_pct
        self.checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR

        # Callbacks for alerts and events
        self._alert_callbacks: list[Any] = []
        self._event_callbacks: list[Any] = []

    def on_alert(self, callback: Any) -> None:
        """Register an alert callback."""
        self._alert_callbacks.append(callback)

    def on_event(self, callback: Any) -> None:
        """Register an event callback for real-time updates."""
        self._event_callbacks.append(callback)

    async def _emit_alert(self, alert_type: str, message: str, data: dict[str, Any]) -> None:
        """Emit an alert to all registered callbacks."""
        alert = {
            "type": alert_type,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        logger.warning(f"ALERT [{alert_type}]: {message}")
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event for real-time updates."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")

    # =========================================================================
    # Daily Update
    # =========================================================================

    async def run_daily_update(self) -> PipelineStats:
        """Run daily data update for all universe symbols.

        Fetches daily OHLCV data for each symbol in the universe,
        stores in database, and handles rate limits with backoff.

        Returns:
            Pipeline statistics with success/failure counts.
        """
        stats = PipelineStats()
        logger.info("Starting daily update...")

        # Get universe symbols
        try:
            symbols = await self.universe.build_universe()
        except Exception as e:
            logger.error(f"Failed to build universe: {e}")
            stats.errors.append({"type": "universe_error", "error": str(e)})
            stats.end_time = datetime.utcnow()
            return stats

        stats.total_symbols = len(symbols)
        logger.info(f"Processing {stats.total_symbols} symbols")

        # Process each symbol
        for i, symbol in enumerate(symbols):
            try:
                await self._fetch_and_store_daily(symbol)
                stats.success_count += 1

                if (i + 1) % 50 == 0:
                    logger.info(
                        f"Progress: {i + 1}/{stats.total_symbols} "
                        f"({stats.success_count} success, {stats.failure_count} failed)"
                    )

            except RateLimitError as e:
                # Wait and retry on rate limit
                wait_time = e.retry_after or 60.0
                logger.warning(f"Rate limited on {symbol}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                try:
                    await self._fetch_and_store_daily(symbol)
                    stats.success_count += 1
                except Exception as retry_error:
                    stats.failure_count += 1
                    stats.errors.append(
                        {
                            "symbol": symbol,
                            "error": str(retry_error),
                            "type": "retry_failed",
                        }
                    )

            except InvalidSymbolError:
                # Skip invalid symbols
                stats.skipped_count += 1
                logger.debug(f"Skipping invalid symbol: {symbol}")

            except Exception as e:
                stats.failure_count += 1
                stats.errors.append(
                    {
                        "symbol": symbol,
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )
                logger.error(f"Failed to process {symbol}: {e}")

        stats.end_time = datetime.utcnow()

        # Check for high failure rate and alert
        if stats.failure_rate > self.alert_threshold_pct:
            await self._emit_alert(
                "HIGH_FAILURE_RATE",
                f"Daily update failure rate {stats.failure_rate:.1f}% exceeds threshold",
                stats.to_dict(),
            )

        logger.info(f"Daily update complete: {json.dumps(stats.to_dict())}")
        return stats

    async def _fetch_and_store_daily(self, symbol: str) -> None:
        """Fetch and store daily data for a single symbol."""
        # Fetch daily OHLCV data
        df = await self.av_client.get_daily_ohlcv(symbol, outputsize="compact")

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return

        # Ensure stock exists in database
        metadata = await self.universe.get_symbol_metadata(symbol)
        await self.storage.save_stock(
            {
                "symbol": symbol,
                "name": metadata.name,
                "sector": metadata.sector,
                "industry": metadata.industry,
                "market_cap": metadata.market_cap,
                "avg_volume": metadata.avg_volume,
            }
        )

        # Store daily bars
        await self.storage.save_daily_bars(symbol, df)

    # =========================================================================
    # Historical Backfill
    # =========================================================================

    async def backfill_history(
        self,
        symbols: list[str],
        start_date: date,
        resume: bool = True,
    ) -> PipelineStats:
        """Backfill historical data for specified symbols.

        Fetches full history for each symbol, stores in batches,
        and supports checkpoint/resume on failure.

        Args:
            symbols: List of symbols to backfill.
            start_date: Start date for historical data.
            resume: Whether to resume from checkpoint if available.

        Returns:
            Pipeline statistics with success/failure counts.
        """
        stats = PipelineStats()
        logger.info(f"Starting backfill for {len(symbols)} symbols from {start_date}")

        # Load or create checkpoint
        checkpoint = None
        if resume:
            checkpoint = self._load_checkpoint(symbols, start_date)

        if checkpoint is None:
            checkpoint = BackfillCheckpoint(
                symbols=symbols,
                start_date=start_date,
            )

        remaining = checkpoint.remaining_symbols
        stats.total_symbols = len(symbols)
        stats.success_count = len(checkpoint.completed_symbols)
        stats.failure_count = len(checkpoint.failed_symbols)

        logger.info(
            f"Backfill progress: {checkpoint.progress_pct:.1f}% complete, "
            f"{len(remaining)} remaining"
        )

        # Process remaining symbols
        for i, symbol in enumerate(remaining):
            try:
                await self._fetch_and_store_history(symbol, start_date)
                checkpoint.completed_symbols.append(symbol)
                stats.success_count += 1

                # Save checkpoint periodically
                if (i + 1) % 10 == 0:
                    checkpoint.last_updated = datetime.utcnow()
                    self._save_checkpoint(checkpoint)
                    logger.info(
                        f"Backfill progress: {checkpoint.progress_pct:.1f}% "
                        f"({len(checkpoint.completed_symbols)} complete)"
                    )

            except RateLimitError as e:
                wait_time = e.retry_after or 60.0
                logger.warning(f"Rate limited on {symbol}, waiting {wait_time}s")
                await asyncio.sleep(wait_time)
                try:
                    await self._fetch_and_store_history(symbol, start_date)
                    checkpoint.completed_symbols.append(symbol)
                    stats.success_count += 1
                except Exception as retry_error:
                    checkpoint.failed_symbols.append(symbol)
                    stats.failure_count += 1
                    stats.errors.append(
                        {
                            "symbol": symbol,
                            "error": str(retry_error),
                            "type": "retry_failed",
                        }
                    )

            except InvalidSymbolError:
                stats.skipped_count += 1
                checkpoint.completed_symbols.append(symbol)  # Mark as done
                logger.debug(f"Skipping invalid symbol: {symbol}")

            except Exception as e:
                checkpoint.failed_symbols.append(symbol)
                stats.failure_count += 1
                stats.errors.append(
                    {
                        "symbol": symbol,
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                )
                logger.error(f"Failed to backfill {symbol}: {e}")

                # Save checkpoint on failure
                checkpoint.last_updated = datetime.utcnow()
                self._save_checkpoint(checkpoint)

        stats.end_time = datetime.utcnow()

        # Final checkpoint save
        checkpoint.last_updated = datetime.utcnow()
        self._save_checkpoint(checkpoint)

        # Check for high failure rate
        if stats.failure_rate > self.alert_threshold_pct:
            await self._emit_alert(
                "BACKFILL_HIGH_FAILURE",
                f"Backfill failure rate {stats.failure_rate:.1f}% exceeds threshold",
                stats.to_dict(),
            )

        logger.info(f"Backfill complete: {json.dumps(stats.to_dict())}")
        return stats

    async def _fetch_and_store_history(self, symbol: str, start_date: date) -> None:
        """Fetch and store full history for a single symbol."""
        # Fetch full daily history
        df = await self.av_client.get_daily_ohlcv(symbol, outputsize="full")

        if df.empty:
            logger.warning(f"No historical data for {symbol}")
            return

        # Filter to start date
        df = df[df.index >= pd.Timestamp(start_date)]

        if df.empty:
            logger.debug(f"No data for {symbol} after {start_date}")
            return

        # Ensure stock exists
        metadata = await self.universe.get_symbol_metadata(symbol)
        await self.storage.save_stock(
            {
                "symbol": symbol,
                "name": metadata.name,
                "sector": metadata.sector,
                "industry": metadata.industry,
                "market_cap": metadata.market_cap,
                "avg_volume": metadata.avg_volume,
            }
        )

        # Store in batches (1000 rows at a time)
        batch_size = 1000
        for batch_start in range(0, len(df), batch_size):
            batch_df = df.iloc[batch_start : batch_start + batch_size]
            await self.storage.save_daily_bars(symbol, batch_df)

    def _get_checkpoint_path(self, symbols: list[str], start_date: date) -> Path:
        """Get checkpoint file path for a backfill operation."""
        # Create a deterministic hash from symbols and start date
        symbol_hash = hash(tuple(sorted(symbols))) % (10**8)
        filename = f"backfill_{start_date.isoformat()}_{symbol_hash}.json"
        return self.checkpoint_dir / filename

    def _load_checkpoint(self, symbols: list[str], start_date: date) -> BackfillCheckpoint | None:
        """Load checkpoint if it exists and matches the current operation."""
        path = self._get_checkpoint_path(symbols, start_date)
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
            checkpoint = BackfillCheckpoint.from_dict(data)

            # Verify checkpoint matches current operation
            if set(checkpoint.symbols) != set(symbols):
                logger.warning("Checkpoint symbols don't match, starting fresh")
                return None
            if checkpoint.start_date != start_date:
                logger.warning("Checkpoint start date doesn't match, starting fresh")
                return None

            logger.info(f"Loaded checkpoint: {checkpoint.progress_pct:.1f}% complete")
            return checkpoint

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _save_checkpoint(self, checkpoint: BackfillCheckpoint) -> None:
        """Save checkpoint to file."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._get_checkpoint_path(checkpoint.symbols, checkpoint.start_date)

        try:
            with open(path, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)
            logger.debug(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    # =========================================================================
    # Real-time Updates
    # =========================================================================

    async def start_realtime_updates(
        self,
        symbols: list[str],
        update_interval_seconds: float = 60.0,
    ) -> None:
        """Start real-time updates for market hours.

        Polls for latest data and emits events for the strategy engine.
        For full real-time streaming, integrate with IB TWS API.

        Args:
            symbols: List of symbols to monitor.
            update_interval_seconds: Polling interval in seconds.
        """
        logger.info(f"Starting real-time updates for {len(symbols)} symbols")

        while True:
            try:
                for symbol in symbols:
                    try:
                        # Fetch latest data (compact returns ~100 days)
                        df = await self.av_client.get_daily_ohlcv(symbol, outputsize="compact")

                        if not df.empty:
                            latest = df.iloc[-1]
                            await self._emit_event(
                                "QUOTE_UPDATE",
                                {
                                    "symbol": symbol,
                                    "date": str(df.index[-1].date()),
                                    "open": float(latest["open"]),
                                    "high": float(latest["high"]),
                                    "low": float(latest["low"]),
                                    "close": float(latest["close"]),
                                    "volume": int(latest["volume"]),
                                },
                            )

                    except Exception as e:
                        logger.error(f"Real-time update failed for {symbol}: {e}")

                await asyncio.sleep(update_interval_seconds)

            except asyncio.CancelledError:
                logger.info("Real-time updates cancelled")
                break
            except Exception as e:
                logger.error(f"Real-time update loop error: {e}")
                await asyncio.sleep(update_interval_seconds)

    # =========================================================================
    # S3 Backup (stub for future implementation)
    # =========================================================================

    async def backup_to_s3(
        self,
        symbols: list[str],
        bucket: str,
        prefix: str = "daily_bars",
    ) -> None:
        """Backup daily bars to S3.

        This is a stub for S3 integration. Implement with boto3.

        Args:
            symbols: Symbols to backup.
            bucket: S3 bucket name.
            prefix: S3 key prefix.
        """
        logger.info(f"S3 backup requested for {len(symbols)} symbols to {bucket}/{prefix}")
        # TODO: Implement S3 backup with boto3
        # For each symbol:
        #   - Fetch data from database
        #   - Convert to parquet
        #   - Upload to S3


# =============================================================================
# Lambda Handler for EventBridge
# =============================================================================


async def _async_handler(
    event: dict[str, Any],
    context: Any,  # noqa: ARG001
) -> dict[str, Any]:
    """Async Lambda handler implementation."""
    event_type = event.get("type") or event.get("detail-type", "DAILY_UPDATE")

    # Initialize clients (in production, use environment variables)
    from config.settings import get_settings

    settings = get_settings()

    av_client = AlphaVantageClient(
        api_key=settings.alpha_vantage_api_key,
        calls_per_minute=75,
    )
    storage = StorageManager(str(settings.database_url))
    universe = StockUniverse(av_client)

    pipeline = DataPipeline(av_client, storage, universe)

    try:
        await storage.init_db()

        if event_type == EventType.DAILY_UPDATE.value:
            stats = await pipeline.run_daily_update()
            return {
                "statusCode": 200,
                "body": json.dumps(stats.to_dict()),
            }

        elif event_type == EventType.BACKFILL.value:
            symbols = event.get("symbols", [])
            start_date_str = event.get("start_date")

            if not symbols or not start_date_str:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"error": "Missing required fields: symbols, start_date"}),
                }

            start_date = date.fromisoformat(start_date_str)
            stats = await pipeline.backfill_history(symbols, start_date)
            return {
                "statusCode": 200,
                "body": json.dumps(stats.to_dict()),
            }

        elif event_type == EventType.UNIVERSE_REFRESH.value:
            symbols = await universe.refresh_universe()
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "message": "Universe refreshed",
                        "symbol_count": len(symbols),
                    }
                ),
            }

        else:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Unknown event type: {event_type}"}),
            }

    finally:
        await av_client.close()
        await storage.close()


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda handler for EventBridge events.

    Supports event types:
    - DAILY_UPDATE: Run daily data refresh
    - BACKFILL: Backfill historical data
    - UNIVERSE_REFRESH: Refresh stock universe

    Args:
        event: Lambda event with type and optional parameters.
        context: Lambda context.

    Returns:
        Response with status and results.
    """
    return asyncio.run(_async_handler(event, context))
