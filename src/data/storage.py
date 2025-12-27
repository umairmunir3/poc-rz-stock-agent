"""PostgreSQL storage layer with async SQLAlchemy ORM."""

import logging
from datetime import date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.data.db_models import (
    Base,
    DailyBar,
    PortfolioSnapshot,
    Signal,
    SignalStatus,
    Stock,
    Trade,
    TradeStatus,
)

logger = logging.getLogger(__name__)


class StorageManager:
    """Async database storage manager using SQLAlchemy.

    Provides CRUD operations for stocks, daily bars, signals,
    trades, and portfolio snapshots.
    """

    def __init__(self, database_url: str, echo: bool = False) -> None:
        """Initialize the storage manager.

        Args:
            database_url: PostgreSQL connection URL (postgresql+asyncpg://...).
            echo: Whether to log SQL statements.
        """
        # Convert standard postgres URL to async format if needed
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )

        # SQLite doesn't support pool_size and max_overflow
        if "sqlite" in database_url:
            self.engine = create_async_engine(database_url, echo=echo)
        else:
            self.engine = create_async_engine(
                database_url,
                echo=echo,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized")

    async def drop_db(self) -> None:
        """Drop all database tables. Use with caution!"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.warning("All database tables dropped")

    async def close(self) -> None:
        """Close the database connection."""
        await self.engine.dispose()

    # =========================================================================
    # Stock Operations
    # =========================================================================

    async def save_stock(self, stock_data: dict[str, Any]) -> Stock:
        """Save or update a stock record.

        Args:
            stock_data: Dictionary with stock fields.

        Returns:
            The saved Stock object.
        """
        async with self.async_session() as session:
            # Use upsert for PostgreSQL
            stmt = insert(Stock).values(**stock_data)
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={
                    "name": stmt.excluded.name,
                    "sector": stmt.excluded.sector,
                    "industry": stmt.excluded.industry,
                    "market_cap": stmt.excluded.market_cap,
                    "avg_volume": stmt.excluded.avg_volume,
                    "last_updated": datetime.utcnow(),
                },
            )
            await session.execute(stmt)
            await session.commit()

            # Fetch and return the saved stock
            result = await session.execute(
                select(Stock).where(Stock.symbol == stock_data["symbol"])
            )
            return result.scalar_one()

    async def get_stock(self, symbol: str) -> Stock | None:
        """Get a stock by symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Stock object or None if not found.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Stock).where(Stock.symbol == symbol.upper())
            )
            return result.scalar_one_or_none()

    async def get_all_stocks(self) -> list[Stock]:
        """Get all stocks.

        Returns:
            List of all Stock objects.
        """
        async with self.async_session() as session:
            result = await session.execute(select(Stock).order_by(Stock.symbol))
            return list(result.scalars().all())

    # =========================================================================
    # Daily Bar Operations
    # =========================================================================

    async def save_daily_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """Save daily OHLCV bars for a symbol.

        Args:
            symbol: Stock ticker symbol.
            df: DataFrame with columns: date, open, high, low, close, volume, adjusted_close.

        Returns:
            Number of rows saved.
        """
        if df.empty:
            return 0

        symbol = symbol.upper()

        # Ensure stock exists
        stock = await self.get_stock(symbol)
        if stock is None:
            await self.save_stock({"symbol": symbol, "name": symbol})

        # Prepare records
        records = []
        for idx, row in df.iterrows():
            record_date = idx if isinstance(idx, (date, datetime)) else row.get("date")
            if isinstance(record_date, pd.Timestamp):
                record_date = record_date.to_pydatetime()

            records.append(
                {
                    "symbol": symbol,
                    "date": record_date,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": int(row["volume"]),
                    "adjusted_close": float(row.get("adjusted_close", row["close"])),
                }
            )

        async with self.async_session() as session:
            # Use upsert to handle duplicates
            for record in records:
                stmt = insert(DailyBar).values(**record)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_daily_bar_symbol_date",
                    set_={
                        "open": stmt.excluded.open,
                        "high": stmt.excluded.high,
                        "low": stmt.excluded.low,
                        "close": stmt.excluded.close,
                        "volume": stmt.excluded.volume,
                        "adjusted_close": stmt.excluded.adjusted_close,
                    },
                )
                await session.execute(stmt)

            await session.commit()

        logger.debug(f"Saved {len(records)} daily bars for {symbol}")
        return len(records)

    async def get_daily_bars(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """Get daily bars for a symbol within date range.

        Args:
            symbol: Stock ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            DataFrame with OHLCV data indexed by date.
        """
        async with self.async_session() as session:
            query = select(DailyBar).where(DailyBar.symbol == symbol.upper())

            if start is not None:
                query = query.where(DailyBar.date >= start)
            if end is not None:
                query = query.where(DailyBar.date <= end)

            query = query.order_by(DailyBar.date)

            result = await session.execute(query)
            rows = result.scalars().all()

            if not rows:
                return pd.DataFrame()

            data = [
                {
                    "date": row.date,
                    "open": row.open,
                    "high": row.high,
                    "low": row.low,
                    "close": row.close,
                    "volume": row.volume,
                    "adjusted_close": row.adjusted_close,
                }
                for row in rows
            ]

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            return df

    async def delete_daily_bars(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> int:
        """Delete daily bars for a symbol.

        Args:
            symbol: Stock ticker symbol.
            start: Start date (inclusive).
            end: End date (inclusive).

        Returns:
            Number of rows deleted.
        """
        async with self.async_session() as session:
            query = delete(DailyBar).where(DailyBar.symbol == symbol.upper())

            if start is not None:
                query = query.where(DailyBar.date >= start)
            if end is not None:
                query = query.where(DailyBar.date <= end)

            result = await session.execute(query)
            await session.commit()

            return result.rowcount

    # =========================================================================
    # Signal Operations
    # =========================================================================

    async def save_signal(self, signal_data: dict[str, Any]) -> Signal:
        """Save a new trading signal.

        Args:
            signal_data: Dictionary with signal fields.

        Returns:
            The saved Signal object.
        """
        async with self.async_session() as session:
            signal = Signal(**signal_data)
            session.add(signal)
            await session.commit()
            await session.refresh(signal)
            return signal

    async def get_signal(self, signal_id: int) -> Signal | None:
        """Get a signal by ID.

        Args:
            signal_id: Signal ID.

        Returns:
            Signal object or None if not found.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Signal).where(Signal.id == signal_id)
            )
            return result.scalar_one_or_none()

    async def get_active_signals(self) -> list[Signal]:
        """Get all pending signals.

        Returns:
            List of pending Signal objects.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Signal)
                .where(Signal.status == SignalStatus.PENDING.value)
                .order_by(Signal.created_at.desc())
            )
            return list(result.scalars().all())

    async def get_signals_by_symbol(self, symbol: str) -> list[Signal]:
        """Get all signals for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of Signal objects.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Signal)
                .where(Signal.symbol == symbol.upper())
                .order_by(Signal.created_at.desc())
            )
            return list(result.scalars().all())

    async def update_signal_status(
        self,
        signal_id: int,
        status: SignalStatus,
    ) -> bool:
        """Update signal status.

        Args:
            signal_id: Signal ID.
            status: New status.

        Returns:
            True if updated, False if signal not found.
        """
        async with self.async_session() as session:
            result = await session.execute(
                update(Signal)
                .where(Signal.id == signal_id)
                .values(status=status.value)
            )
            await session.commit()
            return result.rowcount > 0

    async def expire_old_signals(self) -> int:
        """Expire signals past their expiration time.

        Returns:
            Number of signals expired.
        """
        async with self.async_session() as session:
            now = datetime.utcnow()
            result = await session.execute(
                update(Signal)
                .where(Signal.status == SignalStatus.PENDING.value)
                .where(Signal.expires_at <= now)
                .values(status=SignalStatus.EXPIRED.value)
            )
            await session.commit()
            return result.rowcount

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def save_trade(self, trade_data: dict[str, Any]) -> Trade:
        """Save a new trade.

        Args:
            trade_data: Dictionary with trade fields.

        Returns:
            The saved Trade object.
        """
        async with self.async_session() as session:
            trade = Trade(**trade_data)
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade

    async def get_trade(self, trade_id: int) -> Trade | None:
        """Get a trade by ID.

        Args:
            trade_id: Trade ID.

        Returns:
            Trade object or None if not found.
        """
        async with self.async_session() as session:
            result = await session.execute(select(Trade).where(Trade.id == trade_id))
            return result.scalar_one_or_none()

    async def get_open_trades(self) -> list[Trade]:
        """Get all open trades.

        Returns:
            List of open Trade objects.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Trade)
                .where(Trade.status == TradeStatus.OPEN.value)
                .order_by(Trade.entry_time.desc())
            )
            return list(result.scalars().all())

    async def get_trades_by_symbol(self, symbol: str) -> list[Trade]:
        """Get all trades for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of Trade objects.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(Trade)
                .where(Trade.symbol == symbol.upper())
                .order_by(Trade.entry_time.desc())
            )
            return list(result.scalars().all())

    async def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: datetime | None = None,
        status: TradeStatus = TradeStatus.CLOSED,
    ) -> Trade | None:
        """Close an open trade and calculate PnL.

        Args:
            trade_id: Trade ID.
            exit_price: Exit price.
            exit_time: Exit time (defaults to now).
            status: Closing status (CLOSED or STOPPED).

        Returns:
            Updated Trade object or None if not found.
        """
        async with self.async_session() as session:
            result = await session.execute(select(Trade).where(Trade.id == trade_id))
            trade = result.scalar_one_or_none()

            if trade is None:
                return None

            trade.exit_price = exit_price
            trade.exit_time = exit_time or datetime.utcnow()
            trade.status = status.value
            trade.calculate_pnl()

            await session.commit()
            await session.refresh(trade)
            return trade

    async def get_trade_history(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        symbol: str | None = None,
    ) -> list[Trade]:
        """Get closed trades within date range.

        Args:
            start: Start datetime.
            end: End datetime.
            symbol: Optional symbol filter.

        Returns:
            List of closed Trade objects.
        """
        async with self.async_session() as session:
            query = select(Trade).where(Trade.status != TradeStatus.OPEN.value)

            if start is not None:
                query = query.where(Trade.entry_time >= start)
            if end is not None:
                query = query.where(Trade.entry_time <= end)
            if symbol is not None:
                query = query.where(Trade.symbol == symbol.upper())

            query = query.order_by(Trade.entry_time.desc())

            result = await session.execute(query)
            return list(result.scalars().all())

    # =========================================================================
    # Portfolio Snapshot Operations
    # =========================================================================

    async def save_portfolio_snapshot(
        self,
        snapshot_data: dict[str, Any],
    ) -> PortfolioSnapshot:
        """Save a portfolio snapshot.

        Args:
            snapshot_data: Dictionary with snapshot fields.

        Returns:
            The saved PortfolioSnapshot object.
        """
        async with self.async_session() as session:
            snapshot = PortfolioSnapshot(**snapshot_data)
            session.add(snapshot)
            await session.commit()
            await session.refresh(snapshot)
            return snapshot

    async def get_latest_snapshot(self) -> PortfolioSnapshot | None:
        """Get the most recent portfolio snapshot.

        Returns:
            Latest PortfolioSnapshot or None if no snapshots.
        """
        async with self.async_session() as session:
            result = await session.execute(
                select(PortfolioSnapshot).order_by(
                    PortfolioSnapshot.timestamp.desc()
                ).limit(1)
            )
            return result.scalar_one_or_none()

    async def get_snapshots(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[PortfolioSnapshot]:
        """Get portfolio snapshots within date range.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            List of PortfolioSnapshot objects.
        """
        async with self.async_session() as session:
            query = select(PortfolioSnapshot)

            if start is not None:
                query = query.where(PortfolioSnapshot.timestamp >= start)
            if end is not None:
                query = query.where(PortfolioSnapshot.timestamp <= end)

            query = query.order_by(PortfolioSnapshot.timestamp)

            result = await session.execute(query)
            return list(result.scalars().all())

    async def get_equity_curve(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Get equity curve as DataFrame.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            DataFrame with equity curve data.
        """
        snapshots = await self.get_snapshots(start, end)

        if not snapshots:
            return pd.DataFrame()

        data = [
            {
                "timestamp": s.timestamp,
                "equity": s.equity,
                "cash": s.cash,
                "open_positions": s.open_positions,
                "daily_pnl": s.daily_pnl,
                "drawdown_pct": s.drawdown_pct,
            }
            for s in snapshots
        ]

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        return df

    # =========================================================================
    # Statistics and Analytics
    # =========================================================================

    async def get_trade_statistics(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, Any]:
        """Calculate trade statistics.

        Args:
            start: Start datetime.
            end: End datetime.

        Returns:
            Dictionary with trade statistics.
        """
        trades = await self.get_trade_history(start, end)

        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "avg_r_multiple": 0.0,
            }

        winners = [t for t in trades if t.pnl is not None and t.pnl > 0]
        losers = [t for t in trades if t.pnl is not None and t.pnl < 0]

        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))

        r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]

        return {
            "total_trades": len(trades),
            "winning_trades": len(winners),
            "losing_trades": len(losers),
            "win_rate": len(winners) / len(trades) if trades else 0.0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(trades) if trades else 0.0,
            "avg_win": gross_profit / len(winners) if winners else 0.0,
            "avg_loss": gross_loss / len(losers) if losers else 0.0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else 0.0,
            "avg_r_multiple": sum(r_multiples) / len(r_multiples) if r_multiples else 0.0,
        }
