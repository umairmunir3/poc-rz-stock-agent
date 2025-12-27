"""Unit tests for database storage layer."""

from datetime import date, datetime

import pandas as pd
import pytest

from src.data.db_models import (
    DailyBar,
    PortfolioSnapshot,
    Signal,
    SignalDirection,
    SignalStatus,
    Stock,
    Trade,
    TradeStatus,
)
from src.data.storage import StorageManager

# Use SQLite for testing (in-memory)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


class TestDatabaseModels:
    """Tests for SQLAlchemy ORM models."""

    def test_stock_model(self) -> None:
        """Test Stock model creation."""
        stock = Stock(
            symbol="AAPL",
            name="Apple Inc",
            sector="Technology",
            industry="Consumer Electronics",
            market_cap=3000000000000,
            avg_volume=50000000,
        )

        assert stock.symbol == "AAPL"
        assert stock.name == "Apple Inc"
        assert stock.sector == "Technology"
        assert repr(stock) == "<Stock(symbol='AAPL', name='Apple Inc')>"

    def test_daily_bar_model(self) -> None:
        """Test DailyBar model creation."""
        bar = DailyBar(
            symbol="AAPL",
            date=date(2024, 1, 15),
            open=185.0,
            high=188.0,
            low=184.5,
            close=187.5,
            volume=50000000,
            adjusted_close=187.5,
        )

        assert bar.symbol == "AAPL"
        assert bar.close == 187.5
        assert bar.volume == 50000000

    def test_signal_model(self) -> None:
        """Test Signal model creation."""
        signal = Signal(
            symbol="AAPL",
            strategy="momentum",
            direction=SignalDirection.LONG.value,
            score=85,
            entry_price=185.0,
            stop_price=180.0,
            target_price=195.0,
            position_size=10000,
            status=SignalStatus.PENDING.value,
        )

        assert signal.symbol == "AAPL"
        assert signal.direction == "LONG"
        assert signal.score == 85
        assert signal.status == SignalStatus.PENDING.value

    def test_signal_risk_reward_ratio_long(self) -> None:
        """Test risk/reward calculation for long signal."""
        signal = Signal(
            symbol="AAPL",
            strategy="momentum",
            direction=SignalDirection.LONG.value,
            score=80,
            entry_price=100.0,
            stop_price=95.0,  # Risk = $5
            target_price=115.0,  # Reward = $15
            position_size=1000,
        )

        assert signal.risk_reward_ratio == 3.0  # 15/5 = 3:1

    def test_signal_risk_reward_ratio_short(self) -> None:
        """Test risk/reward calculation for short signal."""
        signal = Signal(
            symbol="AAPL",
            strategy="mean_reversion",
            direction=SignalDirection.SHORT.value,
            score=75,
            entry_price=100.0,
            stop_price=105.0,  # Risk = $5
            target_price=90.0,  # Reward = $10
            position_size=1000,
        )

        assert signal.risk_reward_ratio == 2.0  # 10/5 = 2:1

    def test_trade_model(self) -> None:
        """Test Trade model creation."""
        trade = Trade(
            symbol="AAPL",
            direction=SignalDirection.LONG.value,
            entry_price=185.0,
            entry_time=datetime(2024, 1, 15, 10, 30),
            quantity=100,
            stop_price=180.0,
            status=TradeStatus.OPEN.value,
        )

        assert trade.symbol == "AAPL"
        assert trade.quantity == 100
        assert trade.status == TradeStatus.OPEN.value
        assert trade.pnl is None

    def test_trade_calculate_pnl_long(self) -> None:
        """Test PnL calculation for long trade."""
        trade = Trade(
            symbol="AAPL",
            direction=SignalDirection.LONG.value,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 15, 10, 30),
            quantity=100,
            stop_price=95.0,
        )

        trade.exit_price = 110.0
        trade.calculate_pnl()

        assert trade.pnl == 1000.0  # (110 - 100) * 100
        assert trade.pnl_pct == 0.10  # 10%
        assert trade.r_multiple == 2.0  # $10 gain / $5 risk

    def test_trade_calculate_pnl_short(self) -> None:
        """Test PnL calculation for short trade."""
        trade = Trade(
            symbol="AAPL",
            direction=SignalDirection.SHORT.value,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 15, 10, 30),
            quantity=100,
            stop_price=105.0,
        )

        trade.exit_price = 90.0
        trade.calculate_pnl()

        assert trade.pnl == 1000.0  # (100 - 90) * 100
        assert trade.pnl_pct == 0.10  # 10%
        assert trade.r_multiple == 2.0  # $10 gain / $5 risk

    def test_trade_calculate_pnl_loss(self) -> None:
        """Test PnL calculation for losing trade."""
        trade = Trade(
            symbol="AAPL",
            direction=SignalDirection.LONG.value,
            entry_price=100.0,
            entry_time=datetime(2024, 1, 15, 10, 30),
            quantity=100,
            stop_price=95.0,
        )

        trade.exit_price = 95.0  # Stopped out
        trade.calculate_pnl()

        assert trade.pnl == -500.0  # (95 - 100) * 100
        assert trade.pnl_pct == -0.05  # -5%
        assert trade.r_multiple == -1.0  # -$5 loss / $5 risk

    def test_portfolio_snapshot_model(self) -> None:
        """Test PortfolioSnapshot model creation."""
        snapshot = PortfolioSnapshot(
            equity=100000.0,
            cash=50000.0,
            open_positions=3,
            daily_pnl=500.0,
            drawdown_pct=0.02,
            high_water_mark=102000.0,
        )

        assert snapshot.equity == 100000.0
        assert snapshot.open_positions == 3
        assert snapshot.drawdown_pct == 0.02


@pytest.fixture
async def storage():
    """Create a test storage manager with in-memory SQLite."""
    mgr = StorageManager(TEST_DATABASE_URL)
    await mgr.init_db()
    yield mgr
    await mgr.close()


class TestStorageManager:
    """Tests for StorageManager class."""

    @pytest.mark.asyncio
    async def test_init_db(self, storage: StorageManager) -> None:
        """Test database initialization."""
        # Tables should be created
        # Just verify we can query without error
        stocks = await storage.get_all_stocks()
        assert stocks == []

    @pytest.mark.asyncio
    async def test_save_and_get_stock(self, storage: StorageManager) -> None:
        """Test saving and retrieving a stock."""
        stock_data = {
            "symbol": "AAPL",
            "name": "Apple Inc",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000,
            "avg_volume": 50000000,
        }

        saved = await storage.save_stock(stock_data)
        assert saved.symbol == "AAPL"
        assert saved.name == "Apple Inc"

        retrieved = await storage.get_stock("AAPL")
        assert retrieved is not None
        assert retrieved.symbol == "AAPL"
        assert retrieved.market_cap == 3000000000000

    @pytest.mark.asyncio
    async def test_get_nonexistent_stock(self, storage: StorageManager) -> None:
        """Test getting a stock that doesn't exist."""
        stock = await storage.get_stock("INVALID")
        assert stock is None

    @pytest.mark.asyncio
    async def test_get_all_stocks(self, storage: StorageManager) -> None:
        """Test getting all stocks."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})
        await storage.save_stock({"symbol": "MSFT", "name": "Microsoft"})
        await storage.save_stock({"symbol": "GOOGL", "name": "Alphabet"})

        stocks = await storage.get_all_stocks()
        assert len(stocks) == 3
        symbols = [s.symbol for s in stocks]
        assert "AAPL" in symbols
        assert "MSFT" in symbols

    @pytest.mark.asyncio
    async def test_save_daily_bars(self, storage: StorageManager) -> None:
        """Test saving daily OHLCV bars."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "open": [185.0, 186.0, 187.0],
                "high": [188.0, 189.0, 190.0],
                "low": [184.0, 185.0, 186.0],
                "close": [187.0, 188.0, 189.0],
                "volume": [50000000, 55000000, 45000000],
                "adjusted_close": [187.0, 188.0, 189.0],
            },
            index=pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"]),
        )

        count = await storage.save_daily_bars("AAPL", df)
        assert count == 3

    @pytest.mark.asyncio
    async def test_get_daily_bars(self, storage: StorageManager) -> None:
        """Test retrieving daily bars with date range."""
        # Save test data
        df = pd.DataFrame(
            {
                "open": [185.0, 186.0, 187.0, 188.0, 189.0],
                "high": [188.0, 189.0, 190.0, 191.0, 192.0],
                "low": [184.0, 185.0, 186.0, 187.0, 188.0],
                "close": [187.0, 188.0, 189.0, 190.0, 191.0],
                "volume": [50000000, 55000000, 45000000, 60000000, 52000000],
                "adjusted_close": [187.0, 188.0, 189.0, 190.0, 191.0],
            },
            index=pd.to_datetime(
                ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"]
            ),
        )
        await storage.save_daily_bars("AAPL", df)

        # Get all bars
        result = await storage.get_daily_bars("AAPL")
        assert len(result) == 5

        # Get bars with date range (note: dates stored as datetime, so end is exclusive for same day)
        result = await storage.get_daily_bars(
            "AAPL",
            start=date(2024, 1, 16),
            end=date(2024, 1, 19),  # Include 18th
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_save_and_get_signal(self, storage: StorageManager) -> None:
        """Test saving and retrieving signals."""
        # First create stock
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        signal_data = {
            "symbol": "AAPL",
            "strategy": "momentum",
            "direction": SignalDirection.LONG.value,
            "score": 85,
            "entry_price": 185.0,
            "stop_price": 180.0,
            "target_price": 195.0,
            "position_size": 10000,
        }

        saved = await storage.save_signal(signal_data)
        assert saved.id is not None
        assert saved.symbol == "AAPL"
        assert saved.status == SignalStatus.PENDING.value

        retrieved = await storage.get_signal(saved.id)
        assert retrieved is not None
        assert retrieved.score == 85

    @pytest.mark.asyncio
    async def test_get_active_signals(self, storage: StorageManager) -> None:
        """Test getting pending signals."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        # Create multiple signals
        await storage.save_signal(
            {
                "symbol": "AAPL",
                "strategy": "momentum",
                "direction": "LONG",
                "score": 80,
                "entry_price": 185.0,
                "stop_price": 180.0,
                "target_price": 195.0,
                "position_size": 10000,
                "status": SignalStatus.PENDING.value,
            }
        )
        await storage.save_signal(
            {
                "symbol": "AAPL",
                "strategy": "breakout",
                "direction": "LONG",
                "score": 75,
                "entry_price": 190.0,
                "stop_price": 185.0,
                "target_price": 200.0,
                "position_size": 8000,
                "status": SignalStatus.FILLED.value,  # Not pending
            }
        )

        active = await storage.get_active_signals()
        assert len(active) == 1
        assert active[0].strategy == "momentum"

    @pytest.mark.asyncio
    async def test_update_signal_status(self, storage: StorageManager) -> None:
        """Test updating signal status."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        signal = await storage.save_signal(
            {
                "symbol": "AAPL",
                "strategy": "momentum",
                "direction": "LONG",
                "score": 80,
                "entry_price": 185.0,
                "stop_price": 180.0,
                "target_price": 195.0,
                "position_size": 10000,
            }
        )

        result = await storage.update_signal_status(signal.id, SignalStatus.FILLED)
        assert result is True

        updated = await storage.get_signal(signal.id)
        assert updated.status == SignalStatus.FILLED.value

    @pytest.mark.asyncio
    async def test_save_and_get_trade(self, storage: StorageManager) -> None:
        """Test saving and retrieving trades."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        trade_data = {
            "symbol": "AAPL",
            "direction": SignalDirection.LONG.value,
            "entry_price": 185.0,
            "entry_time": datetime(2024, 1, 15, 10, 30),
            "quantity": 100,
            "stop_price": 180.0,
            "target_price": 195.0,
        }

        saved = await storage.save_trade(trade_data)
        assert saved.id is not None
        assert saved.status == TradeStatus.OPEN.value

        retrieved = await storage.get_trade(saved.id)
        assert retrieved is not None
        assert retrieved.quantity == 100

    @pytest.mark.asyncio
    async def test_get_open_trades(self, storage: StorageManager) -> None:
        """Test getting open trades."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        # Create open and closed trades
        await storage.save_trade(
            {
                "symbol": "AAPL",
                "direction": "LONG",
                "entry_price": 185.0,
                "entry_time": datetime(2024, 1, 15, 10, 30),
                "quantity": 100,
                "status": TradeStatus.OPEN.value,
            }
        )
        await storage.save_trade(
            {
                "symbol": "AAPL",
                "direction": "LONG",
                "entry_price": 180.0,
                "entry_time": datetime(2024, 1, 14, 10, 30),
                "quantity": 50,
                "status": TradeStatus.CLOSED.value,
            }
        )

        open_trades = await storage.get_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].entry_price == 185.0

    @pytest.mark.asyncio
    async def test_close_trade(self, storage: StorageManager) -> None:
        """Test closing a trade and calculating PnL."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        trade = await storage.save_trade(
            {
                "symbol": "AAPL",
                "direction": SignalDirection.LONG.value,
                "entry_price": 100.0,
                "entry_time": datetime(2024, 1, 15, 10, 30),
                "quantity": 100,
                "stop_price": 95.0,
            }
        )

        closed = await storage.close_trade(trade.id, exit_price=110.0)

        assert closed is not None
        assert closed.status == TradeStatus.CLOSED.value
        assert closed.exit_price == 110.0
        assert closed.pnl == 1000.0
        assert closed.r_multiple == 2.0

    @pytest.mark.asyncio
    async def test_save_portfolio_snapshot(self, storage: StorageManager) -> None:
        """Test saving portfolio snapshots."""
        snapshot_data = {
            "equity": 100000.0,
            "cash": 50000.0,
            "open_positions": 3,
            "daily_pnl": 500.0,
            "total_pnl": 5000.0,
            "drawdown_pct": 0.02,
            "high_water_mark": 102000.0,
        }

        saved = await storage.save_portfolio_snapshot(snapshot_data)
        assert saved.id is not None
        assert saved.equity == 100000.0

    @pytest.mark.asyncio
    async def test_get_latest_snapshot(self, storage: StorageManager) -> None:
        """Test getting the most recent snapshot."""
        # Save multiple snapshots
        await storage.save_portfolio_snapshot(
            {
                "equity": 100000.0,
                "cash": 50000.0,
                "open_positions": 3,
                "daily_pnl": 500.0,
                "drawdown_pct": 0.02,
                "high_water_mark": 100000.0,
                "timestamp": datetime(2024, 1, 15, 16, 0),
            }
        )
        await storage.save_portfolio_snapshot(
            {
                "equity": 101000.0,
                "cash": 48000.0,
                "open_positions": 4,
                "daily_pnl": 1000.0,
                "drawdown_pct": 0.01,
                "high_water_mark": 101000.0,
                "timestamp": datetime(2024, 1, 16, 16, 0),
            }
        )

        latest = await storage.get_latest_snapshot()
        assert latest is not None
        assert latest.equity == 101000.0

    @pytest.mark.asyncio
    async def test_get_equity_curve(self, storage: StorageManager) -> None:
        """Test getting equity curve as DataFrame."""
        # Save snapshots
        for i in range(5):
            await storage.save_portfolio_snapshot(
                {
                    "equity": 100000.0 + i * 1000,
                    "cash": 50000.0,
                    "open_positions": i,
                    "daily_pnl": 1000.0,
                    "drawdown_pct": 0.01,
                    "high_water_mark": 100000.0 + i * 1000,
                    "timestamp": datetime(2024, 1, 15 + i, 16, 0),
                }
            )

        curve = await storage.get_equity_curve()
        assert len(curve) == 5
        assert "equity" in curve.columns
        assert "drawdown_pct" in curve.columns

    @pytest.mark.asyncio
    async def test_get_trade_statistics(self, storage: StorageManager) -> None:
        """Test calculating trade statistics."""
        await storage.save_stock({"symbol": "AAPL", "name": "Apple"})

        # Create some closed trades
        trades_data = [
            {"pnl": 500.0, "r_multiple": 2.0},  # Winner
            {"pnl": 300.0, "r_multiple": 1.5},  # Winner
            {"pnl": -200.0, "r_multiple": -1.0},  # Loser
            {"pnl": 400.0, "r_multiple": 2.0},  # Winner
            {"pnl": -150.0, "r_multiple": -0.75},  # Loser
        ]

        for i, td in enumerate(trades_data):
            await storage.save_trade(
                {
                    "symbol": "AAPL",
                    "direction": "LONG",
                    "entry_price": 100.0,
                    "entry_time": datetime(2024, 1, 15 + i, 10, 30),
                    "exit_price": 100.0 + (td["pnl"] / 100),
                    "exit_time": datetime(2024, 1, 15 + i, 14, 30),
                    "quantity": 100,
                    "pnl": td["pnl"],
                    "r_multiple": td["r_multiple"],
                    "status": TradeStatus.CLOSED.value,
                }
            )

        stats = await storage.get_trade_statistics()

        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] == 0.6
        assert stats["total_pnl"] == 850.0
        assert stats["profit_factor"] > 0

    @pytest.mark.asyncio
    async def test_empty_trade_statistics(self, storage: StorageManager) -> None:
        """Test trade statistics with no trades."""
        stats = await storage.get_trade_statistics()

        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["total_pnl"] == 0.0
