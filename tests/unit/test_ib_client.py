"""Unit tests for Interactive Brokers Client."""

from __future__ import annotations

import asyncio
import sys
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.data.ib_client import (
    AccountSummary,
    IBClient,
    IBClientConfig,
    IBConnectionError,
    IBDataError,
    OptionQuote,
    OptionsChain,
    Order,
    Position,
    Quote,
)


# Create mock ib_insync module for testing
def _create_mock_ib_insync() -> MagicMock:
    """Create a mock ib_insync module."""
    mock_module = MagicMock()
    mock_module.IB = MagicMock()
    mock_module.Stock = MagicMock()
    mock_module.Option = MagicMock()
    mock_module.Contract = MagicMock()
    return mock_module


# Add mock to sys.modules if not already present
if "ib_insync" not in sys.modules:
    sys.modules["ib_insync"] = _create_mock_ib_insync()


class TestDataclasses:
    """Tests for IB client dataclasses."""

    def test_quote_creation(self) -> None:
        """Test Quote dataclass."""
        quote = Quote(
            symbol="AAPL",
            bid=150.00,
            ask=150.05,
            last=150.02,
            volume=1000000,
            timestamp=datetime.now(),
        )
        assert quote.symbol == "AAPL"
        assert quote.bid == 150.00
        assert quote.ask == 150.05

    def test_option_quote_creation(self) -> None:
        """Test OptionQuote dataclass."""
        quote = OptionQuote(
            symbol="AAPL",
            strike=150.0,
            expiry=date.today(),
            right="C",
            bid=5.00,
            ask=5.10,
            last=5.05,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.25,
            delta=0.55,
            gamma=0.02,
            theta=-0.05,
            vega=0.10,
        )
        assert quote.symbol == "AAPL"
        assert quote.strike == 150.0
        assert quote.delta == 0.55

    def test_options_chain_creation(self) -> None:
        """Test OptionsChain dataclass."""
        chain = OptionsChain(
            symbol="AAPL",
            expiry=date.today(),
            calls=[],
            puts=[],
            underlying_price=150.0,
        )
        assert chain.symbol == "AAPL"
        assert chain.underlying_price == 150.0

    def test_position_creation(self) -> None:
        """Test Position dataclass."""
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_cost=145.00,
            market_value=15000.0,
            unrealized_pnl=500.0,
            realized_pnl=0.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.quantity == 100

    def test_order_creation(self) -> None:
        """Test Order dataclass."""
        order = Order(
            order_id=123,
            symbol="AAPL",
            action="BUY",
            order_type="LMT",
            quantity=100,
            limit_price=150.0,
            status="Submitted",
            filled=0,
            remaining=100,
        )
        assert order.order_id == 123
        assert order.action == "BUY"

    def test_account_summary_creation(self) -> None:
        """Test AccountSummary dataclass."""
        summary = AccountSummary(
            account_id="DU12345",
            equity=100000.0,
            cash=50000.0,
            buying_power=100000.0,
            net_liquidation=100000.0,
            unrealized_pnl=500.0,
            realized_pnl=200.0,
            is_paper=True,
        )
        assert summary.account_id == "DU12345"
        assert summary.is_paper is True

    def test_config_defaults(self) -> None:
        """Test IBClientConfig defaults."""
        config = IBClientConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 7497
        assert config.client_id == 1
        assert config.readonly is True


class TestIBClientInit:
    """Tests for IBClient initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        client = IBClient()
        assert client.config.host == "127.0.0.1"
        assert client.config.port == 7497
        assert client.config.client_id == 1

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        client = IBClient(host="192.168.1.100", port=7496, client_id=5)
        assert client.config.host == "192.168.1.100"
        assert client.config.port == 7496
        assert client.config.client_id == 5

    def test_init_with_config(self) -> None:
        """Test initialization with config object."""
        config = IBClientConfig(host="localhost", port=4001, client_id=10)
        client = IBClient(config=config)
        assert client.config.host == "localhost"
        assert client.config.port == 4001


class TestConnection:
    """Tests for connection management."""

    @pytest.mark.asyncio
    async def test_connect_success(self) -> None:
        """Test successful connection."""
        client = IBClient()

        with patch("ib_insync.IB") as mock_ib_class:
            mock_ib = MagicMock()
            mock_ib.connectAsync = AsyncMock()
            mock_ib.isConnected.return_value = True
            mock_ib.disconnectedEvent = MagicMock()
            mock_ib_class.return_value = mock_ib

            result = await client.connect()

            assert result is True
            mock_ib.connectAsync.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_raises(self) -> None:
        """Test connection failure raises exception."""
        client = IBClient()

        with patch("ib_insync.IB") as mock_ib_class:
            mock_ib = MagicMock()
            mock_ib.connectAsync = AsyncMock(side_effect=Exception("Connection refused"))
            mock_ib_class.return_value = mock_ib

            with pytest.raises(IBConnectionError, match="Failed to connect"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connect_timeout_raises(self) -> None:
        """Test connection timeout raises exception."""
        client = IBClient()
        client.config.timeout = 1

        with patch("ib_insync.IB") as mock_ib_class:
            mock_ib = MagicMock()

            async def slow_connect(*args: object, **kwargs: object) -> None:
                await asyncio.sleep(10)

            mock_ib.connectAsync = slow_connect
            mock_ib_class.return_value = mock_ib

            with pytest.raises(IBConnectionError, match="timeout"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Test disconnection."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()
        client._ib.disconnectedEvent = MagicMock()
        client._ib.disconnect = MagicMock()

        await client.disconnect()

        assert client._connected is False
        client._ib.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_is_connected_true(self) -> None:
        """Test is_connected when connected."""
        client = IBClient()
        client._ib = MagicMock()
        client._ib.isConnected.return_value = True

        result = await client.is_connected()
        assert result is True

    @pytest.mark.asyncio
    async def test_is_connected_false_no_ib(self) -> None:
        """Test is_connected when IB is None."""
        client = IBClient()
        client._ib = None

        result = await client.is_connected()
        assert result is False


class TestReconnection:
    """Tests for auto-reconnection."""

    @pytest.mark.asyncio
    async def test_reconnect_on_disconnect(self) -> None:
        """Test reconnection after disconnect."""
        client = IBClient()
        client.config.auto_reconnect = True

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True

            # Simulate reconnection
            result = await client._reconnect(max_attempts=1, delay=0.1)

            assert result is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts(self) -> None:
        """Test reconnection stops after max attempts."""
        client = IBClient()

        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = IBConnectionError("Failed")

            result = await client._reconnect(max_attempts=2, delay=0.1)

            assert result is False
            assert mock_connect.call_count == 2


class TestMarketData:
    """Tests for market data methods."""

    @pytest.mark.asyncio
    async def test_get_quote_returns_data(self) -> None:
        """Test get_quote returns valid quote."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        # Mock the Stock and ticker
        mock_ticker = MagicMock()
        mock_ticker.bid = 150.00
        mock_ticker.ask = 150.05
        mock_ticker.last = 150.02
        mock_ticker.volume = 1000000

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqMktData = MagicMock(return_value=mock_ticker)
        client._ib.cancelMktData = MagicMock()

        with patch("ib_insync.Stock"):
            quote = await client.get_quote("AAPL")

        assert quote.symbol == "AAPL"
        assert quote.bid == 150.00
        assert quote.ask == 150.05
        assert quote.last == 150.02

    @pytest.mark.asyncio
    async def test_get_quote_not_connected_raises(self) -> None:
        """Test get_quote raises when not connected."""
        client = IBClient()
        client._connected = False

        with pytest.raises(IBConnectionError, match="Not connected"):
            await client.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_subscribe_quotes_streams(self) -> None:
        """Test subscribe_quotes sets up streaming."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        mock_ticker = MagicMock()
        mock_ticker.updateEvent = MagicMock()

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqMktData = MagicMock(return_value=mock_ticker)

        callback = MagicMock()

        with patch("ib_insync.Stock"):
            await client.subscribe_quotes(["AAPL", "MSFT"], callback)

        assert len(client._subscriptions) == 2
        assert "AAPL" in client._subscriptions
        assert "MSFT" in client._subscriptions

    @pytest.mark.asyncio
    async def test_get_bars_correct_format(self) -> None:
        """Test get_bars returns correct DataFrame format."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        # Mock historical bars
        mock_bar = MagicMock()
        mock_bar.date = datetime.now()
        mock_bar.open = 150.0
        mock_bar.high = 151.0
        mock_bar.low = 149.0
        mock_bar.close = 150.5
        mock_bar.volume = 1000

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqHistoricalData = MagicMock(return_value=[mock_bar])

        with patch("ib_insync.Stock"):
            df = await client.get_bars("AAPL", "1 D", "1 hour")

        assert isinstance(df, pd.DataFrame)
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    @pytest.mark.asyncio
    async def test_get_bars_empty_returns_empty_df(self) -> None:
        """Test get_bars returns empty DataFrame when no data."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqHistoricalData = MagicMock(return_value=[])

        with patch("ib_insync.Stock"):
            df = await client.get_bars("AAPL")

        assert df.empty


class TestOptionsData:
    """Tests for options data methods."""

    @pytest.mark.asyncio
    async def test_options_chain_parsing(self) -> None:
        """Test options chain parsing."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        # Mock chain data
        mock_chain = MagicMock()
        mock_chain.expirations = ["20250117"]
        mock_chain.strikes = [145, 150, 155]

        mock_ticker = MagicMock()
        mock_ticker.last = 150.0
        mock_ticker.close = 150.0

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqMktData = MagicMock(return_value=mock_ticker)
        client._ib.cancelMktData = MagicMock()
        client._ib.reqSecDefOptParams = MagicMock(return_value=[mock_chain])

        # Mock get_option_quote
        with patch.object(client, "get_option_quote", new_callable=AsyncMock) as mock_quote:
            mock_quote.return_value = OptionQuote(
                symbol="AAPL",
                strike=150.0,
                expiry=date(2025, 1, 17),
                right="C",
                bid=5.0,
                ask=5.1,
                last=5.05,
                volume=100,
                open_interest=1000,
                implied_volatility=0.25,
                delta=0.55,
                gamma=0.02,
                theta=-0.05,
                vega=0.10,
            )

            with patch("ib_insync.Stock"):
                chain = await client.get_options_chain("AAPL")

        assert chain.symbol == "AAPL"
        assert chain.underlying_price == 150.0

    @pytest.mark.asyncio
    async def test_get_option_quote(self) -> None:
        """Test getting single option quote."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        mock_greeks = MagicMock()
        mock_greeks.delta = 0.55
        mock_greeks.gamma = 0.02
        mock_greeks.theta = -0.05
        mock_greeks.vega = 0.10
        mock_greeks.impliedVol = 0.25

        mock_ticker = MagicMock()
        mock_ticker.bid = 5.0
        mock_ticker.ask = 5.1
        mock_ticker.last = 5.05
        mock_ticker.volume = 100
        mock_ticker.modelGreeks = mock_greeks
        mock_ticker.lastGreeks = None

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqMktData = MagicMock(return_value=mock_ticker)
        client._ib.cancelMktData = MagicMock()

        with patch("ib_insync.Option"):
            quote = await client.get_option_quote("AAPL", 150.0, date(2025, 1, 17), "C")

        assert quote.symbol == "AAPL"
        assert quote.strike == 150.0
        assert quote.delta == 0.55


class TestAccountData:
    """Tests for account data methods."""

    @pytest.mark.asyncio
    async def test_get_positions(self) -> None:
        """Test getting account positions."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        mock_position = MagicMock()
        mock_position.contract.symbol = "AAPL"
        mock_position.position = 100
        mock_position.avgCost = 145.0

        client._ib.positions = MagicMock(return_value=[mock_position])

        positions = await client.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 100

    @pytest.mark.asyncio
    async def test_get_open_orders(self) -> None:
        """Test getting open orders."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        mock_trade = MagicMock()
        mock_trade.order.orderId = 123
        mock_trade.contract.symbol = "AAPL"
        mock_trade.order.action = "BUY"
        mock_trade.order.orderType = "LMT"
        mock_trade.order.totalQuantity = 100
        mock_trade.order.lmtPrice = 150.0
        mock_trade.orderStatus.status = "Submitted"
        mock_trade.orderStatus.filled = 0
        mock_trade.orderStatus.remaining = 100

        client._ib.openTrades = MagicMock(return_value=[mock_trade])

        orders = await client.get_open_orders()

        assert len(orders) == 1
        assert orders[0].order_id == 123
        assert orders[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_get_account_summary(self) -> None:
        """Test getting account summary."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        # Mock account values
        mock_values = [
            MagicMock(account="DU12345", tag="NetLiquidation", value="100000"),
            MagicMock(account="DU12345", tag="TotalCashValue", value="50000"),
            MagicMock(account="DU12345", tag="BuyingPower", value="100000"),
            MagicMock(account="DU12345", tag="EquityWithLoanValue", value="100000"),
            MagicMock(account="DU12345", tag="UnrealizedPnL", value="500"),
            MagicMock(account="DU12345", tag="RealizedPnL", value="200"),
        ]

        client._ib.accountSummary = MagicMock(return_value=mock_values)

        summary = await client.get_account_summary()

        assert summary.account_id == "DU12345"
        assert summary.net_liquidation == 100000.0
        assert summary.cash == 50000.0
        assert summary.is_paper is True  # Starts with "D"


class TestPacing:
    """Tests for request pacing."""

    @pytest.mark.asyncio
    async def test_respects_request_pacing(self) -> None:
        """Test that pacing delays requests appropriately."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()
        client.config.max_requests_per_second = 10.0  # 100ms between requests

        mock_ticker = MagicMock()
        mock_ticker.bid = 150.0
        mock_ticker.ask = 150.05
        mock_ticker.last = 150.02
        mock_ticker.volume = 1000

        client._ib.qualifyContracts = MagicMock()
        client._ib.reqMktData = MagicMock(return_value=mock_ticker)
        client._ib.cancelMktData = MagicMock()

        import time

        with patch("ib_insync.Stock"):
            start = time.time()
            await client.get_quote("AAPL")
            await client.get_quote("MSFT")
            elapsed = time.time() - start

        # Should have some delay due to pacing
        assert elapsed >= 0.05  # At least some pacing delay


class TestMarketStatus:
    """Tests for market status checks."""

    @pytest.mark.asyncio
    async def test_handles_market_closed(self) -> None:
        """Test handling when market is closed."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        client._ib.reqManagedAccts = MagicMock()
        client._ib.managedAccounts = MagicMock(return_value=["DU12345"])

        status = await client.check_market_status()

        assert "connected" in status
        assert status["connected"] is True

    @pytest.mark.asyncio
    async def test_is_paper_account_detection(self) -> None:
        """Test paper account detection."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()
        client._ib.managedAccounts = MagicMock(return_value=["DU12345"])

        is_paper = await client._is_paper_account()
        assert is_paper is True

        client._ib.managedAccounts = MagicMock(return_value=["U12345"])
        is_paper = await client._is_paper_account()
        assert is_paper is False


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_quote_error_raises_ib_data_error(self) -> None:
        """Test that data errors are properly wrapped."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        client._ib.qualifyContracts = MagicMock(side_effect=Exception("Symbol not found"))

        with patch("ib_insync.Stock"):
            with pytest.raises(IBDataError, match="Failed to get quote"):
                await client.get_quote("INVALID")

    @pytest.mark.asyncio
    async def test_get_positions_error(self) -> None:
        """Test position retrieval error handling."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        client._ib.positions = MagicMock(side_effect=Exception("Connection lost"))

        with pytest.raises(IBDataError, match="Failed to get positions"):
            await client.get_positions()


class TestUnsubscribe:
    """Tests for unsubscribe functionality."""

    @pytest.mark.asyncio
    async def test_unsubscribe_specific_symbols(self) -> None:
        """Test unsubscribing from specific symbols."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        # Set up mock subscriptions
        mock_contract = MagicMock()
        client._subscriptions = {
            "AAPL": (mock_contract, MagicMock()),
            "MSFT": (mock_contract, MagicMock()),
        }

        client._ib.cancelMktData = MagicMock()

        await client.unsubscribe_quotes(["AAPL"])

        assert "AAPL" not in client._subscriptions
        assert "MSFT" in client._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self) -> None:
        """Test unsubscribing from all symbols."""
        client = IBClient()
        client._connected = True
        client._ib = MagicMock()

        mock_contract = MagicMock()
        client._subscriptions = {
            "AAPL": (mock_contract, MagicMock()),
            "MSFT": (mock_contract, MagicMock()),
        }

        client._ib.cancelMktData = MagicMock()

        await client.unsubscribe_quotes()

        assert len(client._subscriptions) == 0
