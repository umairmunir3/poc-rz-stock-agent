"""Interactive Brokers TWS Client - connects to IB TWS for real-time data and orders."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ib_insync import IB

logger = logging.getLogger(__name__)


class IBClientError(Exception):
    """Base exception for IB client errors."""


class IBConnectionError(IBClientError):
    """Raised when connection to IB fails."""


class IBPacingError(IBClientError):
    """Raised when pacing violation occurs."""


class IBDataError(IBClientError):
    """Raised when data retrieval fails."""


@dataclass
class Quote:
    """Real-time quote data.

    Attributes:
        symbol: Stock ticker symbol.
        bid: Best bid price.
        ask: Best ask price.
        last: Last trade price.
        volume: Trading volume.
        timestamp: Quote timestamp.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime


@dataclass
class OptionQuote:
    """Option quote with greeks.

    Attributes:
        symbol: Underlying symbol.
        strike: Strike price.
        expiry: Expiration date.
        right: Option type ("C" or "P").
        bid: Best bid price.
        ask: Best ask price.
        last: Last trade price.
        volume: Trading volume.
        open_interest: Open interest.
        implied_volatility: Implied volatility.
        delta: Delta greek.
        gamma: Gamma greek.
        theta: Theta greek.
        vega: Vega greek.
    """

    symbol: str
    strike: float
    expiry: date
    right: str
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float


@dataclass
class OptionsChain:
    """Options chain for a symbol.

    Attributes:
        symbol: Underlying symbol.
        expiry: Expiration date.
        calls: List of call option quotes.
        puts: List of put option quotes.
        underlying_price: Current underlying price.
    """

    symbol: str
    expiry: date
    calls: list[OptionQuote]
    puts: list[OptionQuote]
    underlying_price: float


@dataclass
class Position:
    """Account position.

    Attributes:
        symbol: Stock/option symbol.
        quantity: Number of shares/contracts.
        avg_cost: Average cost basis.
        market_value: Current market value.
        unrealized_pnl: Unrealized profit/loss.
        realized_pnl: Realized profit/loss.
    """

    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Order:
    """Open order.

    Attributes:
        order_id: IB order ID.
        symbol: Stock/option symbol.
        action: BUY or SELL.
        order_type: Order type (LMT, MKT, etc.).
        quantity: Order quantity.
        limit_price: Limit price (if applicable).
        status: Order status.
        filled: Quantity filled.
        remaining: Quantity remaining.
    """

    order_id: int
    symbol: str
    action: str
    order_type: str
    quantity: float
    limit_price: float | None
    status: str
    filled: float
    remaining: float


@dataclass
class AccountSummary:
    """Account summary information.

    Attributes:
        account_id: Account identifier.
        equity: Total account equity.
        cash: Available cash.
        buying_power: Available buying power.
        net_liquidation: Net liquidation value.
        unrealized_pnl: Total unrealized P&L.
        realized_pnl: Total realized P&L.
        is_paper: Whether this is a paper trading account.
    """

    account_id: str
    equity: float
    cash: float
    buying_power: float
    net_liquidation: float
    unrealized_pnl: float
    realized_pnl: float
    is_paper: bool


@dataclass
class IBClientConfig:
    """Configuration for IB client.

    Attributes:
        host: TWS/Gateway host.
        port: TWS/Gateway port (7497 paper, 7496 live).
        client_id: Unique client identifier.
        timeout: Connection timeout in seconds.
        readonly: Whether to connect in read-only mode.
        auto_reconnect: Whether to auto-reconnect on disconnect.
        max_requests_per_second: Maximum requests per second (pacing).
    """

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1
    timeout: int = 20
    readonly: bool = True
    auto_reconnect: bool = True
    max_requests_per_second: float = 45.0  # IB limit is ~50/sec


class IBClient:
    """Interactive Brokers TWS/Gateway client.

    Provides async interface to IB for market data, options data,
    and account information.

    Example:
        >>> client = IBClient(host="127.0.0.1", port=7497, client_id=1)
        >>> await client.connect()
        >>> quote = await client.get_quote("AAPL")
        >>> print(f"AAPL: {quote.last}")
        >>> await client.disconnect()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        config: IBClientConfig | None = None,
    ) -> None:
        """Initialize the IB client.

        Args:
            host: TWS/Gateway host address.
            port: TWS/Gateway port.
            client_id: Unique client identifier.
            config: Optional full configuration.
        """
        if config:
            self.config = config
        else:
            self.config = IBClientConfig(host=host, port=port, client_id=client_id)

        self._ib: IB | None = None
        self._connected = False
        self._subscriptions: dict[str, Any] = {}
        self._last_request_time = 0.0
        self._request_interval = 1.0 / self.config.max_requests_per_second

    async def connect(self) -> bool:
        """Connect to IB TWS/Gateway.

        Returns:
            True if connection successful.

        Raises:
            IBConnectionError: If connection fails.
        """
        try:
            from ib_insync import IB

            self._ib = IB()

            # Connect with timeout
            await asyncio.wait_for(
                self._ib.connectAsync(
                    host=self.config.host,
                    port=self.config.port,
                    clientId=self.config.client_id,
                    readonly=self.config.readonly,
                ),
                timeout=self.config.timeout,
            )

            self._connected = True

            # Set up disconnect handler for auto-reconnect
            if self.config.auto_reconnect:
                self._ib.disconnectedEvent += self._on_disconnect

            logger.info(
                f"Connected to IB at {self.config.host}:{self.config.port} "
                f"(client_id={self.config.client_id})"
            )
            return True

        except TimeoutError as e:
            raise IBConnectionError(f"Connection timeout after {self.config.timeout}s") from e
        except Exception as e:
            raise IBConnectionError(f"Failed to connect to IB: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from IB TWS/Gateway."""
        if self._ib and self._connected:
            # Remove disconnect handler to prevent auto-reconnect
            if self.config.auto_reconnect:
                self._ib.disconnectedEvent -= self._on_disconnect

            self._ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    async def is_connected(self) -> bool:
        """Check if connected to IB.

        Returns:
            True if connected.
        """
        if self._ib is None:
            return False
        return bool(self._ib.isConnected())

    def _on_disconnect(self) -> None:
        """Handle disconnect event for auto-reconnect."""
        self._connected = False
        logger.warning("Disconnected from IB, attempting reconnect...")

        # Schedule reconnect
        asyncio.create_task(self._reconnect())

    async def _reconnect(self, max_attempts: int = 5, delay: float = 5.0) -> bool:
        """Attempt to reconnect to IB.

        Args:
            max_attempts: Maximum reconnection attempts.
            delay: Delay between attempts in seconds.

        Returns:
            True if reconnection successful.
        """
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Reconnection attempt {attempt}/{max_attempts}")
                await asyncio.sleep(delay)
                await self.connect()
                logger.info("Reconnection successful")
                return True
            except IBConnectionError:
                if attempt == max_attempts:
                    logger.error("Max reconnection attempts reached")
                    return False
        return False

    async def _pace_request(self) -> None:
        """Implement request pacing to avoid IB violations."""
        import time

        current_time = time.time()
        time_since_last = current_time - self._last_request_time

        if time_since_last < self._request_interval:
            await asyncio.sleep(self._request_interval - time_since_last)

        self._last_request_time = time.time()

    def _ensure_connected(self) -> None:
        """Ensure client is connected.

        Raises:
            IBConnectionError: If not connected.
        """
        if not self._connected or self._ib is None:
            raise IBConnectionError("Not connected to IB")

    async def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote with current market data.

        Raises:
            IBDataError: If quote retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected
        await self._pace_request()

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            ticker = self._ib.reqMktData(contract)
            await asyncio.sleep(0.5)  # Wait for data

            # Get snapshot
            self._ib.cancelMktData(contract)

            return Quote(
                symbol=symbol,
                bid=ticker.bid if ticker.bid and ticker.bid > 0 else 0.0,
                ask=ticker.ask if ticker.ask and ticker.ask > 0 else 0.0,
                last=ticker.last if ticker.last and ticker.last > 0 else 0.0,
                volume=int(ticker.volume) if ticker.volume else 0,
                timestamp=datetime.now(),
            )

        except Exception as e:
            raise IBDataError(f"Failed to get quote for {symbol}: {e}") from e

    async def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """Subscribe to real-time quotes for multiple symbols.

        Args:
            symbols: List of stock symbols.
            callback: Function called with each quote update.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected

        from ib_insync import Stock

        for symbol in symbols:
            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            ticker = self._ib.reqMktData(contract)

            def on_update(t: Any, symbol: str = symbol) -> None:
                quote = Quote(
                    symbol=symbol,
                    bid=t.bid if t.bid and t.bid > 0 else 0.0,
                    ask=t.ask if t.ask and t.ask > 0 else 0.0,
                    last=t.last if t.last and t.last > 0 else 0.0,
                    volume=int(t.volume) if t.volume else 0,
                    timestamp=datetime.now(),
                )
                callback(quote)

            ticker.updateEvent += on_update
            self._subscriptions[symbol] = (contract, ticker)

    async def unsubscribe_quotes(self, symbols: list[str] | None = None) -> None:
        """Unsubscribe from quote updates.

        Args:
            symbols: Symbols to unsubscribe (None for all).
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected

        to_unsubscribe = symbols if symbols else list(self._subscriptions.keys())

        for symbol in to_unsubscribe:
            if symbol in self._subscriptions:
                contract, _ = self._subscriptions[symbol]
                self._ib.cancelMktData(contract)
                del self._subscriptions[symbol]

    async def get_bars(
        self,
        symbol: str,
        duration: str = "1 D",
        bar_size: str = "1 hour",
    ) -> pd.DataFrame:
        """Get historical bars for a symbol.

        Args:
            symbol: Stock ticker symbol.
            duration: Duration string (e.g., "1 D", "5 D", "1 M").
            bar_size: Bar size (e.g., "1 min", "5 mins", "1 hour", "1 day").

        Returns:
            DataFrame with OHLCV data.

        Raises:
            IBDataError: If bar retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected
        await self._pace_request()

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(contract)

            bars = self._ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
            )

            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(
                [
                    {
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                    for b in bars
                ],
                index=pd.DatetimeIndex([b.date for b in bars]),
            )
            df.index.name = "date"

            return df

        except Exception as e:
            raise IBDataError(f"Failed to get bars for {symbol}: {e}") from e

    async def get_options_chain(
        self,
        symbol: str,
        expiry: date | None = None,
    ) -> OptionsChain:
        """Get options chain for a symbol.

        Args:
            symbol: Underlying stock symbol.
            expiry: Expiration date (None for nearest expiry).

        Returns:
            OptionsChain with calls and puts.

        Raises:
            IBDataError: If options chain retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected
        await self._pace_request()

        try:
            from ib_insync import Stock

            # Get underlying contract
            stock = Stock(symbol, "SMART", "USD")
            self._ib.qualifyContracts(stock)

            # Get underlying price
            ticker = self._ib.reqMktData(stock, snapshot=True)
            await asyncio.sleep(0.5)
            underlying_price = ticker.last if ticker.last else ticker.close

            # Get option chains
            chains = self._ib.reqSecDefOptParams(
                stock.symbol,
                "",
                stock.secType,
                stock.conId,
            )

            if not chains:
                raise IBDataError(f"No options chains found for {symbol}")

            # Use first chain (usually SMART exchange)
            chain = chains[0]

            # Get expiries
            expirations = sorted(chain.expirations)
            if expiry:
                target_expiry = expiry.strftime("%Y%m%d")
                if target_expiry not in expirations:
                    # Find nearest expiry
                    target_expiry = min(
                        expirations,
                        key=lambda x: abs(datetime.strptime(x, "%Y%m%d").date() - expiry),
                    )
            else:
                target_expiry = expirations[0]  # Nearest expiry

            expiry_date = datetime.strptime(target_expiry, "%Y%m%d").date()

            # Get strikes near the money
            strikes = sorted(chain.strikes)
            atm_strikes = [
                s for s in strikes if abs(s - underlying_price) / underlying_price < 0.10
            ]

            calls: list[OptionQuote] = []
            puts: list[OptionQuote] = []

            # Get option quotes for each strike
            for strike in atm_strikes[:10]:  # Limit to 10 strikes
                for right in ["C", "P"]:
                    try:
                        quote = await self.get_option_quote(symbol, strike, expiry_date, right)
                        if right == "C":
                            calls.append(quote)
                        else:
                            puts.append(quote)
                    except IBDataError:
                        pass  # Skip strikes with no data

            return OptionsChain(
                symbol=symbol,
                expiry=expiry_date,
                calls=calls,
                puts=puts,
                underlying_price=underlying_price,
            )

        except Exception as e:
            if isinstance(e, IBDataError):
                raise
            raise IBDataError(f"Failed to get options chain for {symbol}: {e}") from e

    async def get_option_quote(
        self,
        symbol: str,
        strike: float,
        expiry: date,
        right: str,
    ) -> OptionQuote:
        """Get quote for a specific option.

        Args:
            symbol: Underlying stock symbol.
            strike: Strike price.
            expiry: Expiration date.
            right: Option type ("C" for call, "P" for put).

        Returns:
            OptionQuote with price and greeks.

        Raises:
            IBDataError: If option quote retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected
        await self._pace_request()

        try:
            from ib_insync import Option

            contract = Option(
                symbol,
                expiry.strftime("%Y%m%d"),
                strike,
                right,
                "SMART",
            )
            self._ib.qualifyContracts(contract)

            # Get market data with greeks
            ticker = self._ib.reqMktData(contract, genericTickList="106")
            await asyncio.sleep(0.5)

            self._ib.cancelMktData(contract)

            # Extract greeks
            greeks = ticker.modelGreeks or ticker.lastGreeks
            delta = gamma = theta = vega = iv = 0.0

            if greeks:
                delta = greeks.delta or 0.0
                gamma = greeks.gamma or 0.0
                theta = greeks.theta or 0.0
                vega = greeks.vega or 0.0
                iv = greeks.impliedVol or 0.0

            return OptionQuote(
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                right=right,
                bid=ticker.bid if ticker.bid and ticker.bid > 0 else 0.0,
                ask=ticker.ask if ticker.ask and ticker.ask > 0 else 0.0,
                last=ticker.last if ticker.last and ticker.last > 0 else 0.0,
                volume=int(ticker.volume) if ticker.volume else 0,
                open_interest=0,  # Not available in real-time
                implied_volatility=iv,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
            )

        except Exception as e:
            raise IBDataError(
                f"Failed to get option quote for {symbol} {strike} {expiry} {right}: {e}"
            ) from e

    async def get_positions(self) -> list[Position]:
        """Get current account positions.

        Returns:
            List of Position objects.

        Raises:
            IBDataError: If position retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected

        try:
            positions = self._ib.positions()

            return [
                Position(
                    symbol=pos.contract.symbol,
                    quantity=pos.position,
                    avg_cost=pos.avgCost,
                    market_value=pos.position * pos.avgCost,  # Approximate
                    unrealized_pnl=0.0,  # Need portfolio data for this
                    realized_pnl=0.0,
                )
                for pos in positions
            ]

        except Exception as e:
            raise IBDataError(f"Failed to get positions: {e}") from e

    async def get_open_orders(self) -> list[Order]:
        """Get open orders.

        Returns:
            List of Order objects.

        Raises:
            IBDataError: If order retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected

        try:
            trades = self._ib.openTrades()

            return [
                Order(
                    order_id=trade.order.orderId,
                    symbol=trade.contract.symbol,
                    action=trade.order.action,
                    order_type=trade.order.orderType,
                    quantity=trade.order.totalQuantity,
                    limit_price=trade.order.lmtPrice if trade.order.lmtPrice else None,
                    status=trade.orderStatus.status,
                    filled=trade.orderStatus.filled,
                    remaining=trade.orderStatus.remaining,
                )
                for trade in trades
            ]

        except Exception as e:
            raise IBDataError(f"Failed to get open orders: {e}") from e

    async def get_account_summary(self) -> AccountSummary:
        """Get account summary information.

        Returns:
            AccountSummary with equity, cash, buying power.

        Raises:
            IBDataError: If account retrieval fails.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected

        try:
            # Get account values
            account_values = self._ib.accountSummary()

            # Build summary from account values
            summary_dict: dict[str, float] = {}
            account_id = ""

            for val in account_values:
                account_id = val.account
                if val.tag in [
                    "NetLiquidation",
                    "TotalCashValue",
                    "BuyingPower",
                    "UnrealizedPnL",
                    "RealizedPnL",
                    "EquityWithLoanValue",
                ]:
                    summary_dict[val.tag] = float(val.value)

            # Detect paper account (paper accounts typically start with "D")
            is_paper = account_id.startswith("D") if account_id else True

            return AccountSummary(
                account_id=account_id,
                equity=summary_dict.get("EquityWithLoanValue", 0.0),
                cash=summary_dict.get("TotalCashValue", 0.0),
                buying_power=summary_dict.get("BuyingPower", 0.0),
                net_liquidation=summary_dict.get("NetLiquidation", 0.0),
                unrealized_pnl=summary_dict.get("UnrealizedPnL", 0.0),
                realized_pnl=summary_dict.get("RealizedPnL", 0.0),
                is_paper=is_paper,
            )

        except Exception as e:
            raise IBDataError(f"Failed to get account summary: {e}") from e

    async def check_market_status(self) -> dict[str, bool]:
        """Check market data farm status.

        Returns:
            Dictionary with connection status for each farm.
        """
        self._ensure_connected()
        assert self._ib is not None  # Guaranteed by _ensure_connected

        try:
            # Request managed accounts to trigger farm status
            self._ib.reqManagedAccts()

            # The farm status would be in connection messages
            # For now, return basic connection status
            return {
                "connected": self._connected,
                "is_paper": await self._is_paper_account(),
            }

        except Exception:
            return {"connected": False, "is_paper": True}

    async def _is_paper_account(self) -> bool:
        """Check if connected to paper trading account.

        Returns:
            True if paper trading account.
        """
        if self._ib is None:
            return True
        try:
            accounts = self._ib.managedAccounts()
            if accounts:
                # Paper accounts typically start with "D"
                return bool(accounts[0].startswith("D"))
            return True
        except Exception:
            return True
