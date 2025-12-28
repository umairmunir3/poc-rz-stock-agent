"""FastAPI Dashboard Backend - REST API for trading dashboard."""

from __future__ import annotations

import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Annotated, Any

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


class Settings(BaseModel):
    """Application settings."""

    api_key: str = "default-api-key"
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


settings = Settings()


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class Direction(str, Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(str, Enum):
    """Trade status."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Period(str, Enum):
    """Performance period."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL = "all"


class BreakerStatus(str, Enum):
    """Circuit breaker status."""

    OK = "OK"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"


class SignalResponse(BaseModel):
    """Response model for a trading signal."""

    id: str
    symbol: str
    direction: Direction
    strategy: str
    score: int
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class TradeResponse(BaseModel):
    """Response model for a trade."""

    id: str
    symbol: str
    direction: Direction
    status: TradeStatus
    entry_price: float
    entry_date: date
    shares: int
    stop_loss: float
    take_profit: float
    exit_price: float | None = None
    exit_date: date | None = None
    pnl: float = 0.0
    pnl_percent: float = 0.0


class PositionResponse(BaseModel):
    """Response model for a position."""

    symbol: str
    shares: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float


class PortfolioResponse(BaseModel):
    """Response model for portfolio."""

    total_value: float
    cash: float
    positions: list[PositionResponse]
    total_exposure: float
    exposure_percent: float


class EquityCurvePoint(BaseModel):
    """Response model for equity curve data point."""

    date: date
    equity: float
    drawdown: float


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""

    period: Period
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_hold_days: float


class CircuitBreakerResponse(BaseModel):
    """Response model for circuit breaker."""

    name: str
    status: BreakerStatus
    can_trade: bool
    message: str
    triggered_at: datetime | None = None


class RiskStatusResponse(BaseModel):
    """Response model for risk status."""

    portfolio_heat: float
    daily_pnl: float
    daily_pnl_percent: float
    current_drawdown: float
    max_drawdown: float
    circuit_breakers: list[CircuitBreakerResponse]


class SettingsUpdateRequest(BaseModel):
    """Request model for updating settings."""

    risk_per_trade: float | None = Field(None, ge=0.001, le=0.1)
    max_positions: int | None = Field(None, ge=1, le=50)
    max_portfolio_heat: float | None = Field(None, ge=0.01, le=0.5)
    min_signal_score: int | None = Field(None, ge=0, le=100)


class SettingsResponse(BaseModel):
    """Response model for settings."""

    risk_per_trade: float
    max_positions: int
    max_portfolio_heat: float
    min_signal_score: int
    strategies_enabled: list[str]


# -----------------------------------------------------------------------------
# Mock Data Store (would be replaced with real storage)
# -----------------------------------------------------------------------------


@dataclass
class MockDataStore:
    """In-memory data store for demo purposes."""

    signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    trades: dict[str, dict[str, Any]] = field(default_factory=dict)
    positions: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    circuit_breakers: dict[str, dict[str, Any]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize with sample data."""
        # Sample signals
        self.signals = {
            "SIG-001": {
                "id": "SIG-001",
                "symbol": "AAPL",
                "direction": "LONG",
                "strategy": "RSI Mean Reversion",
                "score": 85,
                "entry_price": 178.50,
                "stop_loss": 174.20,
                "take_profit": 186.50,
                "reasoning": "RSI oversold at 28, price at 50 SMA support",
                "timestamp": datetime.now(),
                "metadata": {},
            },
            "SIG-002": {
                "id": "SIG-002",
                "symbol": "MSFT",
                "direction": "LONG",
                "strategy": "EMA Crossover",
                "score": 72,
                "entry_price": 420.00,
                "stop_loss": 410.00,
                "take_profit": 440.00,
                "reasoning": "EMA 9 crossed above EMA 21",
                "timestamp": datetime.now(),
                "metadata": {},
            },
        }

        # Sample trades
        self.trades = {
            "TRD-001": {
                "id": "TRD-001",
                "symbol": "GOOGL",
                "direction": "LONG",
                "status": "CLOSED",
                "entry_price": 140.00,
                "entry_date": date(2024, 1, 15),
                "shares": 10,
                "stop_loss": 135.00,
                "take_profit": 150.00,
                "exit_price": 148.00,
                "exit_date": date(2024, 1, 25),
                "pnl": 80.00,
                "pnl_percent": 5.71,
            },
            "TRD-002": {
                "id": "TRD-002",
                "symbol": "AAPL",
                "direction": "LONG",
                "status": "OPEN",
                "entry_price": 175.00,
                "entry_date": date(2024, 2, 1),
                "shares": 5,
                "stop_loss": 170.00,
                "take_profit": 185.00,
                "exit_price": None,
                "exit_date": None,
                "pnl": 0.0,
                "pnl_percent": 0.0,
            },
        }

        # Sample positions
        self.positions = [
            {
                "symbol": "AAPL",
                "shares": 5,
                "entry_price": 175.00,
                "current_price": 180.00,
                "market_value": 900.00,
                "unrealized_pnl": 25.00,
                "unrealized_pnl_percent": 2.86,
            }
        ]

        # Sample equity curve
        self.equity_curve = [
            {"date": date(2024, 1, 1), "equity": 10000.0, "drawdown": 0.0},
            {"date": date(2024, 1, 15), "equity": 10200.0, "drawdown": 0.0},
            {"date": date(2024, 2, 1), "equity": 10350.0, "drawdown": 0.0},
            {"date": date(2024, 2, 15), "equity": 10100.0, "drawdown": 0.024},
        ]

        # Sample circuit breakers
        self.circuit_breakers = {
            "DrawdownBreaker": {
                "name": "DrawdownBreaker",
                "status": "OK",
                "can_trade": True,
                "message": "Daily drawdown within limits",
                "triggered_at": None,
            },
            "LossStreakBreaker": {
                "name": "LossStreakBreaker",
                "status": "OK",
                "can_trade": True,
                "message": "No consecutive losses",
                "triggered_at": None,
            },
        }

        # Sample settings
        self.settings = {
            "risk_per_trade": 0.02,
            "max_positions": 10,
            "max_portfolio_heat": 0.20,
            "min_signal_score": 70,
            "strategies_enabled": [
                "RSI Mean Reversion",
                "EMA Crossover",
                "Breakout",
                "Support Bounce",
            ],
        }


# Global data store instance
data_store = MockDataStore()


# -----------------------------------------------------------------------------
# Rate Limiting
# -----------------------------------------------------------------------------


@dataclass
class RateLimiter:
    """Simple token bucket rate limiter."""

    max_requests: int = 100
    window_seconds: int = 60
    requests: dict[str, deque[float]] = field(default_factory=dict)

    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed.

        Args:
            client_id: Client identifier.

        Returns:
            True if allowed, False if rate limited.
        """
        current_time = time.time()

        if client_id not in self.requests:
            self.requests[client_id] = deque()

        # Remove old requests
        while (
            self.requests[client_id]
            and current_time - self.requests[client_id][0] > self.window_seconds
        ):
            self.requests[client_id].popleft()

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Record request
        self.requests[client_id].append(current_time)
        return True


rate_limiter = RateLimiter()


# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Verify API key from header.

    Args:
        x_api_key: API key from header.

    Returns:
        The verified API key.

    Raises:
        HTTPException: If API key is invalid or missing.
    """
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return x_api_key


async def check_rate_limit(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Check rate limit for client.

    Args:
        x_api_key: API key (used as client identifier).

    Raises:
        HTTPException: If rate limit exceeded.
    """
    client_id = x_api_key or "anonymous"
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------


app = FastAPI(
    title="Swing Trader Dashboard API",
    description="REST API for the swing trading dashboard",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Signal Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/signals", response_model=list[SignalResponse])
async def get_signals(
    strategy: str | None = Query(None, description="Filter by strategy name"),
    min_score: int | None = Query(None, ge=0, le=100, description="Minimum signal score"),
    direction: Direction | None = Query(None, description="Filter by direction"),
    _rate_limit: None = Depends(check_rate_limit),
) -> list[dict[str, Any]]:
    """Get active signals with optional filtering.

    Args:
        strategy: Filter by strategy name.
        min_score: Minimum signal score.
        direction: Filter by direction.

    Returns:
        List of signals.
    """
    signals = list(data_store.signals.values())

    if strategy:
        signals = [s for s in signals if s["strategy"] == strategy]
    if min_score is not None:
        signals = [s for s in signals if s["score"] >= min_score]
    if direction:
        signals = [s for s in signals if s["direction"] == direction.value]

    return signals


@app.get("/api/signals/{signal_id}", response_model=SignalResponse)
async def get_signal(
    signal_id: str,
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get a single signal by ID.

    Args:
        signal_id: Signal ID.

    Returns:
        Signal details.

    Raises:
        HTTPException: If signal not found.
    """
    if signal_id not in data_store.signals:
        raise HTTPException(status_code=404, detail="Signal not found")
    return data_store.signals[signal_id]


# -----------------------------------------------------------------------------
# Trade Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/trades", response_model=list[TradeResponse])
async def get_trades(
    status_filter: TradeStatus | None = Query(None, alias="status", description="Filter by status"),
    start_date: date | None = Query(None, description="Filter from date"),
    end_date: date | None = Query(None, description="Filter to date"),
    _rate_limit: None = Depends(check_rate_limit),
) -> list[dict[str, Any]]:
    """Get trade history with optional filtering.

    Args:
        status_filter: Filter by trade status.
        start_date: Filter from date.
        end_date: Filter to date.

    Returns:
        List of trades.
    """
    trades = list(data_store.trades.values())

    if status_filter:
        trades = [t for t in trades if t["status"] == status_filter.value]
    if start_date:
        trades = [t for t in trades if t["entry_date"] >= start_date]
    if end_date:
        trades = [t for t in trades if t["entry_date"] <= end_date]

    return trades


@app.get("/api/trades/{trade_id}", response_model=TradeResponse)
async def get_trade(
    trade_id: str,
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get a single trade by ID.

    Args:
        trade_id: Trade ID.

    Returns:
        Trade details.

    Raises:
        HTTPException: If trade not found.
    """
    if trade_id not in data_store.trades:
        raise HTTPException(status_code=404, detail="Trade not found")
    return data_store.trades[trade_id]


# -----------------------------------------------------------------------------
# Portfolio Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/portfolio", response_model=PortfolioResponse)
async def get_portfolio(
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get current portfolio status.

    Returns:
        Portfolio with positions and exposure.
    """
    positions = data_store.positions
    total_market_value = sum(p["market_value"] for p in positions)
    cash = 10000.0 - total_market_value  # Simplified calculation
    total_value = cash + total_market_value

    return {
        "total_value": total_value,
        "cash": cash,
        "positions": positions,
        "total_exposure": total_market_value,
        "exposure_percent": (total_market_value / total_value) * 100 if total_value > 0 else 0,
    }


@app.get("/api/portfolio/equity-curve", response_model=list[EquityCurvePoint])
async def get_equity_curve(
    start_date: date | None = Query(None, description="Start date"),
    end_date: date | None = Query(None, description="End date"),
    _rate_limit: None = Depends(check_rate_limit),
) -> list[dict[str, Any]]:
    """Get equity curve data.

    Args:
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        Equity curve data points.
    """
    curve = data_store.equity_curve

    if start_date:
        curve = [p for p in curve if p["date"] >= start_date]
    if end_date:
        curve = [p for p in curve if p["date"] <= end_date]

    return curve


# -----------------------------------------------------------------------------
# Performance Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/performance", response_model=PerformanceMetricsResponse)
async def get_performance(
    period: Period = Query(Period.ALL, description="Performance period"),
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get performance metrics.

    Args:
        period: Performance period.

    Returns:
        Performance metrics for the period.
    """
    # Simplified metrics - in production would calculate from real data
    return {
        "period": period,
        "total_return": 0.035,
        "cagr": 0.042,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.1,
        "max_drawdown": 0.024,
        "win_rate": 0.6,
        "profit_factor": 1.8,
        "total_trades": 15,
        "avg_hold_days": 5.2,
    }


# -----------------------------------------------------------------------------
# Risk Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/risk", response_model=RiskStatusResponse)
async def get_risk_status(
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get current risk status.

    Returns:
        Risk status including circuit breakers.
    """
    return {
        "portfolio_heat": 0.08,
        "daily_pnl": 150.0,
        "daily_pnl_percent": 1.5,
        "current_drawdown": 0.024,
        "max_drawdown": 0.05,
        "circuit_breakers": list(data_store.circuit_breakers.values()),
    }


@app.post("/api/circuit-breakers/{name}/reset")
async def reset_circuit_breaker(
    name: str,
    _api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, str]:
    """Reset a circuit breaker.

    Args:
        name: Circuit breaker name.

    Returns:
        Confirmation message.

    Raises:
        HTTPException: If circuit breaker not found.
    """
    if name not in data_store.circuit_breakers:
        raise HTTPException(status_code=404, detail="Circuit breaker not found")

    data_store.circuit_breakers[name]["status"] = "OK"
    data_store.circuit_breakers[name]["can_trade"] = True
    data_store.circuit_breakers[name]["triggered_at"] = None
    data_store.circuit_breakers[name]["message"] = "Reset by user"

    return {"message": f"Circuit breaker {name} has been reset"}


# -----------------------------------------------------------------------------
# Settings Endpoints
# -----------------------------------------------------------------------------


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings(
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Get current settings.

    Returns:
        Current configuration settings.
    """
    return data_store.settings


@app.put("/api/settings", response_model=SettingsResponse)
async def update_settings(
    updates: SettingsUpdateRequest,
    _api_key: str = Depends(verify_api_key),
    _rate_limit: None = Depends(check_rate_limit),
) -> dict[str, Any]:
    """Update settings.

    Args:
        updates: Settings to update.

    Returns:
        Updated settings.
    """
    update_dict = updates.model_dump(exclude_unset=True)
    for key, value in update_dict.items():
        if value is not None:
            data_store.settings[key] = value

    return data_store.settings


# -----------------------------------------------------------------------------
# WebSocket Connections
# -----------------------------------------------------------------------------


class ConnectionManager:
    """WebSocket connection manager."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        self.active_connections: dict[str, list[WebSocket]] = {
            "signals": [],
            "prices": [],
            "portfolio": [],
        }

    async def connect(self, websocket: WebSocket, channel: str) -> None:
        """Accept and register a WebSocket connection.

        Args:
            websocket: WebSocket connection.
            channel: Channel name (signals, prices, portfolio).
        """
        await websocket.accept()
        if channel in self.active_connections:
            self.active_connections[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str) -> None:
        """Remove a WebSocket connection.

        Args:
            websocket: WebSocket connection.
            channel: Channel name.
        """
        if channel in self.active_connections:
            if websocket in self.active_connections[channel]:
                self.active_connections[channel].remove(websocket)

    async def broadcast(self, channel: str, message: dict[str, Any]) -> None:
        """Broadcast message to all connections in a channel.

        Args:
            channel: Channel name.
            message: Message to broadcast.
        """
        if channel in self.active_connections:
            for connection in self.active_connections[channel]:
                with contextlib.suppress(Exception):
                    await connection.send_json(message)


manager = ConnectionManager()


async def verify_ws_token(token: str | None) -> bool:
    """Verify WebSocket authentication token.

    Args:
        token: Authentication token.

    Returns:
        True if valid, False otherwise.
    """
    return token == settings.api_key


@app.websocket("/ws/signals")
async def websocket_signals(
    websocket: WebSocket,
    token: str | None = Query(None),
) -> None:
    """WebSocket endpoint for real-time signal updates.

    Args:
        websocket: WebSocket connection.
        token: Authentication token.
    """
    if not await verify_ws_token(token):
        await websocket.close(code=4001)
        return

    await manager.connect(websocket, "signals")
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back or handle commands
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "signals")


@app.websocket("/ws/prices")
async def websocket_prices(
    websocket: WebSocket,
    token: str | None = Query(None),
) -> None:
    """WebSocket endpoint for real-time price updates.

    Args:
        websocket: WebSocket connection.
        token: Authentication token.
    """
    if not await verify_ws_token(token):
        await websocket.close(code=4001)
        return

    await manager.connect(websocket, "prices")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "prices")


@app.websocket("/ws/portfolio")
async def websocket_portfolio(
    websocket: WebSocket,
    token: str | None = Query(None),
) -> None:
    """WebSocket endpoint for real-time portfolio updates.

    Args:
        websocket: WebSocket connection.
        token: Authentication token.
    """
    if not await verify_ws_token(token):
        await websocket.close(code=4001)
        return

    await manager.connect(websocket, "portfolio")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, "portfolio")


# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy"}


# -----------------------------------------------------------------------------
# Helper Functions for Broadcasting
# -----------------------------------------------------------------------------


async def broadcast_new_signal(signal: dict[str, Any]) -> None:
    """Broadcast a new signal to connected clients.

    Args:
        signal: Signal data to broadcast.
    """
    await manager.broadcast(
        "signals",
        {"type": "new_signal", "data": signal},
    )


async def broadcast_price_update(symbol: str, price: float) -> None:
    """Broadcast a price update to connected clients.

    Args:
        symbol: Stock symbol.
        price: Current price.
    """
    await manager.broadcast(
        "prices",
        {"type": "price_update", "symbol": symbol, "price": price},
    )


async def broadcast_portfolio_update(portfolio: dict[str, Any]) -> None:
    """Broadcast a portfolio update to connected clients.

    Args:
        portfolio: Portfolio data.
    """
    await manager.broadcast(
        "portfolio",
        {"type": "portfolio_update", "data": portfolio},
    )
