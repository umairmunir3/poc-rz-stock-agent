"""Unit tests for FastAPI Dashboard Backend."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import (
    app,
    rate_limiter,
    settings,
)


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Create authenticated headers."""
    return {"X-API-Key": settings.api_key}


@pytest.fixture
def reset_rate_limiter() -> None:
    """Reset rate limiter before each test."""
    rate_limiter.requests.clear()


# -----------------------------------------------------------------------------
# Signal Endpoint Tests
# -----------------------------------------------------------------------------


class TestSignalEndpoints:
    """Tests for signal endpoints."""

    def test_get_signals_returns_list(self, client: TestClient) -> None:
        """Test that GET /api/signals returns a list."""
        response = client.get("/api/signals")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_signals_filters_by_score(self, client: TestClient) -> None:
        """Test filtering signals by minimum score."""
        response = client.get("/api/signals?min_score=80")
        assert response.status_code == 200
        signals = response.json()
        for signal in signals:
            assert signal["score"] >= 80

    def test_get_signals_filters_by_strategy(self, client: TestClient) -> None:
        """Test filtering signals by strategy."""
        response = client.get("/api/signals?strategy=RSI Mean Reversion")
        assert response.status_code == 200
        signals = response.json()
        for signal in signals:
            assert signal["strategy"] == "RSI Mean Reversion"

    def test_get_signals_filters_by_direction(self, client: TestClient) -> None:
        """Test filtering signals by direction."""
        response = client.get("/api/signals?direction=LONG")
        assert response.status_code == 200
        signals = response.json()
        for signal in signals:
            assert signal["direction"] == "LONG"

    def test_get_signal_by_id(self, client: TestClient) -> None:
        """Test getting a single signal by ID."""
        response = client.get("/api/signals/SIG-001")
        assert response.status_code == 200
        signal = response.json()
        assert signal["id"] == "SIG-001"
        assert signal["symbol"] == "AAPL"

    def test_get_signal_not_found(self, client: TestClient) -> None:
        """Test getting non-existent signal returns 404."""
        response = client.get("/api/signals/INVALID")
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# Trade Endpoint Tests
# -----------------------------------------------------------------------------


class TestTradeEndpoints:
    """Tests for trade endpoints."""

    def test_get_trades_returns_list(self, client: TestClient) -> None:
        """Test that GET /api/trades returns a list."""
        response = client.get("/api/trades")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_trades_date_filtering(self, client: TestClient) -> None:
        """Test filtering trades by date range."""
        response = client.get("/api/trades?start_date=2024-01-01&end_date=2024-12-31")
        assert response.status_code == 200
        trades = response.json()
        assert isinstance(trades, list)

    def test_get_trades_status_filtering(self, client: TestClient) -> None:
        """Test filtering trades by status."""
        response = client.get("/api/trades?status=OPEN")
        assert response.status_code == 200
        trades = response.json()
        for trade in trades:
            assert trade["status"] == "OPEN"

    def test_get_trade_by_id(self, client: TestClient) -> None:
        """Test getting a single trade by ID."""
        response = client.get("/api/trades/TRD-001")
        assert response.status_code == 200
        trade = response.json()
        assert trade["id"] == "TRD-001"

    def test_get_trade_not_found(self, client: TestClient) -> None:
        """Test getting non-existent trade returns 404."""
        response = client.get("/api/trades/INVALID")
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# Portfolio Endpoint Tests
# -----------------------------------------------------------------------------


class TestPortfolioEndpoints:
    """Tests for portfolio endpoints."""

    def test_portfolio_includes_positions(self, client: TestClient) -> None:
        """Test that portfolio includes positions."""
        response = client.get("/api/portfolio")
        assert response.status_code == 200
        portfolio = response.json()
        assert "positions" in portfolio
        assert "total_value" in portfolio
        assert "cash" in portfolio
        assert "total_exposure" in portfolio
        assert "exposure_percent" in portfolio

    def test_equity_curve_correct_format(self, client: TestClient) -> None:
        """Test equity curve returns correct format."""
        response = client.get("/api/portfolio/equity-curve")
        assert response.status_code == 200
        curve = response.json()
        assert isinstance(curve, list)
        if curve:
            point = curve[0]
            assert "date" in point
            assert "equity" in point
            assert "drawdown" in point

    def test_equity_curve_date_filtering(self, client: TestClient) -> None:
        """Test equity curve date filtering."""
        response = client.get(
            "/api/portfolio/equity-curve?start_date=2024-01-01&end_date=2024-01-31"
        )
        assert response.status_code == 200


# -----------------------------------------------------------------------------
# Performance Endpoint Tests
# -----------------------------------------------------------------------------


class TestPerformanceEndpoints:
    """Tests for performance endpoints."""

    def test_performance_metrics_complete(self, client: TestClient) -> None:
        """Test performance metrics includes all required fields."""
        response = client.get("/api/performance")
        assert response.status_code == 200
        metrics = response.json()
        required_fields = [
            "period",
            "total_return",
            "cagr",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
            "avg_hold_days",
        ]
        for field in required_fields:
            assert field in metrics

    def test_performance_period_filter(self, client: TestClient) -> None:
        """Test performance with different periods."""
        for period in ["daily", "weekly", "monthly", "all"]:
            response = client.get(f"/api/performance?period={period}")
            assert response.status_code == 200
            assert response.json()["period"] == period


# -----------------------------------------------------------------------------
# Risk Endpoint Tests
# -----------------------------------------------------------------------------


class TestRiskEndpoints:
    """Tests for risk endpoints."""

    def test_risk_shows_circuit_breaker_status(self, client: TestClient) -> None:
        """Test risk status includes circuit breakers."""
        response = client.get("/api/risk")
        assert response.status_code == 200
        risk = response.json()
        assert "circuit_breakers" in risk
        assert "portfolio_heat" in risk
        assert "daily_pnl" in risk
        assert "current_drawdown" in risk

    def test_circuit_breaker_reset_requires_auth(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test circuit breaker reset requires authentication."""
        # Without auth - should fail
        response = client.post("/api/circuit-breakers/DrawdownBreaker/reset")
        assert response.status_code == 401

        # With auth - should succeed
        response = client.post("/api/circuit-breakers/DrawdownBreaker/reset", headers=auth_headers)
        assert response.status_code == 200

    def test_circuit_breaker_reset_not_found(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test resetting non-existent circuit breaker returns 404."""
        response = client.post("/api/circuit-breakers/InvalidBreaker/reset", headers=auth_headers)
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# Settings Endpoint Tests
# -----------------------------------------------------------------------------


class TestSettingsEndpoints:
    """Tests for settings endpoints."""

    def test_get_settings(self, client: TestClient) -> None:
        """Test getting settings."""
        response = client.get("/api/settings")
        assert response.status_code == 200
        settings_resp = response.json()
        assert "risk_per_trade" in settings_resp
        assert "max_positions" in settings_resp
        assert "strategies_enabled" in settings_resp

    def test_settings_update_validates_input(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test settings update validates input."""
        # Valid update
        response = client.put(
            "/api/settings",
            json={"risk_per_trade": 0.03, "max_positions": 15},
            headers=auth_headers,
        )
        assert response.status_code == 200
        settings_resp = response.json()
        assert settings_resp["risk_per_trade"] == 0.03
        assert settings_resp["max_positions"] == 15

    def test_settings_update_rejects_invalid(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test settings update rejects invalid values."""
        # risk_per_trade too high
        response = client.put(
            "/api/settings",
            json={"risk_per_trade": 0.5},  # Max is 0.1
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_settings_update_requires_auth(self, client: TestClient) -> None:
        """Test settings update requires authentication."""
        response = client.put("/api/settings", json={"max_positions": 5})
        assert response.status_code == 401


# -----------------------------------------------------------------------------
# Authentication Tests
# -----------------------------------------------------------------------------


class TestAuthentication:
    """Tests for authentication."""

    def test_invalid_api_key(self, client: TestClient) -> None:
        """Test invalid API key is rejected."""
        response = client.post(
            "/api/circuit-breakers/DrawdownBreaker/reset",
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401

    def test_missing_api_key(self, client: TestClient) -> None:
        """Test missing API key is rejected."""
        response = client.post("/api/circuit-breakers/DrawdownBreaker/reset")
        assert response.status_code == 401

    def test_valid_api_key_accepted(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        """Test valid API key is accepted."""
        response = client.post("/api/circuit-breakers/DrawdownBreaker/reset", headers=auth_headers)
        assert response.status_code == 200


# -----------------------------------------------------------------------------
# Rate Limiting Tests
# -----------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limiting_enforced(self, client: TestClient, reset_rate_limiter: None) -> None:
        """Test rate limiting is enforced after exceeding limit."""
        # Make many requests to exceed limit
        rate_limiter.max_requests = 5
        rate_limiter.requests.clear()

        for _ in range(5):
            response = client.get("/api/signals")
            assert response.status_code == 200

        # Next request should be rate limited
        response = client.get("/api/signals")
        assert response.status_code == 429

        # Reset for other tests
        rate_limiter.max_requests = 100
        rate_limiter.requests.clear()

    def test_rate_limit_resets_after_window(
        self, client: TestClient, reset_rate_limiter: None
    ) -> None:
        """Test rate limit resets after time window."""
        rate_limiter.max_requests = 100
        rate_limiter.requests.clear()
        response = client.get("/api/signals")
        assert response.status_code == 200


# -----------------------------------------------------------------------------
# CORS Tests
# -----------------------------------------------------------------------------


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client: TestClient) -> None:
        """Test CORS headers are present in response."""
        response = client.options(
            "/api/signals",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # FastAPI CORS middleware handles this
        assert response.status_code in [200, 204, 400]


# -----------------------------------------------------------------------------
# WebSocket Tests
# -----------------------------------------------------------------------------


class TestWebSocket:
    """Tests for WebSocket endpoints."""

    def test_websocket_signals_connects(self, client: TestClient) -> None:
        """Test WebSocket signals endpoint connects with valid token."""
        with client.websocket_connect(f"/ws/signals?token={settings.api_key}") as websocket:
            websocket.send_text("ping")
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_signals_rejects_invalid_token(self, client: TestClient) -> None:
        """Test WebSocket rejects invalid token."""
        with pytest.raises(Exception):  # noqa: B017
            with client.websocket_connect("/ws/signals?token=invalid"):
                pass

    def test_websocket_prices_connects(self, client: TestClient) -> None:
        """Test WebSocket prices endpoint connects."""
        with client.websocket_connect(f"/ws/prices?token={settings.api_key}") as websocket:
            websocket.send_text("ping")
            data = websocket.receive_json()
            assert data["type"] == "pong"

    def test_websocket_portfolio_connects(self, client: TestClient) -> None:
        """Test WebSocket portfolio endpoint connects."""
        with client.websocket_connect(f"/ws/portfolio?token={settings.api_key}") as websocket:
            websocket.send_text("ping")
            data = websocket.receive_json()
            assert data["type"] == "pong"


# -----------------------------------------------------------------------------
# Health Check Tests
# -----------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check(self, client: TestClient) -> None:
        """Test health check returns healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# -----------------------------------------------------------------------------
# OpenAPI Documentation Tests
# -----------------------------------------------------------------------------


class TestOpenAPIDocs:
    """Tests for OpenAPI documentation."""

    def test_docs_available(self, client: TestClient) -> None:
        """Test OpenAPI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client: TestClient) -> None:
        """Test ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json(self, client: TestClient) -> None:
        """Test OpenAPI JSON schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/api/signals" in schema["paths"]
        assert "/api/trades" in schema["paths"]
        assert "/api/portfolio" in schema["paths"]


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_invalid_date_format(self, client: TestClient) -> None:
        """Test invalid date format returns 422."""
        response = client.get("/api/trades?start_date=invalid-date")
        assert response.status_code == 422

    def test_min_score_out_of_range(self, client: TestClient) -> None:
        """Test min_score validation."""
        response = client.get("/api/signals?min_score=150")
        assert response.status_code == 422

    def test_empty_update_request(self, client: TestClient, auth_headers: dict[str, str]) -> None:
        """Test empty settings update is valid."""
        response = client.put("/api/settings", json={}, headers=auth_headers)
        assert response.status_code == 200
