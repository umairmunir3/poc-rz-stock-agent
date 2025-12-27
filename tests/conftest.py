"""Pytest configuration and shared fixtures."""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment variables."""
    env_vars = {
        "ALPHA_VANTAGE_API_KEY": "test_api_key",
        "IB_HOST": "127.0.0.1",
        "IB_PORT": "7497",
        "IB_CLIENT_ID": "1",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET": "test-bucket",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "TELEGRAM_BOT_TOKEN": "test_token",
        "TELEGRAM_CHAT_ID": "test_chat_id",
        "RISK_PER_TRADE": "0.01",
        "MAX_POSITIONS": "5",
        "MAX_DRAWDOWN": "0.10",
        "ENVIRONMENT": "development",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def sample_ohlcv_data() -> dict[str, list]:
    """Sample OHLCV data for testing."""
    return {
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [102.0, 103.0, 104.0, 103.5, 105.0],
        "low": [99.0, 100.0, 101.0, 100.5, 102.0],
        "close": [101.0, 102.0, 103.0, 102.5, 104.0],
        "volume": [1000000, 1100000, 1050000, 950000, 1200000],
    }
