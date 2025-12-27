"""Integration tests for Alpha Vantage API with live API calls.

These tests require a valid ALPHA_VANTAGE_API_KEY environment variable.
Run with: pytest -m integration
"""

import os

import pandas as pd
import pytest

from src.data.alpha_vantage import AlphaVantageClient


# Skip all tests in this module if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("ALPHA_VANTAGE_API_KEY"),
    reason="ALPHA_VANTAGE_API_KEY not set in environment",
)


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        pytest.skip("ALPHA_VANTAGE_API_KEY not set")
    return key


@pytest.fixture
def client(api_key: str) -> AlphaVantageClient:
    """Create a live API client."""
    return AlphaVantageClient(
        api_key=api_key,
        calls_per_minute=5,  # Very conservative for testing
    )


@pytest.mark.integration
class TestAlphaVantageLive:
    """Live API integration tests."""

    @pytest.mark.asyncio
    async def test_get_daily_ohlcv_live(self, client: AlphaVantageClient) -> None:
        """Test fetching real daily OHLCV data from API."""
        df = await client.get_daily_ohlcv("IBM", outputsize="compact")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert "adjusted_close" in df.columns

        # Verify dtypes
        assert df["open"].dtype == "float64"
        assert df["high"].dtype == "float64"
        assert df["low"].dtype == "float64"
        assert df["close"].dtype == "float64"
        assert df["volume"].dtype == "int64"

        # Verify data makes sense
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["volume"] >= 0).all()

        # Verify index is sorted ascending
        assert df.index.is_monotonic_increasing

        await client.close()

    @pytest.mark.asyncio
    async def test_get_company_overview_live(
        self, client: AlphaVantageClient
    ) -> None:
        """Test fetching real company overview from API."""
        result = await client.get_company_overview("IBM")

        # Verify basic fields
        assert result["symbol"] == "IBM"
        assert "name" in result
        assert "sector" in result
        assert "industry" in result

        await client.close()
