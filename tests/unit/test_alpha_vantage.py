"""Unit tests for Alpha Vantage client."""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pandas as pd
import pytest
from freezegun import freeze_time

from src.data.alpha_vantage import AlphaVantageClient
from src.data.exceptions import AlphaVantageError, InvalidSymbolError, RateLimitError
from src.data.rate_limiter import TokenBucket
from tests.fixtures.alpha_vantage_responses import (
    API_ERROR_RESPONSE,
    COMPANY_OVERVIEW_RESPONSE,
    COMPANY_OVERVIEW_WITH_NULLS,
    DAILY_OHLCV_RESPONSE,
    DAILY_OHLCV_WITH_GAPS,
    EMPTY_OPTIONS_RESPONSE,
    INFORMATION_RESPONSE,
    INTRADAY_RESPONSE,
    INVALID_SYMBOL_RESPONSE,
    OPTIONS_CHAIN_RESPONSE,
    RATE_LIMIT_RESPONSE,
)


class TestTokenBucket:
    """Tests for the token bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_initial_tokens_available(self) -> None:
        """Test that bucket starts with full capacity."""
        bucket = TokenBucket(calls_per_minute=60)
        assert bucket.available_tokens() == 60.0

    @pytest.mark.asyncio
    async def test_acquire_decrements_tokens(self) -> None:
        """Test that acquiring tokens decrements the count."""
        bucket = TokenBucket(calls_per_minute=60)
        await bucket.acquire(1)
        # Should be approximately 59 (may have slightly refilled)
        assert bucket.available_tokens() < 60.0

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self) -> None:
        """Test acquiring multiple tokens at once."""
        bucket = TokenBucket(calls_per_minute=60)
        await bucket.acquire(5)
        assert bucket.available_tokens() < 56.0

    @pytest.mark.asyncio
    async def test_rate_limiter_enforces_limit(self) -> None:
        """Test that rate limiter blocks when tokens exhausted."""
        bucket = TokenBucket(calls_per_minute=2)  # Very low limit for testing

        # Exhaust tokens quickly
        await bucket.acquire(2)

        # Next acquire should wait
        start = asyncio.get_event_loop().time()
        await bucket.acquire(1)
        elapsed = asyncio.get_event_loop().time() - start

        # Should have waited for token replenishment
        assert elapsed > 0.1  # At least some wait time

    @pytest.mark.asyncio
    async def test_tokens_replenish_over_time(self) -> None:
        """Test that tokens replenish after time passes."""
        bucket = TokenBucket(calls_per_minute=60)  # 1 token per second
        await bucket.acquire(60)  # Exhaust all tokens

        # Wait a bit for replenishment
        await asyncio.sleep(0.5)

        # Should have some tokens back (approximately 0.5)
        tokens = bucket.available_tokens()
        assert tokens > 0

    def test_reset_restores_full_capacity(self) -> None:
        """Test that reset restores the bucket to full capacity."""
        bucket = TokenBucket(calls_per_minute=60)
        bucket._tokens = 0.0  # Simulate exhausted bucket
        bucket.reset()
        assert bucket.available_tokens() == 60.0


class TestAlphaVantageClient:
    """Tests for the Alpha Vantage API client."""

    @pytest.fixture
    def client(self) -> AlphaVantageClient:
        """Create a test client."""
        return AlphaVantageClient(api_key="test_api_key", calls_per_minute=1000)

    @pytest.fixture
    def mock_response(self) -> AsyncMock:
        """Create a mock HTTP response."""
        response = AsyncMock(spec=httpx.Response)
        response.status_code = 200
        response.raise_for_status = AsyncMock()
        return response

    @pytest.mark.asyncio
    async def test_get_daily_ohlcv_success(self, client: AlphaVantageClient) -> None:
        """Test successful daily OHLCV data fetch."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = DAILY_OHLCV_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            df = await client.get_daily_ohlcv("AAPL")

        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjusted_close",
        ]

        # Verify dtypes
        assert df["open"].dtype == "float64"
        assert df["high"].dtype == "float64"
        assert df["low"].dtype == "float64"
        assert df["close"].dtype == "float64"
        assert df["volume"].dtype == "int64"
        assert df["adjusted_close"].dtype == "float64"

        # Verify index is DatetimeIndex with UTC
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None

        # Verify data sorted by date ascending
        assert df.index.is_monotonic_increasing

        await client.close()

    @pytest.mark.asyncio
    async def test_get_daily_ohlcv_invalid_symbol(
        self, client: AlphaVantageClient
    ) -> None:
        """Test that InvalidSymbolError is raised for unknown symbols."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = INVALID_SYMBOL_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(InvalidSymbolError) as exc_info:
                await client.get_daily_ohlcv("INVALID")

        assert "INVALID" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_error_raised(self, client: AlphaVantageClient) -> None:
        """Test that RateLimitError is raised when API rate limits."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = RATE_LIMIT_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(RateLimitError) as exc_info:
                await client.get_daily_ohlcv("AAPL")

        assert exc_info.value.retry_after == 60.0
        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self, client: AlphaVantageClient) -> None:
        """Test that retry logic handles 500 errors."""
        # First two calls fail with 500, third succeeds
        fail_response = AsyncMock(spec=httpx.Response)
        fail_response.status_code = 500
        fail_response.raise_for_status = AsyncMock()

        success_response = AsyncMock(spec=httpx.Response)
        success_response.status_code = 200
        success_response.json.return_value = DAILY_OHLCV_RESPONSE
        success_response.raise_for_status = AsyncMock()

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return fail_response
            return success_response

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            # Mock sleep to speed up test
            with patch.object(client, "_sleep", new_callable=AsyncMock):
                df = await client.get_daily_ohlcv("AAPL")

        assert call_count == 3  # Two retries + success
        assert isinstance(df, pd.DataFrame)
        await client.close()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, client: AlphaVantageClient) -> None:
        """Test that error is raised after max retries."""
        fail_response = AsyncMock(spec=httpx.Response)
        fail_response.status_code = 500
        fail_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=fail_response):
            with patch.object(client, "_sleep", new_callable=AsyncMock):
                with pytest.raises(AlphaVantageError) as exc_info:
                    await client.get_daily_ohlcv("AAPL")

        assert "Max retries exceeded" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_options_chain_parsing(self, client: AlphaVantageClient) -> None:
        """Test that options chain data is parsed correctly."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = OPTIONS_CHAIN_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await client.get_options_chain("AAPL")

        assert result["symbol"] == "AAPL"
        assert len(result["calls"]) == 2
        assert len(result["puts"]) == 1

        # Verify call option data
        call = result["calls"][0]
        assert call["strike"] == "180.00"
        assert call["type"] == "call"
        assert call["delta"] == "0.65"

        # Verify put option data
        put = result["puts"][0]
        assert put["strike"] == "180.00"
        assert put["type"] == "put"
        assert put["delta"] == "-0.35"

        await client.close()

    @pytest.mark.asyncio
    async def test_empty_options_chain(self, client: AlphaVantageClient) -> None:
        """Test handling of empty options chain."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = EMPTY_OPTIONS_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await client.get_options_chain("NOOPT")

        assert result["calls"] == []
        assert result["puts"] == []
        await client.close()

    @pytest.mark.asyncio
    async def test_handles_missing_data(self, client: AlphaVantageClient) -> None:
        """Test that missing data is handled gracefully."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = DAILY_OHLCV_WITH_GAPS
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            df = await client.get_daily_ohlcv("TEST")

        # Should have all valid data points
        assert len(df) == 2
        assert not df.isnull().any().any()  # No NaN values in this case
        await client.close()

    @pytest.mark.asyncio
    async def test_intraday_success(self, client: AlphaVantageClient) -> None:
        """Test successful intraday data fetch."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = INTRADAY_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            df = await client.get_intraday("AAPL", interval="5min")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df["volume"].dtype == "int64"
        await client.close()

    @pytest.mark.asyncio
    async def test_intraday_invalid_interval(self, client: AlphaVantageClient) -> None:
        """Test that invalid interval raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            await client.get_intraday("AAPL", interval="invalid")

        assert "Invalid interval" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_company_overview_success(self, client: AlphaVantageClient) -> None:
        """Test successful company overview fetch."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = COMPANY_OVERVIEW_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await client.get_company_overview("AAPL")

        assert result["symbol"] == "AAPL"
        assert result["name"] == "Apple Inc"
        assert result["sector"] == "TECHNOLOGY"
        assert result["market_capitalization"] == 2950000000000
        assert result["pe_ratio"] == 29.50
        await client.close()

    @pytest.mark.asyncio
    async def test_company_overview_with_nulls(
        self, client: AlphaVantageClient
    ) -> None:
        """Test handling of null values in company overview."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = COMPANY_OVERVIEW_WITH_NULLS
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await client.get_company_overview("STARTUP")

        assert result["symbol"] == "STARTUP"
        assert result["market_capitalization"] is None
        assert result["pe_ratio"] is None
        assert result["eps"] is None
        await client.close()

    @pytest.mark.asyncio
    async def test_api_error_handling(self, client: AlphaVantageClient) -> None:
        """Test handling of generic API errors."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = API_ERROR_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(AlphaVantageError) as exc_info:
                await client.get_daily_ohlcv("AAPL")

        assert "Something went wrong" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager usage."""
        async with AlphaVantageClient(api_key="test_key") as client:
            assert client._client is None  # Client not created until first request

        # Client should be closed after context exit
        assert client._client is None or client._client.is_closed

    @pytest.mark.asyncio
    async def test_timeout_retry(self, client: AlphaVantageClient) -> None:
        """Test that timeout errors trigger retry."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.TimeoutException("Request timed out")
            response = AsyncMock(spec=httpx.Response)
            response.status_code = 200
            response.json.return_value = DAILY_OHLCV_RESPONSE
            response.raise_for_status = AsyncMock()
            return response

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            with patch.object(client, "_sleep", new_callable=AsyncMock):
                df = await client.get_daily_ohlcv("AAPL")

        assert call_count == 3
        assert isinstance(df, pd.DataFrame)
        await client.close()


class TestRateLimiterIntegration:
    """Integration tests for rate limiter with client."""

    @pytest.mark.asyncio
    @freeze_time("2024-01-15 12:00:00", auto_tick_seconds=0.5)
    async def test_rate_limiter_prevents_burst(self) -> None:
        """Test that rate limiter prevents exceeding calls/minute."""
        client = AlphaVantageClient(api_key="test_key", calls_per_minute=3)

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = DAILY_OHLCV_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            # Make 3 quick calls (should exhaust bucket)
            for _ in range(3):
                await client.get_daily_ohlcv("AAPL")

            # Bucket should be nearly empty
            available = client.rate_limiter.available_tokens()
            assert available < 1.0

        await client.close()


class TestAdditionalCoverage:
    """Additional tests for improved coverage."""

    @pytest.fixture
    def client(self) -> AlphaVantageClient:
        """Create a test client."""
        return AlphaVantageClient(api_key="test_api_key", calls_per_minute=1000)

    @pytest.mark.asyncio
    async def test_http_status_error(self, client: AlphaVantageClient) -> None:
        """Test handling of HTTP status errors."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Forbidden",
            request=AsyncMock(),
            response=mock_response,
        )

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(AlphaVantageError) as exc_info:
                await client.get_daily_ohlcv("AAPL")

        assert exc_info.value.status_code == 403
        await client.close()

    @pytest.mark.asyncio
    async def test_information_message_error(self, client: AlphaVantageClient) -> None:
        """Test handling of information messages (subscription issues)."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = INFORMATION_RESPONSE
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(AlphaVantageError) as exc_info:
                await client.get_daily_ohlcv("AAPL")

        assert "premium" in str(exc_info.value).lower()
        await client.close()

    @pytest.mark.asyncio
    async def test_unexpected_response_format_daily(
        self, client: AlphaVantageClient
    ) -> None:
        """Test handling of unexpected response format for daily data."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(AlphaVantageError) as exc_info:
                await client.get_daily_ohlcv("AAPL")

        assert "Unexpected response format" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_unexpected_response_format_intraday(
        self, client: AlphaVantageClient
    ) -> None:
        """Test handling of unexpected response format for intraday data."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(AlphaVantageError) as exc_info:
                await client.get_intraday("AAPL", interval="5min")

        assert "Unexpected response format" in str(exc_info.value)
        await client.close()

    @pytest.mark.asyncio
    async def test_company_overview_invalid_symbol(
        self, client: AlphaVantageClient
    ) -> None:
        """Test InvalidSymbolError for empty company overview."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # Empty response
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            with pytest.raises(InvalidSymbolError):
                await client.get_company_overview("INVALID")

        await client.close()

    @pytest.mark.asyncio
    async def test_options_chain_no_data_key(
        self, client: AlphaVantageClient
    ) -> None:
        """Test options chain with no data key returns empty lists."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"symbol": "TEST"}  # No data key
        mock_response.raise_for_status = AsyncMock()

        with patch.object(httpx.AsyncClient, "get", return_value=mock_response):
            result = await client.get_options_chain("TEST")

        assert result["calls"] == []
        assert result["puts"] == []
        await client.close()

    @pytest.mark.asyncio
    async def test_timeout_max_retries(self, client: AlphaVantageClient) -> None:
        """Test that timeout errors exhaust retries."""
        async def mock_get(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        with patch.object(httpx.AsyncClient, "get", side_effect=mock_get):
            with patch.object(client, "_sleep", new_callable=AsyncMock):
                with pytest.raises(AlphaVantageError) as exc_info:
                    await client.get_daily_ohlcv("AAPL")

        assert "timeout" in str(exc_info.value).lower()
        await client.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client: AlphaVantageClient) -> None:
        """Test that close() can be called multiple times."""
        await client.close()
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(
        self, client: AlphaVantageClient
    ) -> None:
        """Test that _get_client reuses existing client."""
        client1 = await client._get_client()
        client2 = await client._get_client()
        assert client1 is client2
        await client.close()


class TestPydanticModels:
    """Tests for pydantic model validation."""

    def test_ohlcv_data_string_parsing(self) -> None:
        """Test OHLCVData parses string values correctly."""
        from src.data.models import OHLCVData

        data = OHLCVData.model_validate({
            "1. open": "100.50",
            "2. high": "105.25",
            "3. low": "99.75",
            "4. close": "104.00",
            "5. volume": "1000000",
        })
        assert data.open == 100.50
        assert data.high == 105.25
        assert data.volume == 1000000

    def test_daily_ohlcv_data_parsing(self) -> None:
        """Test DailyOHLCVData parses correctly."""
        from src.data.models import DailyOHLCVData

        data = DailyOHLCVData.model_validate({
            "1. open": "100.50",
            "2. high": "105.25",
            "3. low": "99.75",
            "4. close": "104.00",
            "5. adjusted close": "103.50",
            "6. volume": "1000000",
        })
        assert data.adjusted_close == 103.50
        assert data.volume == 1000000

    def test_company_overview_numeric_parsing(self) -> None:
        """Test CompanyOverview handles various numeric formats."""
        from src.data.models import CompanyOverview

        data = CompanyOverview.model_validate({
            "Symbol": "TEST",
            "Name": "Test Corp",
            "Description": "Test company",
            "Exchange": "NYSE",
            "Currency": "USD",
            "Country": "USA",
            "Sector": "Technology",
            "Industry": "Software",
            "MarketCapitalization": "1000000000",
            "PERatio": "25.5",
            "EPS": "-",  # Dash should become None
            "DividendYield": "None",  # None string should become None
            "52WeekHigh": "150.00",
            "52WeekLow": "100.00",
        })
        assert data.market_capitalization == 1000000000
        assert data.pe_ratio == 25.5
        assert data.eps is None
        assert data.dividend_yield is None

    def test_option_contract_parsing(self) -> None:
        """Test OptionContract parses all fields."""
        from src.data.models import OptionContract

        data = OptionContract.model_validate({
            "contractID": "TEST240119C00100000",
            "symbol": "TEST",
            "expiration": "2024-01-19",
            "strike": "100.00",
            "type": "call",
            "lastPrice": "5.50",
            "bid": "5.40",
            "ask": "5.60",
            "volume": "1000",
            "openInterest": "5000",
            "impliedVolatility": "0.25",
            "delta": "0.55",
            "gamma": "0.03",
            "theta": "-0.10",
            "vega": "0.15",
        })
        assert data.strike == 100.00
        assert data.delta == 0.55
        assert data.volume == 1000

    def test_option_contract_none_handling(self) -> None:
        """Test OptionContract handles None and dash values."""
        from src.data.models import OptionContract

        data = OptionContract.model_validate({
            "contractID": "TEST240119C00100000",
            "symbol": "TEST",
            "expiration": "2024-01-19",
            "strike": "100.00",
            "type": "call",
            "lastPrice": "None",
            "bid": "-",
            "ask": None,
            "volume": "None",
            "openInterest": "-",
        })
        assert data.last_price is None
        assert data.bid is None
        assert data.ask is None
        assert data.volume is None
        assert data.open_interest is None
