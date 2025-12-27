"""Alpha Vantage API client with rate limiting and error handling."""

import logging
import time
from typing import Any

import httpx
import pandas as pd

from src.data.exceptions import AlphaVantageError, InvalidSymbolError, RateLimitError
from src.data.models import CompanyOverview, DailyOHLCVData, OHLCVData
from src.data.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)

# Alpha Vantage base URL
BASE_URL = "https://www.alphavantage.co/query"

# Valid intraday intervals
VALID_INTERVALS = {"1min", "5min", "15min", "30min", "60min"}

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2.0
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


class AlphaVantageClient:
    """Async client for Alpha Vantage API with rate limiting.

    Implements token bucket rate limiting to stay within API limits
    and retry logic for transient failures.
    """

    def __init__(
        self,
        api_key: str,
        calls_per_minute: int = 75,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Alpha Vantage client.

        Args:
            api_key: Alpha Vantage API key.
            calls_per_minute: Maximum API calls per minute (75 for premium).
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.rate_limiter = TokenBucket(calls_per_minute=calls_per_minute)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AlphaVantageClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def _request(
        self,
        params: dict[str, str],
        retry_count: int = 0,
    ) -> dict[str, Any]:
        """Make an API request with rate limiting and retry logic.

        Args:
            params: Query parameters for the request.
            retry_count: Current retry attempt number.

        Returns:
            JSON response as dictionary.

        Raises:
            AlphaVantageError: For API errors.
            RateLimitError: When rate limited by the API.
            InvalidSymbolError: For unknown symbols.
        """
        # Acquire rate limit token
        wait_time = await self.rate_limiter.acquire()
        if wait_time > 0:
            logger.debug(f"Rate limiter waited {wait_time:.2f}s")

        # Add API key to params
        params["apikey"] = self.api_key

        start_time = time.perf_counter()
        client = await self._get_client()

        try:
            response = await client.get(BASE_URL, params=params)
            elapsed = time.perf_counter() - start_time

            logger.info(
                f"API call: {params.get('function', 'unknown')} "
                f"symbol={params.get('symbol', 'N/A')} "
                f"status={response.status_code} "
                f"time={elapsed:.3f}s"
            )

            # Handle retryable errors
            if response.status_code in RETRYABLE_STATUS_CODES:
                if retry_count < MAX_RETRIES:
                    backoff = RETRY_BACKOFF_FACTOR ** retry_count
                    logger.warning(
                        f"Retrying after {backoff}s due to {response.status_code}"
                    )
                    await self._sleep(backoff)
                    return await self._request(params, retry_count + 1)
                raise AlphaVantageError(
                    f"Max retries exceeded. Last status: {response.status_code}",
                    status_code=response.status_code,
                )

            response.raise_for_status()
            data = response.json()

            # Check for API error messages
            if "Error Message" in data:
                error_msg = data["Error Message"]
                if "Invalid API call" in error_msg or "invalid" in error_msg.lower():
                    symbol = params.get("symbol", "unknown")
                    raise InvalidSymbolError(symbol)
                raise AlphaVantageError(error_msg)

            # Check for rate limit message
            if "Note" in data and "call frequency" in data["Note"].lower():
                raise RateLimitError(
                    message=data["Note"],
                    retry_after=60.0,  # Wait a minute before retrying
                )

            # Check for information messages (often indicates issues)
            if "Information" in data and "Time Series" not in str(data.keys()):
                raise AlphaVantageError(data["Information"])

            return data

        except httpx.TimeoutException as e:
            if retry_count < MAX_RETRIES:
                backoff = RETRY_BACKOFF_FACTOR ** retry_count
                logger.warning(f"Timeout, retrying after {backoff}s")
                await self._sleep(backoff)
                return await self._request(params, retry_count + 1)
            raise AlphaVantageError(f"Request timeout: {e}") from e

        except httpx.HTTPStatusError as e:
            raise AlphaVantageError(
                f"HTTP error: {e.response.status_code}",
                status_code=e.response.status_code,
            ) from e

    async def _sleep(self, seconds: float) -> None:
        """Sleep for retry backoff. Extracted for testing."""
        import asyncio

        await asyncio.sleep(seconds)

    async def get_daily_ohlcv(
        self,
        symbol: str,
        outputsize: str = "full",
    ) -> pd.DataFrame:
        """Get daily OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol.
            outputsize: 'compact' (100 days) or 'full' (20+ years).

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, adjusted_close
            Index is DatetimeIndex in UTC.

        Raises:
            InvalidSymbolError: If symbol is unknown.
            AlphaVantageError: For other API errors.
        """
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol.upper(),
            "outputsize": outputsize,
        }

        data = await self._request(params)

        # Parse time series data
        time_series_key = "Time Series (Daily)"
        if time_series_key not in data:
            raise AlphaVantageError(f"Unexpected response format for {symbol}")

        time_series = data[time_series_key]

        # Convert to DataFrame
        records = []
        for date_str, values in time_series.items():
            try:
                ohlcv = DailyOHLCVData.model_validate(values)
                records.append(
                    {
                        "date": pd.to_datetime(date_str).tz_localize("UTC"),
                        "open": ohlcv.open,
                        "high": ohlcv.high,
                        "low": ohlcv.low,
                        "close": ohlcv.close,
                        "volume": ohlcv.volume,
                        "adjusted_close": ohlcv.adjusted_close,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse data for {date_str}: {e}")
                continue

        if not records:
            raise AlphaVantageError(f"No valid data returned for {symbol}")

        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        df = df.set_index("date")

        # Ensure correct dtypes
        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].astype("int64")
        df["adjusted_close"] = df["adjusted_close"].astype("float64")

        return df

    async def get_intraday(
        self,
        symbol: str,
        interval: str = "5min",
        outputsize: str = "compact",
    ) -> pd.DataFrame:
        """Get intraday OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol.
            interval: Time interval (1min, 5min, 15min, 30min, 60min).
            outputsize: 'compact' (100 points) or 'full' (30 days).

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
            Index is DatetimeIndex in UTC.

        Raises:
            ValueError: If interval is invalid.
            InvalidSymbolError: If symbol is unknown.
            AlphaVantageError: For other API errors.
        """
        if interval not in VALID_INTERVALS:
            raise ValueError(
                f"Invalid interval: {interval}. Must be one of {VALID_INTERVALS}"
            )

        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol.upper(),
            "interval": interval,
            "outputsize": outputsize,
        }

        data = await self._request(params)

        # Parse time series data
        time_series_key = f"Time Series ({interval})"
        if time_series_key not in data:
            raise AlphaVantageError(f"Unexpected response format for {symbol}")

        time_series = data[time_series_key]

        # Convert to DataFrame
        records = []
        for datetime_str, values in time_series.items():
            try:
                ohlcv = OHLCVData.model_validate(values)
                records.append(
                    {
                        "datetime": pd.to_datetime(datetime_str).tz_localize("UTC"),
                        "open": ohlcv.open,
                        "high": ohlcv.high,
                        "low": ohlcv.low,
                        "close": ohlcv.close,
                        "volume": ohlcv.volume,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to parse data for {datetime_str}: {e}")
                continue

        if not records:
            raise AlphaVantageError(f"No valid data returned for {symbol}")

        df = pd.DataFrame(records)
        df = df.sort_values("datetime").reset_index(drop=True)
        df = df.set_index("datetime")

        # Ensure correct dtypes
        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].astype("int64")

        return df

    async def get_options_chain(self, symbol: str) -> dict[str, Any]:
        """Get options chain data for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with 'calls' and 'puts' lists containing option contracts.

        Raises:
            InvalidSymbolError: If symbol is unknown.
            AlphaVantageError: For other API errors.
        """
        params = {
            "function": "REALTIME_OPTIONS",
            "symbol": symbol.upper(),
        }

        data = await self._request(params)

        # Parse options data
        if "data" not in data:
            # Return empty chain if no options available
            return {"symbol": symbol.upper(), "calls": [], "puts": []}

        calls = []
        puts = []

        for contract in data.get("data", []):
            option_type = contract.get("type", "").lower()
            contract_data = {
                "contractID": contract.get("contractID", ""),
                "symbol": contract.get("symbol", symbol.upper()),
                "expiration": contract.get("expiration", ""),
                "strike": contract.get("strike", 0),
                "type": option_type,
                "lastPrice": contract.get("last", contract.get("lastPrice")),
                "bid": contract.get("bid"),
                "ask": contract.get("ask"),
                "volume": contract.get("volume"),
                "openInterest": contract.get("open_interest"),
                "impliedVolatility": contract.get("implied_volatility"),
                "delta": contract.get("delta"),
                "gamma": contract.get("gamma"),
                "theta": contract.get("theta"),
                "vega": contract.get("vega"),
            }

            if option_type == "call":
                calls.append(contract_data)
            elif option_type == "put":
                puts.append(contract_data)

        return {
            "symbol": symbol.upper(),
            "calls": calls,
            "puts": puts,
        }

    async def get_company_overview(self, symbol: str) -> dict[str, Any]:
        """Get company overview data.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with company information including market cap, sector, etc.

        Raises:
            InvalidSymbolError: If symbol is unknown.
            AlphaVantageError: For other API errors.
        """
        params = {
            "function": "OVERVIEW",
            "symbol": symbol.upper(),
        }

        data = await self._request(params)

        # Check if we got valid data
        if not data or "Symbol" not in data:
            raise InvalidSymbolError(symbol)

        # Parse and validate with pydantic model
        try:
            overview = CompanyOverview.model_validate(data)
            return overview.model_dump(by_alias=False)
        except Exception as e:
            logger.warning(f"Failed to fully parse company overview: {e}")
            # Return raw data if parsing fails
            return data
