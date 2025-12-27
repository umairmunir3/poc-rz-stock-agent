"""Pydantic models for Alpha Vantage API responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OHLCVData(BaseModel):
    """Single OHLCV data point."""

    open: float = Field(..., alias="1. open")
    high: float = Field(..., alias="2. high")
    low: float = Field(..., alias="3. low")
    close: float = Field(..., alias="4. close")
    volume: int = Field(..., alias="5. volume")
    adjusted_close: float | None = Field(default=None, alias="5. adjusted close")

    model_config = {"populate_by_name": True}

    @field_validator("volume", mode="before")
    @classmethod
    def parse_volume(cls, v: Any) -> int:
        """Parse volume as integer."""
        if isinstance(v, str):
            return int(float(v))
        return int(v)

    @field_validator("open", "high", "low", "close", "adjusted_close", mode="before")
    @classmethod
    def parse_float(cls, v: Any) -> float | None:
        """Parse price fields as float."""
        if v is None:
            return None
        if isinstance(v, str):
            return float(v)
        return float(v)


class DailyOHLCVData(BaseModel):
    """Single daily OHLCV data point with adjusted close."""

    open: float = Field(..., alias="1. open")
    high: float = Field(..., alias="2. high")
    low: float = Field(..., alias="3. low")
    close: float = Field(..., alias="4. close")
    adjusted_close: float = Field(..., alias="5. adjusted close")
    volume: int = Field(..., alias="6. volume")

    model_config = {"populate_by_name": True}

    @field_validator("volume", mode="before")
    @classmethod
    def parse_volume(cls, v: Any) -> int:
        """Parse volume as integer."""
        if isinstance(v, str):
            return int(float(v))
        return int(v)

    @field_validator("open", "high", "low", "close", "adjusted_close", mode="before")
    @classmethod
    def parse_float(cls, v: Any) -> float:
        """Parse price fields as float."""
        if isinstance(v, str):
            return float(v)
        return float(v)


class TimeSeriesMetadata(BaseModel):
    """Metadata from time series response."""

    information: str = Field(..., alias="1. Information")
    symbol: str = Field(..., alias="2. Symbol")
    last_refreshed: str = Field(..., alias="3. Last Refreshed")
    output_size: str | None = Field(default=None, alias="4. Output Size")
    time_zone: str = Field(..., alias="4. Time Zone")

    model_config = {"populate_by_name": True}


class CompanyOverview(BaseModel):
    """Company overview data."""

    symbol: str = Field(..., alias="Symbol")
    name: str = Field(..., alias="Name")
    description: str = Field(..., alias="Description")
    exchange: str = Field(..., alias="Exchange")
    currency: str = Field(..., alias="Currency")
    country: str = Field(..., alias="Country")
    sector: str = Field(..., alias="Sector")
    industry: str = Field(..., alias="Industry")
    market_capitalization: int | None = Field(None, alias="MarketCapitalization")
    pe_ratio: float | None = Field(None, alias="PERatio")
    eps: float | None = Field(None, alias="EPS")
    dividend_yield: float | None = Field(None, alias="DividendYield")
    fifty_two_week_high: float | None = Field(None, alias="52WeekHigh")
    fifty_two_week_low: float | None = Field(None, alias="52WeekLow")
    fifty_day_moving_average: float | None = Field(None, alias="50DayMovingAverage")
    two_hundred_day_moving_average: float | None = Field(
        None, alias="200DayMovingAverage"
    )
    beta: float | None = Field(None, alias="Beta")

    model_config = {"populate_by_name": True}

    @field_validator(
        "market_capitalization",
        "pe_ratio",
        "eps",
        "dividend_yield",
        "fifty_two_week_high",
        "fifty_two_week_low",
        "fifty_day_moving_average",
        "two_hundred_day_moving_average",
        "beta",
        mode="before",
    )
    @classmethod
    def parse_numeric(cls, v: Any) -> float | int | None:
        """Parse numeric fields, handling 'None' strings."""
        if v is None or v == "None" or v == "-":
            return None
        if isinstance(v, str):
            try:
                # Try int first for market cap
                if "." not in v:
                    return int(v)
                return float(v)
            except ValueError:
                return None
        return v


class OptionContract(BaseModel):
    """Single option contract data."""

    contract_id: str = Field(..., alias="contractID")
    symbol: str
    expiration: datetime
    strike: float
    option_type: str = Field(..., alias="type")
    last_price: float | None = Field(None, alias="lastPrice")
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    open_interest: int | None = Field(None, alias="openInterest")
    implied_volatility: float | None = Field(None, alias="impliedVolatility")
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None

    model_config = {"populate_by_name": True}

    @field_validator("expiration", mode="before")
    @classmethod
    def parse_expiration(cls, v: Any) -> datetime:
        """Parse expiration date."""
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator(
        "strike",
        "last_price",
        "bid",
        "ask",
        "implied_volatility",
        "delta",
        "gamma",
        "theta",
        "vega",
        mode="before",
    )
    @classmethod
    def parse_float_or_none(cls, v: Any) -> float | None:
        """Parse float fields, handling None values."""
        if v is None or v == "None" or v == "-":
            return None
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return float(v)

    @field_validator("volume", "open_interest", mode="before")
    @classmethod
    def parse_int_or_none(cls, v: Any) -> int | None:
        """Parse integer fields, handling None values."""
        if v is None or v == "None" or v == "-":
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return int(v)


class OptionsChain(BaseModel):
    """Full options chain data."""

    symbol: str
    calls: list[OptionContract] = Field(default_factory=list)
    puts: list[OptionContract] = Field(default_factory=list)
