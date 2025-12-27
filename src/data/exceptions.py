"""Custom exceptions for data module."""


class AlphaVantageError(Exception):
    """Base exception for Alpha Vantage API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class RateLimitError(AlphaVantageError):
    """Raised when API rate limit is exceeded."""

    def __init__(
        self,
        message: str = "API rate limit exceeded",
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message)


class InvalidSymbolError(AlphaVantageError):
    """Raised when an invalid or unknown symbol is requested."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        super().__init__(f"Invalid or unknown symbol: {symbol}")
