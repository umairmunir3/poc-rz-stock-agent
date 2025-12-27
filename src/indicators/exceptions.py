"""Custom exceptions for indicators module."""


class IndicatorError(Exception):
    """Base exception for technical indicator errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class InsufficientDataError(IndicatorError):
    """Raised when there is not enough data to calculate an indicator."""

    def __init__(self, required: int, available: int, indicator: str) -> None:
        self.required = required
        self.available = available
        self.indicator = indicator
        super().__init__(
            f"Insufficient data for {indicator}: requires {required} rows, got {available}"
        )


class InvalidDataError(IndicatorError):
    """Raised when input data is invalid or contains unexpected values."""

    def __init__(self, message: str, column: str | None = None) -> None:
        self.column = column
        if column:
            message = f"Invalid data in column '{column}': {message}"
        super().__init__(message)
