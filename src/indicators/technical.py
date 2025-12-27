"""Technical indicators calculation module."""

from __future__ import annotations

import pandas as pd

from src.indicators.exceptions import InsufficientDataError, InvalidDataError


class TechnicalIndicators:
    """Calculate technical indicators for OHLCV data.

    This class provides methods to calculate various technical indicators
    commonly used in swing trading strategies.

    Attributes:
        df: DataFrame with OHLCV data (must have open, high, low, close, volume columns)
    """

    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
                Index should be datetime or date

        Raises:
            InvalidDataError: If required columns are missing or data is invalid
        """
        self._validate_input(df)
        self.df = df.copy()

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame has required columns and valid data."""
        if df.empty:
            raise InvalidDataError("DataFrame is empty")

        # Check for required columns (case-insensitive)
        df_columns = {col.lower() for col in df.columns}
        missing = self.REQUIRED_COLUMNS - df_columns
        if missing:
            raise InvalidDataError(f"Missing required columns: {missing}")

        # Check for NaN in critical columns
        for col in ["close", "high", "low"]:
            # Find the actual column name (case-insensitive)
            actual_col = next((c for c in df.columns if c.lower() == col), col)
            if df[actual_col].isna().all():
                raise InvalidDataError("Column contains all NaN values", column=col)

    def _check_sufficient_data(self, required: int, indicator: str) -> None:
        """Check if DataFrame has enough rows for the indicator."""
        if len(self.df) < required:
            raise InsufficientDataError(
                required=required, available=len(self.df), indicator=indicator
            )

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI).

        RSI measures the magnitude of recent price changes to evaluate
        overbought or oversold conditions.

        Args:
            period: Lookback period for RSI calculation (default: 14)

        Returns:
            Series with RSI values (0-100)

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period + 1, f"RSI({period})")

        close = self.df["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        # Use exponential moving average (Wilder's smoothing)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Handle division by zero (when avg_loss is 0)
        rsi = rsi.fillna(100)  # If no losses, RSI = 100
        rsi = rsi.replace([float("inf"), float("-inf")], 100)

        return rsi

    def calculate_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Moving Average Convergence Divergence (MACD).

        MACD shows the relationship between two exponential moving averages.

        Args:
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line EMA period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        required = slow + signal
        self._check_sufficient_data(required, f"MACD({fast},{slow},{signal})")

        close = self.df["close"]

        # Calculate EMAs
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_ema(self, period: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA).

        Args:
            period: EMA period

        Returns:
            Series with EMA values

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period, f"EMA({period})")

        return self.df["close"].ewm(span=period, adjust=False).mean()

    def calculate_multiple_emas(self, periods: list[int] | None = None) -> dict[int, pd.Series]:
        """Calculate multiple EMAs at once.

        Args:
            periods: List of EMA periods (default: [9, 21, 50, 200])

        Returns:
            Dictionary mapping period to EMA Series

        Raises:
            InsufficientDataError: If not enough data for longest EMA
        """
        if periods is None:
            periods = [9, 21, 50, 200]

        max_period = max(periods)
        self._check_sufficient_data(max_period, f"EMA({max_period})")

        return {period: self.calculate_ema(period) for period in periods}

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR).

        ATR measures market volatility by decomposing the entire range
        of an asset price for that period.

        Args:
            period: ATR period (default: 14)

        Returns:
            Series with ATR values

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period + 1, f"ATR({period})")

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # True Range components
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        # True Range is the maximum of the three
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR is the smoothed average of True Range (Wilder's smoothing)
        atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        return atr

    def calculate_bollinger(
        self, period: int = 20, std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands.

        Bollinger Bands consist of a middle band (SMA) with an upper and
        lower band at standard deviation intervals.

        Args:
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2.0)

        Returns:
            Tuple of (upper_band, middle_band, lower_band)

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period, f"Bollinger({period})")

        close = self.df["close"]

        # Middle band (SMA)
        middle = close.rolling(window=period).mean()

        # Standard deviation
        std = close.rolling(window=period).std()

        # Upper and lower bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def calculate_volume_sma(self, period: int = 20) -> pd.Series:
        """Calculate Simple Moving Average of volume.

        Args:
            period: SMA period (default: 20)

        Returns:
            Series with volume SMA values

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period, f"Volume SMA({period})")

        return self.df["volume"].rolling(window=period).mean()

    def calculate_relative_volume(self, period: int = 20) -> pd.Series:
        """Calculate Relative Volume (RVOL).

        Relative volume compares current volume to the average volume
        over a specified period.

        Args:
            period: Period for average volume calculation (default: 20)

        Returns:
            Series with relative volume values (1.0 = average)

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period, f"Relative Volume({period})")

        volume = self.df["volume"]
        avg_volume = volume.rolling(window=period).mean()

        # Avoid division by zero
        relative_volume = volume / avg_volume.replace(0, float("nan"))

        return relative_volume

    def calculate_sma(self, period: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA).

        Args:
            period: SMA period

        Returns:
            Series with SMA values

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period, f"SMA({period})")

        return self.df["close"].rolling(window=period).mean()

    def calculate_stochastic(
        self, k_period: int = 14, d_period: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator.

        Args:
            k_period: Lookback period for %K (default: 14)
            d_period: Smoothing period for %D (default: 3)

        Returns:
            Tuple of (%K, %D)

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(k_period + d_period, f"Stochastic({k_period},{d_period})")

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # Lowest low and highest high over k_period
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        # %K
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # %D (SMA of %K)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def calculate_adx(self, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX).

        ADX measures trend strength regardless of direction.

        Args:
            period: ADX period (default: 14)

        Returns:
            Series with ADX values

        Raises:
            InsufficientDataError: If not enough data for calculation
        """
        self._check_sufficient_data(period * 2, f"ADX({period})")

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # Smoothed values (Wilder's smoothing)
        atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        plus_di = 100 * (
            plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
        )
        minus_di = 100 * (
            minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
        )

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        return adx

    def calculate_obv(self) -> pd.Series:
        """Calculate On-Balance Volume (OBV).

        OBV uses volume flow to predict changes in stock price.

        Returns:
            Series with OBV values
        """
        close = self.df["close"]
        volume = self.df["volume"]

        # Calculate direction
        direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        # OBV is cumulative sum of signed volume
        obv = (volume * direction).cumsum()

        return obv

    def calculate_all(self) -> pd.DataFrame:
        """Calculate all standard indicators and add to DataFrame.

        Adds the following columns:
        - rsi_14: RSI with period 14
        - macd, macd_signal, macd_hist: MACD components
        - ema_9, ema_21, ema_50, ema_200: EMAs
        - atr_14: ATR with period 14
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - volume_sma_20: 20-day volume SMA
        - relative_volume: Current volume / 20-day average

        Returns:
            DataFrame with all indicators added

        Raises:
            InsufficientDataError: If not enough data for EMA 200
        """
        # Need at least 200 rows for EMA 200
        self._check_sufficient_data(200, "calculate_all (EMA 200)")

        result = self.df.copy()

        # RSI
        result["rsi_14"] = self.calculate_rsi(14)

        # MACD
        macd, signal, hist = self.calculate_macd()
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist

        # EMAs
        emas = self.calculate_multiple_emas([9, 21, 50, 200])
        result["ema_9"] = emas[9]
        result["ema_21"] = emas[21]
        result["ema_50"] = emas[50]
        result["ema_200"] = emas[200]

        # ATR
        result["atr_14"] = self.calculate_atr(14)

        # Bollinger Bands
        upper, middle, lower = self.calculate_bollinger()
        result["bb_upper"] = upper
        result["bb_middle"] = middle
        result["bb_lower"] = lower

        # Volume indicators
        result["volume_sma_20"] = self.calculate_volume_sma(20)
        result["relative_volume"] = self.calculate_relative_volume(20)

        return result
