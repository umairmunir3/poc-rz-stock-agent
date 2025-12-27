"""Support and resistance level detection module."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd

from src.indicators.exceptions import InsufficientDataError, InvalidDataError


class LevelType(str, Enum):
    """Type of price level."""

    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"


@dataclass
class PriceLevel:
    """Represents a support or resistance price level.

    Attributes:
        price: The price level
        strength: Number of times price has touched this level
        level_type: Whether this is support or resistance
        last_touch: Date of the most recent touch
    """

    price: float
    strength: int
    level_type: LevelType
    last_touch: date

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.strength < 1:
            raise ValueError("Strength must be at least 1")
        if self.price <= 0:
            raise ValueError("Price must be positive")


@dataclass
class SupportResistanceLevels:
    """Container for detected support and resistance levels.

    Attributes:
        support_levels: List of support levels sorted by strength (descending)
        resistance_levels: List of resistance levels sorted by strength (descending)
        nearest_support: Closest support level below current price
        nearest_resistance: Closest resistance level above current price
    """

    support_levels: list[PriceLevel] = field(default_factory=list)
    resistance_levels: list[PriceLevel] = field(default_factory=list)
    nearest_support: PriceLevel | None = None
    nearest_resistance: PriceLevel | None = None

    @property
    def all_levels(self) -> list[PriceLevel]:
        """Return all levels combined."""
        return self.support_levels + self.resistance_levels

    def has_levels(self) -> bool:
        """Check if any levels were detected."""
        return len(self.support_levels) > 0 or len(self.resistance_levels) > 0


class SupportResistanceDetector:
    """Detects support and resistance levels from price data.

    This class identifies key price levels where the price has historically
    reversed or consolidated, which can indicate future support or resistance.

    Attributes:
        df: DataFrame with OHLCV data
        lookback: Number of periods to analyze
    """

    REQUIRED_COLUMNS = {"high", "low", "close"}
    MIN_DATA_POINTS = 20

    def __init__(self, df: pd.DataFrame, lookback: int = 60) -> None:
        """Initialize with price data.

        Args:
            df: DataFrame with high, low, close columns
            lookback: Number of periods to analyze for level detection

        Raises:
            InvalidDataError: If required columns are missing
            InsufficientDataError: If not enough data points
        """
        self._validate_input(df)
        self.lookback = lookback
        self.df = df.tail(lookback).copy() if len(df) > lookback else df.copy()
        self._current_price = float(self.df["close"].iloc[-1])

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if df.empty:
            raise InvalidDataError("DataFrame is empty")

        df_columns = {col.lower() for col in df.columns}
        missing = self.REQUIRED_COLUMNS - df_columns
        if missing:
            raise InvalidDataError(f"Missing required columns: {missing}")

        if len(df) < self.MIN_DATA_POINTS:
            raise InsufficientDataError(
                required=self.MIN_DATA_POINTS,
                available=len(df),
                indicator="SupportResistanceDetector",
            )

    def find_pivot_highs(self, left: int = 5, right: int = 5) -> pd.Series:
        """Find pivot high points in the price data.

        A pivot high is a point where the high is greater than the highs
        of 'left' bars to the left and 'right' bars to the right.

        Args:
            left: Number of bars to the left to compare
            right: Number of bars to the right to compare

        Returns:
            Boolean Series with True at pivot high points
        """
        highs = self.df["high"]
        pivot_highs = pd.Series(False, index=self.df.index)

        for i in range(left, len(highs) - right):
            current_high = highs.iloc[i]

            # Check if current high is greater than all surrounding highs
            left_highs = highs.iloc[i - left : i]
            right_highs = highs.iloc[i + 1 : i + right + 1]

            if (current_high > left_highs).all() and (current_high > right_highs).all():
                pivot_highs.iloc[i] = True

        return pivot_highs

    def find_pivot_lows(self, left: int = 5, right: int = 5) -> pd.Series:
        """Find pivot low points in the price data.

        A pivot low is a point where the low is less than the lows
        of 'left' bars to the left and 'right' bars to the right.

        Args:
            left: Number of bars to the left to compare
            right: Number of bars to the right to compare

        Returns:
            Boolean Series with True at pivot low points
        """
        lows = self.df["low"]
        pivot_lows = pd.Series(False, index=self.df.index)

        for i in range(left, len(lows) - right):
            current_low = lows.iloc[i]

            # Check if current low is less than all surrounding lows
            left_lows = lows.iloc[i - left : i]
            right_lows = lows.iloc[i + 1 : i + right + 1]

            if (current_low < left_lows).all() and (current_low < right_lows).all():
                pivot_lows.iloc[i] = True

        return pivot_lows

    def cluster_levels(
        self,
        prices: list[tuple[float, date]],
        tolerance_pct: float = 0.02,
        level_type: LevelType = LevelType.SUPPORT,
    ) -> list[PriceLevel]:
        """Cluster nearby price levels into zones.

        Groups pivot points that are within tolerance_pct of each other
        into a single level, tracking the number of touches.

        Args:
            prices: List of (price, date) tuples for pivot points
            tolerance_pct: Percentage tolerance for clustering (default 2%)
            level_type: Type of level (SUPPORT or RESISTANCE)

        Returns:
            List of PriceLevel objects representing clustered zones
        """
        if not prices:
            return []

        # Sort by price
        sorted_prices = sorted(prices, key=lambda x: x[0])

        clusters: list[list[tuple[float, date]]] = []
        current_cluster: list[tuple[float, date]] = [sorted_prices[0]]

        for price, touch_date in sorted_prices[1:]:
            # Get the average price of current cluster
            cluster_avg = sum(p for p, _ in current_cluster) / len(current_cluster)

            # Check if this price is within tolerance of the cluster
            if abs(price - cluster_avg) / cluster_avg <= tolerance_pct:
                current_cluster.append((price, touch_date))
            else:
                # Start a new cluster
                clusters.append(current_cluster)
                current_cluster = [(price, touch_date)]

        # Don't forget the last cluster
        clusters.append(current_cluster)

        # Convert clusters to PriceLevel objects
        levels = []
        for cluster in clusters:
            avg_price = sum(p for p, _ in cluster) / len(cluster)
            latest_touch = max(d for _, d in cluster)
            levels.append(
                PriceLevel(
                    price=avg_price,
                    strength=len(cluster),
                    level_type=level_type,
                    last_touch=latest_touch,
                )
            )

        return levels

    def detect_levels(self, min_touches: int = 2) -> SupportResistanceLevels:
        """Detect support and resistance levels.

        Finds pivot points and clusters them into support and resistance
        levels based on price proximity.

        Args:
            min_touches: Minimum number of touches required for a level

        Returns:
            SupportResistanceLevels object with detected levels
        """
        # Find pivot points
        pivot_highs = self.find_pivot_highs()
        pivot_lows = self.find_pivot_lows()

        # Extract prices and dates for pivot points
        high_prices: list[tuple[float, date]] = []
        low_prices: list[tuple[float, date]] = []

        for idx in self.df.index[pivot_highs]:
            price = float(self.df.loc[idx, "high"])
            touch_date = idx.date() if hasattr(idx, "date") else idx
            high_prices.append((price, touch_date))

        for idx in self.df.index[pivot_lows]:
            price = float(self.df.loc[idx, "low"])
            touch_date = idx.date() if hasattr(idx, "date") else idx
            low_prices.append((price, touch_date))

        # Cluster into levels
        resistance_levels = self.cluster_levels(high_prices, level_type=LevelType.RESISTANCE)
        support_levels = self.cluster_levels(low_prices, level_type=LevelType.SUPPORT)

        # Filter by minimum touches
        resistance_levels = [level for level in resistance_levels if level.strength >= min_touches]
        support_levels = [level for level in support_levels if level.strength >= min_touches]

        # Sort by strength (descending)
        resistance_levels.sort(key=lambda x: x.strength, reverse=True)
        support_levels.sort(key=lambda x: x.strength, reverse=True)

        # Find nearest levels relative to current price
        nearest_support = self._find_nearest_support(support_levels)
        nearest_resistance = self._find_nearest_resistance(resistance_levels)

        return SupportResistanceLevels(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

    def _find_nearest_support(self, support_levels: list[PriceLevel]) -> PriceLevel | None:
        """Find the nearest support level below current price."""
        below_price = [level for level in support_levels if level.price < self._current_price]
        if not below_price:
            return None
        # Return the one closest to current price
        return max(below_price, key=lambda x: x.price)

    def _find_nearest_resistance(self, resistance_levels: list[PriceLevel]) -> PriceLevel | None:
        """Find the nearest resistance level above current price."""
        above_price = [level for level in resistance_levels if level.price > self._current_price]
        if not above_price:
            return None
        # Return the one closest to current price
        return min(above_price, key=lambda x: x.price)

    def is_at_support(self, price: float, tolerance_pct: float = 0.01) -> bool:
        """Check if price is at a support level.

        Args:
            price: Price to check
            tolerance_pct: Percentage tolerance (default 1%)

        Returns:
            True if price is within tolerance of a support level
        """
        levels = self.detect_levels()
        for level in levels.support_levels:
            if abs(price - level.price) / level.price <= tolerance_pct:
                return True
        return False

    def is_at_resistance(self, price: float, tolerance_pct: float = 0.01) -> bool:
        """Check if price is at a resistance level.

        Args:
            price: Price to check
            tolerance_pct: Percentage tolerance (default 1%)

        Returns:
            True if price is within tolerance of a resistance level
        """
        levels = self.detect_levels()
        for level in levels.resistance_levels:
            if abs(price - level.price) / level.price <= tolerance_pct:
                return True
        return False

    def distance_to_support(self, price: float) -> float:
        """Calculate percentage distance to nearest support.

        Args:
            price: Current price

        Returns:
            Percentage distance to nearest support (positive value)
            Returns float('inf') if no support levels found
        """
        levels = self.detect_levels()
        if levels.nearest_support is None:
            # Check all support levels for any below price
            below = [level for level in levels.support_levels if level.price < price]
            if not below:
                return float("inf")
            nearest = max(below, key=lambda x: x.price)
            return ((price - nearest.price) / price) * 100

        return ((price - levels.nearest_support.price) / price) * 100

    def distance_to_resistance(self, price: float) -> float:
        """Calculate percentage distance to nearest resistance.

        Args:
            price: Current price

        Returns:
            Percentage distance to nearest resistance (positive value)
            Returns float('inf') if no resistance levels found
        """
        levels = self.detect_levels()
        if levels.nearest_resistance is None:
            # Check all resistance levels for any above price
            above = [level for level in levels.resistance_levels if level.price > price]
            if not above:
                return float("inf")
            nearest = min(above, key=lambda x: x.price)
            return ((nearest.price - price) / price) * 100

        return ((levels.nearest_resistance.price - price) / price) * 100

    def get_level_at_price(self, price: float, tolerance_pct: float = 0.02) -> PriceLevel | None:
        """Get the level at or near a specific price.

        Args:
            price: Price to look up
            tolerance_pct: Percentage tolerance for matching

        Returns:
            PriceLevel if found, None otherwise
        """
        levels = self.detect_levels()
        for level in levels.all_levels:
            if abs(price - level.price) / level.price <= tolerance_pct:
                return level
        return None

    def get_levels_in_range(self, low_price: float, high_price: float) -> list[PriceLevel]:
        """Get all levels within a price range.

        Args:
            low_price: Lower bound of range
            high_price: Upper bound of range

        Returns:
            List of PriceLevel objects within the range
        """
        levels = self.detect_levels()
        return [level for level in levels.all_levels if low_price <= level.price <= high_price]
