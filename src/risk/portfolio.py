"""Portfolio Risk Manager - manages overall portfolio risk and limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING, Literal

import pandas as pd

from src.strategies.base import Signal

if TYPE_CHECKING:
    from src.data.storage import StorageManager


@dataclass
class RiskConfig:
    """Configuration for portfolio risk management.

    Attributes:
        max_positions: Maximum number of open positions.
        max_sector_positions: Maximum positions per sector.
        max_correlated_positions: Maximum correlated positions.
        correlation_threshold: Threshold for considering positions correlated.
        max_portfolio_heat: Maximum sum of position risks (as decimal).
        daily_loss_limit: Maximum daily loss (as decimal).
        weekly_loss_limit: Maximum weekly loss (as decimal).
        max_drawdown: Maximum drawdown before halting (as decimal).
    """

    max_positions: int = 5
    max_sector_positions: int = 2
    max_correlated_positions: int = 3
    correlation_threshold: float = 0.7
    max_portfolio_heat: float = 0.05
    daily_loss_limit: float = 0.03
    weekly_loss_limit: float = 0.05
    max_drawdown: float = 0.10


@dataclass
class Position:
    """Represents an open position.

    Attributes:
        symbol: Stock ticker symbol.
        sector: Sector of the stock.
        entry_price: Entry price per share.
        current_price: Current price per share.
        shares: Number of shares.
        stop_loss: Stop loss price.
        direction: Trade direction (LONG or SHORT).
        entry_date: When position was opened.
        risk_amount: Dollar amount at risk.
        risk_percent: Risk as percentage of account.
    """

    symbol: str
    sector: str
    entry_price: float
    current_price: float
    shares: int
    stop_loss: float
    direction: Literal["LONG", "SHORT"]
    entry_date: date
    risk_amount: float
    risk_percent: float

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.direction == "LONG":
            return (self.current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - self.current_price) * self.shares

    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage of position value."""
        position_value = self.entry_price * self.shares
        if position_value == 0:
            return 0.0
        return (self.unrealized_pnl / position_value) * 100


@dataclass
class ValidationResult:
    """Result of trade validation.

    Attributes:
        allowed: Whether the trade is allowed.
        checks_passed: List of checks that passed.
        checks_failed: List of checks that failed.
        warnings: List of warnings (approaching limits).
    """

    allowed: bool
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PortfolioRiskManager:
    """Manages overall portfolio risk and enforces limits.

    This class tracks open positions, calculates portfolio heat,
    monitors correlations, and enforces various risk limits.

    Example:
        >>> config = RiskConfig(max_positions=5, daily_loss_limit=0.03)
        >>> manager = PortfolioRiskManager(config)
        >>> can_trade, reason = manager.can_open_position(signal)
    """

    def __init__(
        self,
        config: RiskConfig,
        storage: StorageManager | None = None,
        account_equity: float = 100000.0,
    ) -> None:
        """Initialize the portfolio risk manager.

        Args:
            config: Risk configuration parameters.
            storage: StorageManager for fetching price data.
            account_equity: Total account equity.
        """
        self.config = config
        self.storage = storage
        self.account_equity = account_equity
        self._peak_equity = account_equity

        # Internal state
        self._positions: list[Position] = []
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._last_daily_reset: date = date.today()
        self._last_weekly_reset: date = date.today()
        self._correlation_cache: dict[str, pd.DataFrame] = {}

    @property
    def positions(self) -> list[Position]:
        """Get current open positions."""
        return self._positions

    def set_positions(self, positions: list[Position]) -> None:
        """Set open positions (for testing).

        Args:
            positions: List of Position objects.
        """
        self._positions = positions

    def add_position(self, position: Position) -> None:
        """Add a new position.

        Args:
            position: Position to add.
        """
        self._positions.append(position)

    def remove_position(self, symbol: str) -> bool:
        """Remove a position by symbol.

        Args:
            symbol: Symbol to remove.

        Returns:
            True if position was removed, False if not found.
        """
        for i, pos in enumerate(self._positions):
            if pos.symbol == symbol:
                del self._positions[i]
                return True
        return False

    def set_daily_pnl(self, pnl: float) -> None:
        """Set daily P&L (for testing).

        Args:
            pnl: Daily P&L value.
        """
        self._daily_pnl = pnl

    def set_weekly_pnl(self, pnl: float) -> None:
        """Set weekly P&L (for testing).

        Args:
            pnl: Weekly P&L value.
        """
        self._weekly_pnl = pnl

    def set_peak_equity(self, equity: float) -> None:
        """Set peak equity (for testing).

        Args:
            equity: Peak equity value.
        """
        self._peak_equity = equity

    def update_equity(self, current_equity: float) -> None:
        """Update current equity and track peak.

        Args:
            current_equity: Current account equity.
        """
        self.account_equity = current_equity
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def can_open_position(self, signal: Signal) -> tuple[bool, str]:
        """Check if a new position can be opened.

        Args:
            signal: Trading signal to evaluate.

        Returns:
            Tuple of (allowed, reason_if_not_allowed).
        """
        validation = self.validate_trade(signal)
        if not validation.allowed:
            return False, "; ".join(validation.checks_failed)
        return True, "All checks passed"

    def get_portfolio_heat(self) -> float:
        """Get sum of all open position risks.

        Returns:
            Total portfolio heat as decimal (e.g., 0.05 = 5%).
        """
        if not self._positions:
            return 0.0

        total_risk = sum(pos.risk_percent for pos in self._positions)
        return total_risk / 100  # Convert from percent to decimal

    def get_sector_exposure(self) -> dict[str, int]:
        """Get count of positions per sector.

        Returns:
            Dictionary mapping sector to position count.
        """
        exposure: dict[str, int] = {}
        for pos in self._positions:
            exposure[pos.sector] = exposure.get(pos.sector, 0) + 1
        return exposure

    def get_position_count(self) -> int:
        """Get total number of open positions.

        Returns:
            Number of open positions.
        """
        return len(self._positions)

    def calculate_correlation_matrix(
        self,
        symbols: list[str],
        lookback: int = 60,
    ) -> pd.DataFrame:
        """Calculate correlation matrix for symbols.

        Args:
            symbols: List of symbols.
            lookback: Number of days for correlation calculation.

        Returns:
            DataFrame with correlation matrix.
        """
        if not symbols or len(symbols) < 2:
            return pd.DataFrame()

        # Check cache
        cache_key = ",".join(sorted(symbols)) + f"_{lookback}"
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]

        # Build returns matrix
        returns_data: dict[str, pd.Series] = {}
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback * 2)  # Extra days for weekends

        for symbol in symbols:
            try:
                if self.storage:
                    # Use storage to get data (async, but we'll handle sync for now)
                    import asyncio

                    df = asyncio.get_event_loop().run_until_complete(
                        self.storage.get_daily_bars(symbol, start_date, end_date)
                    )
                    if not df.empty:
                        returns = df["close"].pct_change().dropna().tail(lookback)
                        returns_data[symbol] = returns
            except Exception:
                pass

        if len(returns_data) < 2:
            return pd.DataFrame()

        # Build DataFrame and calculate correlation
        returns_df = pd.DataFrame(returns_data)
        correlation_matrix = returns_df.corr()

        # Cache result
        self._correlation_cache[cache_key] = correlation_matrix
        return correlation_matrix

    def get_correlated_positions(
        self,
        symbol: str,
        symbol_sector: str | None = None,
    ) -> list[str]:
        """Get open positions correlated with a symbol.

        Args:
            symbol: Symbol to check against.
            symbol_sector: Sector of the symbol (for sector-based correlation).

        Returns:
            List of correlated position symbols.
        """
        correlated: list[str] = []

        # Get all position symbols including the new one
        position_symbols = [pos.symbol for pos in self._positions]
        if symbol in position_symbols:
            position_symbols.remove(symbol)

        if not position_symbols:
            return correlated

        # Calculate correlations if storage is available
        if self.storage:
            all_symbols = position_symbols + [symbol]
            corr_matrix = self.calculate_correlation_matrix(all_symbols)

            if not corr_matrix.empty and symbol in corr_matrix.columns:
                for pos_symbol in position_symbols:
                    if pos_symbol in corr_matrix.columns:
                        corr_value = corr_matrix.loc[symbol, pos_symbol]
                        correlation = abs(float(corr_value))  # type: ignore[arg-type]
                        if correlation >= self.config.correlation_threshold:
                            correlated.append(pos_symbol)

        # Also check sector-based correlation
        if symbol_sector:
            for pos in self._positions:
                if pos.sector == symbol_sector and pos.symbol not in correlated:
                    correlated.append(pos.symbol)

        return correlated

    def get_daily_pnl(self) -> float:
        """Get today's P&L.

        Returns:
            Daily P&L in dollars.
        """
        self._check_daily_reset()
        return self._daily_pnl

    def get_weekly_pnl(self) -> float:
        """Get this week's P&L.

        Returns:
            Weekly P&L in dollars.
        """
        self._check_weekly_reset()
        return self._weekly_pnl

    def is_daily_limit_hit(self) -> bool:
        """Check if daily loss limit is hit.

        Returns:
            True if daily loss limit is reached.
        """
        daily_loss_amount = self.account_equity * self.config.daily_loss_limit
        return self.get_daily_pnl() <= -daily_loss_amount

    def is_weekly_limit_hit(self) -> bool:
        """Check if weekly loss limit is hit.

        Returns:
            True if weekly loss limit is reached.
        """
        weekly_loss_amount = self.account_equity * self.config.weekly_loss_limit
        return self.get_weekly_pnl() <= -weekly_loss_amount

    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak.

        Returns:
            Drawdown as decimal (e.g., 0.10 = 10%).
        """
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.account_equity) / self._peak_equity

    def get_peak_equity(self) -> float:
        """Get peak equity value.

        Returns:
            Peak equity in dollars.
        """
        return self._peak_equity

    def is_max_drawdown_hit(self) -> bool:
        """Check if maximum drawdown is hit.

        Returns:
            True if max drawdown limit is reached.
        """
        return self.get_current_drawdown() >= self.config.max_drawdown

    def validate_trade(self, signal: Signal) -> ValidationResult:
        """Validate a trade against all risk rules.

        Args:
            signal: Trading signal to validate.

        Returns:
            ValidationResult with detailed check results.
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        warnings: list[str] = []

        # Check 1: Maximum positions
        current_positions = self.get_position_count()
        if current_positions >= self.config.max_positions:
            checks_failed.append(
                f"Max positions reached: {current_positions}/{self.config.max_positions}"
            )
        else:
            checks_passed.append(
                f"Position count OK: {current_positions}/{self.config.max_positions}"
            )
            if current_positions >= self.config.max_positions * 0.8:
                warnings.append(
                    f"Approaching position limit: {current_positions}/{self.config.max_positions}"
                )

        # Check 2: Sector limit
        sector = signal.metadata.get("sector", "Unknown")
        sector_exposure = self.get_sector_exposure()
        sector_count = sector_exposure.get(sector, 0)

        if sector_count >= self.config.max_sector_positions:
            checks_failed.append(
                f"Max {sector} positions reached: {sector_count}/{self.config.max_sector_positions}"
            )
        else:
            checks_passed.append(
                f"Sector {sector} exposure OK: {sector_count}/{self.config.max_sector_positions}"
            )
            if sector_count >= self.config.max_sector_positions * 0.8:
                warnings.append(
                    f"Approaching {sector} limit: {sector_count}/{self.config.max_sector_positions}"
                )

        # Check 3: Correlation limit
        correlated = self.get_correlated_positions(signal.symbol, sector)
        if len(correlated) >= self.config.max_correlated_positions:
            checks_failed.append(
                f"Too many correlated positions: {len(correlated)}/{self.config.max_correlated_positions} "
                f"({', '.join(correlated)})"
            )
        else:
            checks_passed.append(
                f"Correlation OK: {len(correlated)}/{self.config.max_correlated_positions}"
            )
            if len(correlated) >= self.config.max_correlated_positions * 0.8:
                warnings.append(
                    f"Approaching correlation limit: {len(correlated)} correlated positions"
                )

        # Check 4: Portfolio heat
        current_heat = self.get_portfolio_heat()
        new_risk = signal.risk_percent / 100  # Convert to decimal
        projected_heat = current_heat + new_risk

        if projected_heat > self.config.max_portfolio_heat:
            checks_failed.append(
                f"Portfolio heat exceeded: {projected_heat:.1%} > {self.config.max_portfolio_heat:.1%}"
            )
        else:
            checks_passed.append(
                f"Portfolio heat OK: {projected_heat:.1%}/{self.config.max_portfolio_heat:.1%}"
            )
            if projected_heat >= self.config.max_portfolio_heat * 0.8:
                warnings.append(
                    f"Approaching heat limit: {projected_heat:.1%}/{self.config.max_portfolio_heat:.1%}"
                )

        # Check 5: Daily loss limit
        if self.is_daily_limit_hit():
            checks_failed.append(f"Daily loss limit hit: ${abs(self.get_daily_pnl()):.2f}")
        else:
            checks_passed.append("Daily loss limit OK")
            daily_loss_amount = self.account_equity * self.config.daily_loss_limit
            if self.get_daily_pnl() <= -daily_loss_amount * 0.8:
                warnings.append(f"Approaching daily loss limit: ${abs(self.get_daily_pnl()):.2f}")

        # Check 6: Weekly loss limit
        if self.is_weekly_limit_hit():
            checks_failed.append(f"Weekly loss limit hit: ${abs(self.get_weekly_pnl()):.2f}")
        else:
            checks_passed.append("Weekly loss limit OK")
            weekly_loss_amount = self.account_equity * self.config.weekly_loss_limit
            if self.get_weekly_pnl() <= -weekly_loss_amount * 0.8:
                warnings.append(f"Approaching weekly loss limit: ${abs(self.get_weekly_pnl()):.2f}")

        # Check 7: Maximum drawdown
        if self.is_max_drawdown_hit():
            checks_failed.append(f"Max drawdown hit: {self.get_current_drawdown():.1%}")
        else:
            checks_passed.append(
                f"Drawdown OK: {self.get_current_drawdown():.1%}/{self.config.max_drawdown:.1%}"
            )
            if self.get_current_drawdown() >= self.config.max_drawdown * 0.8:
                warnings.append(f"Approaching max drawdown: {self.get_current_drawdown():.1%}")

        allowed = len(checks_failed) == 0

        return ValidationResult(
            allowed=allowed,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
        )

    def _check_daily_reset(self) -> None:
        """Reset daily P&L if new day."""
        today = date.today()
        if today > self._last_daily_reset:
            self._daily_pnl = 0.0
            self._last_daily_reset = today

    def _check_weekly_reset(self) -> None:
        """Reset weekly P&L if new week."""
        today = date.today()
        days_since_reset = (today - self._last_weekly_reset).days
        if days_since_reset >= 7:
            self._weekly_pnl = 0.0
            self._last_weekly_reset = today

    def record_trade_pnl(self, pnl: float) -> None:
        """Record P&L from a closed trade.

        Args:
            pnl: P&L amount (positive for profit, negative for loss).
        """
        self._check_daily_reset()
        self._check_weekly_reset()

        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self.account_equity += pnl

        # Update peak equity if we have a new high
        if self.account_equity > self._peak_equity:
            self._peak_equity = self.account_equity

    def get_portfolio_summary(self) -> dict:
        """Get a summary of portfolio risk metrics.

        Returns:
            Dictionary with portfolio risk summary.
        """
        return {
            "positions_count": self.get_position_count(),
            "max_positions": self.config.max_positions,
            "portfolio_heat": self.get_portfolio_heat(),
            "max_heat": self.config.max_portfolio_heat,
            "sector_exposure": self.get_sector_exposure(),
            "daily_pnl": self.get_daily_pnl(),
            "daily_loss_limit": self.account_equity * self.config.daily_loss_limit,
            "weekly_pnl": self.get_weekly_pnl(),
            "weekly_loss_limit": self.account_equity * self.config.weekly_loss_limit,
            "current_drawdown": self.get_current_drawdown(),
            "max_drawdown": self.config.max_drawdown,
            "peak_equity": self.get_peak_equity(),
            "current_equity": self.account_equity,
        }
