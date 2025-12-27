"""Circuit Breaker System - automated safeguards that halt trading."""

from __future__ import annotations

import logging
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from src.risk.portfolio import PortfolioRiskManager

logger = logging.getLogger(__name__)


class BreakerState(str, Enum):
    """Circuit breaker state."""

    OK = "OK"
    WARNING = "WARNING"
    TRIGGERED = "TRIGGERED"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers.

    Attributes:
        drawdown_warning_threshold: Drawdown level for warning (default 7%).
        drawdown_halt_threshold: Drawdown level for trading halt (default 10%).
        losing_streak_limit: Consecutive losses before pause (default 5).
        volatility_warning_vix: VIX level for warning (default 25).
        volatility_halt_vix: VIX level for halt (default 35).
        market_regime_sma: SMA period for market regime (default 200).
        losing_streak_pause_hours: Hours to pause after losing streak (default 24).
    """

    drawdown_warning_threshold: float = 0.07
    drawdown_halt_threshold: float = 0.10
    losing_streak_limit: int = 5
    volatility_warning_vix: float = 25.0
    volatility_halt_vix: float = 35.0
    market_regime_sma: int = 200
    losing_streak_pause_hours: int = 24


@dataclass
class BreakerStatus:
    """Status of a circuit breaker.

    Attributes:
        breaker_name: Name of the breaker.
        status: Current status (OK, WARNING, TRIGGERED).
        action_required: Description of required action.
        can_trade: Whether trading is allowed.
        position_size_multiplier: Multiplier for position sizes (1.0 = normal).
        message: Human-readable status message.
    """

    breaker_name: str
    status: Literal["OK", "WARNING", "TRIGGERED"]
    action_required: str
    can_trade: bool
    position_size_multiplier: float
    message: str


@dataclass
class SystemStatus:
    """Combined status of all circuit breakers.

    Attributes:
        overall_status: Most restrictive status.
        can_trade: Whether trading is allowed.
        position_size_multiplier: Combined position size multiplier.
        breaker_statuses: Individual breaker statuses.
        warnings: List of warning messages.
        actions_required: List of required actions.
    """

    overall_status: Literal["OK", "WARNING", "TRIGGERED"]
    can_trade: bool
    position_size_multiplier: float
    breaker_statuses: list[BreakerStatus]
    warnings: list[str] = field(default_factory=list)
    actions_required: list[str] = field(default_factory=list)


@dataclass
class BreakerEvent:
    """Event logged when a breaker state changes.

    Attributes:
        breaker_name: Name of the breaker.
        event_type: Type of event.
        old_status: Previous status.
        new_status: New status.
        timestamp: When the event occurred.
        details: Additional details.
        user: User who triggered the event (if applicable).
    """

    breaker_name: str
    event_type: str
    old_status: str
    new_status: str
    timestamp: datetime
    details: str
    user: str | None = None


class CircuitBreaker(ABC):
    """Abstract base class for circuit breakers."""

    def __init__(self, name: str) -> None:
        """Initialize the breaker.

        Args:
            name: Name of the breaker.
        """
        self.name = name
        self._acknowledged = False
        self._manually_reset = False
        self._override_until: datetime | None = None

    @abstractmethod
    def check(self, *args: object, **kwargs: object) -> BreakerStatus:
        """Check the breaker status.

        Args:
            **kwargs: Breaker-specific parameters.

        Returns:
            Current status of the breaker.
        """
        pass

    def acknowledge(self) -> None:
        """Acknowledge a warning."""
        self._acknowledged = True
        logger.info(f"Breaker {self.name} warning acknowledged")

    def reset(self) -> None:
        """Reset the breaker after a halt."""
        self._manually_reset = True
        logger.info(f"Breaker {self.name} manually reset")

    def override(self, duration_hours: int) -> None:
        """Override the breaker temporarily.

        Args:
            duration_hours: Hours to override.
        """
        self._override_until = datetime.now() + timedelta(hours=duration_hours)
        logger.warning(f"Breaker {self.name} overridden for {duration_hours} hours")

    def is_overridden(self) -> bool:
        """Check if breaker is currently overridden.

        Returns:
            True if override is active.
        """
        if self._override_until is None:
            return False
        return datetime.now() < self._override_until

    def _make_status(
        self,
        status: Literal["OK", "WARNING", "TRIGGERED"],
        action: str,
        can_trade: bool,
        multiplier: float,
        message: str,
    ) -> BreakerStatus:
        """Create a BreakerStatus with override handling.

        Args:
            status: Raw status.
            action: Required action.
            can_trade: Trading allowed.
            multiplier: Position size multiplier.
            message: Status message.

        Returns:
            BreakerStatus, potentially modified by override.
        """
        if self.is_overridden() and status == "TRIGGERED":
            return BreakerStatus(
                breaker_name=self.name,
                status="WARNING",
                action_required="Override active - monitoring",
                can_trade=True,
                position_size_multiplier=0.5,
                message=f"{message} (OVERRIDDEN until {self._override_until})",
            )
        return BreakerStatus(
            breaker_name=self.name,
            status=status,
            action_required=action,
            can_trade=can_trade,
            position_size_multiplier=multiplier,
            message=message,
        )


class DrawdownBreaker(CircuitBreaker):
    """Circuit breaker based on portfolio drawdown."""

    def __init__(
        self,
        portfolio_manager: PortfolioRiskManager,
        warning_threshold: float = 0.07,
        halt_threshold: float = 0.10,
    ) -> None:
        """Initialize the drawdown breaker.

        Args:
            portfolio_manager: Portfolio risk manager instance.
            warning_threshold: Drawdown level for warning.
            halt_threshold: Drawdown level for trading halt.
        """
        super().__init__("DrawdownBreaker")
        self.portfolio_manager = portfolio_manager
        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold

    def check(self, **_kwargs: object) -> BreakerStatus:  # type: ignore[override]
        """Check drawdown level.

        Returns:
            Current status based on drawdown.
        """
        drawdown = self.portfolio_manager.get_current_drawdown()

        if drawdown >= self.halt_threshold:
            if not self._manually_reset:
                return self._make_status(
                    status="TRIGGERED",
                    action="Manual reset required to resume trading",
                    can_trade=False,
                    multiplier=0.0,
                    message=f"Drawdown at {drawdown:.1%} - trading halted",
                )
            # Reset was done, allow trading with reduced size
            return self._make_status(
                status="WARNING",
                action="Monitor closely, reduce size",
                can_trade=True,
                multiplier=0.5,
                message=f"Drawdown at {drawdown:.1%} - manually reset, reduced size",
            )

        if drawdown >= self.warning_threshold:
            return self._make_status(
                status="WARNING",
                action="Reduce position sizes by 50%",
                can_trade=True,
                multiplier=0.5,
                message=f"Drawdown at {drawdown:.1%} - reduce position sizes",
            )

        self._manually_reset = False  # Clear reset flag when back to normal
        return self._make_status(
            status="OK",
            action="None",
            can_trade=True,
            multiplier=1.0,
            message=f"Drawdown at {drawdown:.1%} - within limits",
        )


class LosingStreakBreaker(CircuitBreaker):
    """Circuit breaker based on consecutive losing trades."""

    def __init__(
        self,
        streak_limit: int = 5,
        pause_hours: int = 24,
    ) -> None:
        """Initialize the losing streak breaker.

        Args:
            streak_limit: Consecutive losses before pause.
            pause_hours: Hours to pause after streak.
        """
        super().__init__("LosingStreakBreaker")
        self.streak_limit = streak_limit
        self.pause_hours = pause_hours
        self._current_streak = 0
        self._pause_until: datetime | None = None

    def record_trade(self, is_winner: bool) -> None:
        """Record a trade result.

        Args:
            is_winner: True if trade was profitable.
        """
        if is_winner:
            self._current_streak = 0
            self._pause_until = None
        else:
            self._current_streak += 1
            if self._current_streak >= self.streak_limit:
                self._pause_until = datetime.now() + timedelta(hours=self.pause_hours)
                logger.warning(f"Losing streak limit reached: {self._current_streak} losses")

    def check(self, **_kwargs: object) -> BreakerStatus:  # type: ignore[override]
        """Check losing streak status.

        Returns:
            Current status based on losing streak.
        """
        if self._pause_until is not None:
            if datetime.now() < self._pause_until and not self._manually_reset:
                remaining = self._pause_until - datetime.now()
                return self._make_status(
                    status="TRIGGERED",
                    action="Manual confirmation required to resume",
                    can_trade=False,
                    multiplier=0.0,
                    message=f"Losing streak of {self._current_streak} - paused for {remaining.seconds // 3600}h",
                )
            # Pause expired or manually reset
            self._pause_until = None
            self._current_streak = 0
            self._manually_reset = False

        if self._current_streak >= self.streak_limit - 1:
            return self._make_status(
                status="WARNING",
                action="Consider reducing size or pausing",
                can_trade=True,
                multiplier=0.75,
                message=f"Approaching losing streak limit: {self._current_streak}/{self.streak_limit}",
            )

        return self._make_status(
            status="OK",
            action="None",
            can_trade=True,
            multiplier=1.0,
            message=f"Losing streak: {self._current_streak}/{self.streak_limit}",
        )

    @property
    def current_streak(self) -> int:
        """Get current losing streak count."""
        return self._current_streak


class VolatilityBreaker(CircuitBreaker):
    """Circuit breaker based on market volatility (VIX)."""

    def __init__(
        self,
        warning_vix: float = 25.0,
        halt_vix: float = 35.0,
    ) -> None:
        """Initialize the volatility breaker.

        Args:
            warning_vix: VIX level for warning.
            halt_vix: VIX level for halt.
        """
        super().__init__("VolatilityBreaker")
        self.warning_vix = warning_vix
        self.halt_vix = halt_vix

    def check(self, vix: float = 15.0, **_kwargs: object) -> BreakerStatus:  # type: ignore[override]
        """Check volatility level.

        Args:
            vix: Current VIX level.

        Returns:
            Current status based on VIX.
        """
        if vix >= self.halt_vix:
            return self._make_status(
                status="TRIGGERED",
                action="Only highest conviction trades at 50% size",
                can_trade=True,  # Can trade but restricted
                multiplier=0.5,
                message=f"VIX at {vix:.1f} - extreme volatility",
            )

        if vix >= self.warning_vix:
            return self._make_status(
                status="WARNING",
                action="Tighten stops, reduce size",
                can_trade=True,
                multiplier=0.75,
                message=f"VIX at {vix:.1f} - elevated volatility",
            )

        return self._make_status(
            status="OK",
            action="None",
            can_trade=True,
            multiplier=1.0,
            message=f"VIX at {vix:.1f} - normal volatility",
        )


class MarketRegimeBreaker(CircuitBreaker):
    """Circuit breaker based on market regime (SPY vs 200 SMA)."""

    def __init__(
        self,
        sma_period: int = 200,
    ) -> None:
        """Initialize the market regime breaker.

        Args:
            sma_period: SMA period for regime detection.
        """
        super().__init__("MarketRegimeBreaker")
        self.sma_period = sma_period

    def check(self, spy_data: pd.DataFrame | None = None, **_kwargs: object) -> BreakerStatus:  # type: ignore[override]
        """Check market regime.

        Args:
            spy_data: DataFrame with SPY OHLCV data.

        Returns:
            Current status based on market regime.
        """
        if spy_data is None or spy_data.empty:
            return self._make_status(
                status="OK",
                action="None",
                can_trade=True,
                multiplier=1.0,
                message="No SPY data available - assuming normal",
            )

        if len(spy_data) < self.sma_period:
            return self._make_status(
                status="OK",
                action="None",
                can_trade=True,
                multiplier=1.0,
                message=f"Insufficient data for {self.sma_period} SMA",
            )

        # Calculate SMA
        close_col = "close" if "close" in spy_data.columns else "Close"
        sma = spy_data[close_col].rolling(window=self.sma_period).mean()
        current_price = spy_data[close_col].iloc[-1]
        current_sma = sma.iloc[-1]

        # Calculate SMA slope (using last 5 days)
        sma_slope = (sma.iloc[-1] - sma.iloc[-5]) / sma.iloc[-5] if len(sma.dropna()) >= 5 else 0

        # Below SMA with negative slope = bearish
        below_sma = current_price < current_sma
        negative_slope = sma_slope < 0

        if below_sma and negative_slope:
            return self._make_status(
                status="WARNING",
                action="Defensive mode: reduce longs, allow shorts",
                can_trade=True,
                multiplier=0.75,
                message=f"Bearish regime: SPY {current_price:.2f} < SMA {current_sma:.2f}, slope {sma_slope:.2%}",
            )

        if below_sma:
            return self._make_status(
                status="WARNING",
                action="Caution: SPY below 200 SMA",
                can_trade=True,
                multiplier=0.9,
                message=f"Cautious: SPY {current_price:.2f} < SMA {current_sma:.2f}",
            )

        return self._make_status(
            status="OK",
            action="None",
            can_trade=True,
            multiplier=1.0,
            message=f"Bullish regime: SPY {current_price:.2f} > SMA {current_sma:.2f}",
        )


class CircuitBreakerSystem:
    """System that manages all circuit breakers.

    Example:
        >>> config = CircuitBreakerConfig()
        >>> system = CircuitBreakerSystem(portfolio_manager, config)
        >>> status = system.check_all(vix=28)
        >>> if not status.can_trade:
        ...     print("Trading halted:", status.actions_required)
    """

    def __init__(
        self,
        portfolio_manager: PortfolioRiskManager,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        """Initialize the circuit breaker system.

        Args:
            portfolio_manager: Portfolio risk manager instance.
            config: Circuit breaker configuration.
        """
        self.config = config or CircuitBreakerConfig()
        self.portfolio_manager = portfolio_manager

        # Initialize breakers
        self.drawdown_breaker = DrawdownBreaker(
            portfolio_manager=portfolio_manager,
            warning_threshold=self.config.drawdown_warning_threshold,
            halt_threshold=self.config.drawdown_halt_threshold,
        )
        self.losing_streak_breaker = LosingStreakBreaker(
            streak_limit=self.config.losing_streak_limit,
            pause_hours=self.config.losing_streak_pause_hours,
        )
        self.volatility_breaker = VolatilityBreaker(
            warning_vix=self.config.volatility_warning_vix,
            halt_vix=self.config.volatility_halt_vix,
        )
        self.market_regime_breaker = MarketRegimeBreaker(
            sma_period=self.config.market_regime_sma,
        )

        self._breakers: dict[str, CircuitBreaker] = {
            "drawdown": self.drawdown_breaker,
            "losing_streak": self.losing_streak_breaker,
            "volatility": self.volatility_breaker,
            "market_regime": self.market_regime_breaker,
        }

        # Event logging
        self._events: list[BreakerEvent] = []
        self._admin_key = secrets.token_hex(16)
        self._last_statuses: dict[str, str] = {}

    def check_all(
        self,
        vix: float = 15.0,
        spy_data: pd.DataFrame | None = None,
    ) -> SystemStatus:
        """Check all circuit breakers.

        Args:
            vix: Current VIX level.
            spy_data: SPY OHLCV data for market regime.

        Returns:
            Combined system status.
        """
        statuses: list[BreakerStatus] = []

        # Check each breaker
        statuses.append(self.drawdown_breaker.check())
        statuses.append(self.losing_streak_breaker.check())
        statuses.append(self.volatility_breaker.check(vix=vix))
        statuses.append(self.market_regime_breaker.check(spy_data=spy_data))

        # Log state changes
        for status in statuses:
            old_status = self._last_statuses.get(status.breaker_name, "OK")
            if status.status != old_status:
                self._log_event(
                    breaker_name=status.breaker_name,
                    event_type="status_change",
                    old_status=old_status,
                    new_status=status.status,
                    details=status.message,
                )
            self._last_statuses[status.breaker_name] = status.status

        # Determine overall status (most restrictive wins)
        can_trade = all(s.can_trade for s in statuses)
        position_multiplier = min(s.position_size_multiplier for s in statuses)

        overall: Literal["OK", "WARNING", "TRIGGERED"]
        if any(s.status == "TRIGGERED" for s in statuses):
            overall = "TRIGGERED"
        elif any(s.status == "WARNING" for s in statuses):
            overall = "WARNING"
        else:
            overall = "OK"

        warnings = [s.message for s in statuses if s.status == "WARNING"]
        actions = [s.action_required for s in statuses if s.status != "OK"]

        return SystemStatus(
            overall_status=overall,
            can_trade=can_trade,
            position_size_multiplier=position_multiplier,
            breaker_statuses=statuses,
            warnings=warnings,
            actions_required=actions,
        )

    def acknowledge_warning(self, breaker: str) -> bool:
        """Acknowledge a warning from a breaker.

        Args:
            breaker: Name of the breaker.

        Returns:
            True if acknowledgement successful.
        """
        if breaker not in self._breakers:
            return False

        self._breakers[breaker].acknowledge()
        self._log_event(
            breaker_name=breaker,
            event_type="acknowledge",
            old_status=self._last_statuses.get(breaker, "UNKNOWN"),
            new_status=self._last_statuses.get(breaker, "UNKNOWN"),
            details="Warning acknowledged by user",
        )
        return True

    def reset_breaker(self, breaker: str, confirmation_code: str) -> bool:
        """Reset a breaker after a halt.

        Args:
            breaker: Name of the breaker.
            confirmation_code: Confirmation code (must be "CONFIRM_RESET").

        Returns:
            True if reset successful.
        """
        if breaker not in self._breakers:
            return False

        if confirmation_code != "CONFIRM_RESET":
            logger.warning(f"Invalid confirmation code for resetting {breaker}")
            return False

        self._breakers[breaker].reset()
        self._log_event(
            breaker_name=breaker,
            event_type="reset",
            old_status="TRIGGERED",
            new_status="OK",
            details="Breaker manually reset",
        )
        return True

    def override_breaker(
        self,
        breaker: str,
        admin_key: str,
        duration_hours: int,
    ) -> bool:
        """Override a breaker temporarily.

        Args:
            breaker: Name of the breaker.
            admin_key: Admin authentication key.
            duration_hours: Hours to override.

        Returns:
            True if override successful.
        """
        if breaker not in self._breakers:
            return False

        if admin_key != self._admin_key:
            logger.warning(f"Invalid admin key for overriding {breaker}")
            return False

        if duration_hours > 24:
            logger.warning("Override duration limited to 24 hours")
            duration_hours = 24

        self._breakers[breaker].override(duration_hours)
        self._log_event(
            breaker_name=breaker,
            event_type="override",
            old_status="TRIGGERED",
            new_status="OVERRIDE",
            details=f"Overridden for {duration_hours} hours",
            user="admin",
        )
        return True

    def record_trade_result(self, is_winner: bool) -> None:
        """Record a trade result for the losing streak breaker.

        Args:
            is_winner: True if trade was profitable.
        """
        self.losing_streak_breaker.record_trade(is_winner)

    def get_events(self, limit: int = 100) -> list[BreakerEvent]:
        """Get recent breaker events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of recent events.
        """
        return self._events[-limit:]

    def get_admin_key(self) -> str:
        """Get the admin key (for testing only).

        Returns:
            Admin key for overrides.
        """
        return self._admin_key

    def _log_event(
        self,
        breaker_name: str,
        event_type: str,
        old_status: str,
        new_status: str,
        details: str,
        user: str | None = None,
    ) -> None:
        """Log a breaker event.

        Args:
            breaker_name: Name of the breaker.
            event_type: Type of event.
            old_status: Previous status.
            new_status: New status.
            details: Event details.
            user: User who triggered (if applicable).
        """
        event = BreakerEvent(
            breaker_name=breaker_name,
            event_type=event_type,
            old_status=old_status,
            new_status=new_status,
            timestamp=datetime.now(),
            details=details,
            user=user,
        )
        self._events.append(event)
        logger.info(f"Circuit breaker event: {breaker_name} {event_type} - {details}")

    def get_status_summary(self) -> dict:
        """Get a summary of all breaker statuses.

        Returns:
            Dictionary with status summary.
        """
        system_status = self.check_all()
        return {
            "overall_status": system_status.overall_status,
            "can_trade": system_status.can_trade,
            "position_size_multiplier": system_status.position_size_multiplier,
            "breakers": {
                s.breaker_name: {
                    "status": s.status,
                    "can_trade": s.can_trade,
                    "multiplier": s.position_size_multiplier,
                    "message": s.message,
                }
                for s in system_status.breaker_statuses
            },
            "warnings": system_status.warnings,
            "actions_required": system_status.actions_required,
        }
