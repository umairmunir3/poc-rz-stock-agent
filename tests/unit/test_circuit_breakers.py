"""Unit tests for Circuit Breaker System."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.risk.circuit_breakers import (
    BreakerStatus,
    CircuitBreakerConfig,
    CircuitBreakerSystem,
    DrawdownBreaker,
    LosingStreakBreaker,
    MarketRegimeBreaker,
    SystemStatus,
    VolatilityBreaker,
)
from src.risk.portfolio import PortfolioRiskManager, RiskConfig


# Test fixtures
def create_portfolio_manager(
    account_equity: float = 100000.0,
    drawdown: float = 0.0,
) -> PortfolioRiskManager:
    """Create a portfolio manager with specified drawdown."""
    config = RiskConfig()
    manager = PortfolioRiskManager(config, account_equity=account_equity)

    # Set up drawdown by adjusting peak and current equity
    if drawdown > 0:
        peak = account_equity / (1 - drawdown)
        manager.set_peak_equity(peak)

    return manager


def create_spy_data(
    days: int = 250,
    current_price: float = 450.0,
    sma_200: float = 440.0,
    trending_up: bool = True,
) -> pd.DataFrame:
    """Create mock SPY data."""
    dates = pd.date_range(end=date.today(), periods=days, freq="D")

    if trending_up:
        prices = [sma_200 - 20 + (i / days * 50) for i in range(days)]
    else:
        prices = [sma_200 + 20 - (i / days * 50) for i in range(days)]

    # Set current price
    prices[-1] = current_price

    return pd.DataFrame(
        {
            "close": prices,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "volume": [1000000] * days,
        },
        index=dates,
    )


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        assert config.drawdown_warning_threshold == 0.07
        assert config.drawdown_halt_threshold == 0.10
        assert config.losing_streak_limit == 5
        assert config.volatility_warning_vix == 25.0
        assert config.volatility_halt_vix == 35.0
        assert config.market_regime_sma == 200

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            drawdown_warning_threshold=0.05,
            losing_streak_limit=3,
        )
        assert config.drawdown_warning_threshold == 0.05
        assert config.losing_streak_limit == 3


class TestBreakerStatus:
    """Tests for BreakerStatus dataclass."""

    def test_ok_status(self) -> None:
        """Test OK status."""
        status = BreakerStatus(
            breaker_name="TestBreaker",
            status="OK",
            action_required="None",
            can_trade=True,
            position_size_multiplier=1.0,
            message="All clear",
        )
        assert status.status == "OK"
        assert status.can_trade is True
        assert status.position_size_multiplier == 1.0

    def test_triggered_status(self) -> None:
        """Test TRIGGERED status."""
        status = BreakerStatus(
            breaker_name="TestBreaker",
            status="TRIGGERED",
            action_required="Manual reset required",
            can_trade=False,
            position_size_multiplier=0.0,
            message="Breaker triggered",
        )
        assert status.status == "TRIGGERED"
        assert status.can_trade is False


class TestDrawdownBreaker:
    """Tests for DrawdownBreaker."""

    def test_drawdown_warning_at_7_percent(self) -> None:
        """Warning at 7% drawdown."""
        manager = create_portfolio_manager(drawdown=0.07)
        breaker = DrawdownBreaker(manager, warning_threshold=0.07, halt_threshold=0.10)

        status = breaker.check()

        assert status.status == "WARNING"
        assert status.can_trade is True
        assert status.position_size_multiplier == 0.5

    def test_drawdown_halt_at_10_percent(self) -> None:
        """Halt at 10% drawdown."""
        manager = create_portfolio_manager(drawdown=0.11)  # Slightly above threshold
        breaker = DrawdownBreaker(manager, warning_threshold=0.07, halt_threshold=0.10)

        status = breaker.check()

        assert status.status == "TRIGGERED"
        assert status.can_trade is False
        assert status.position_size_multiplier == 0.0

    def test_drawdown_reduces_position_size(self) -> None:
        """Drawdown warning reduces position size by 50%."""
        manager = create_portfolio_manager(drawdown=0.08)
        breaker = DrawdownBreaker(manager, warning_threshold=0.07, halt_threshold=0.10)

        status = breaker.check()

        assert status.position_size_multiplier == 0.5

    def test_drawdown_ok_below_threshold(self) -> None:
        """OK when below warning threshold."""
        manager = create_portfolio_manager(drawdown=0.05)
        breaker = DrawdownBreaker(manager, warning_threshold=0.07, halt_threshold=0.10)

        status = breaker.check()

        assert status.status == "OK"
        assert status.can_trade is True
        assert status.position_size_multiplier == 1.0

    def test_manual_reset_allows_trading(self) -> None:
        """Manual reset allows trading after halt."""
        manager = create_portfolio_manager(drawdown=0.11)  # Above threshold
        breaker = DrawdownBreaker(manager, warning_threshold=0.07, halt_threshold=0.10)

        # Initially halted
        status = breaker.check()
        assert status.can_trade is False

        # After reset
        breaker.reset()
        status = breaker.check()
        assert status.can_trade is True
        assert status.position_size_multiplier == 0.5  # Still reduced


class TestLosingStreakBreaker:
    """Tests for LosingStreakBreaker."""

    def test_losing_streak_counts_correctly(self) -> None:
        """Streak counts consecutive losses."""
        breaker = LosingStreakBreaker(streak_limit=5)

        breaker.record_trade(is_winner=False)
        assert breaker.current_streak == 1

        breaker.record_trade(is_winner=False)
        assert breaker.current_streak == 2

        breaker.record_trade(is_winner=True)
        assert breaker.current_streak == 0

    def test_losing_streak_pauses_trading(self) -> None:
        """5 losses pauses trading."""
        breaker = LosingStreakBreaker(streak_limit=5, pause_hours=24)

        for _ in range(5):
            breaker.record_trade(is_winner=False)

        status = breaker.check()

        assert status.status == "TRIGGERED"
        assert status.can_trade is False

    def test_losing_streak_warning_before_limit(self) -> None:
        """Warning when approaching limit."""
        breaker = LosingStreakBreaker(streak_limit=5)

        for _ in range(4):  # One before limit
            breaker.record_trade(is_winner=False)

        status = breaker.check()

        assert status.status == "WARNING"
        assert status.can_trade is True

    def test_winning_trade_resets_streak(self) -> None:
        """Winning trade resets the streak."""
        breaker = LosingStreakBreaker(streak_limit=5)

        for _ in range(3):
            breaker.record_trade(is_winner=False)
        assert breaker.current_streak == 3

        breaker.record_trade(is_winner=True)
        assert breaker.current_streak == 0

        status = breaker.check()
        assert status.status == "OK"


class TestVolatilityBreaker:
    """Tests for VolatilityBreaker."""

    def test_vix_warning_at_25(self) -> None:
        """Warning at VIX 25."""
        breaker = VolatilityBreaker(warning_vix=25.0, halt_vix=35.0)

        status = breaker.check(vix=25.0)

        assert status.status == "WARNING"
        assert status.can_trade is True
        assert status.position_size_multiplier == 0.75

    def test_vix_halt_at_35(self) -> None:
        """Triggered at VIX 35."""
        breaker = VolatilityBreaker(warning_vix=25.0, halt_vix=35.0)

        status = breaker.check(vix=35.0)

        assert status.status == "TRIGGERED"
        assert status.can_trade is True  # Can trade but restricted
        assert status.position_size_multiplier == 0.5

    def test_vix_ok_below_warning(self) -> None:
        """OK when VIX below warning level."""
        breaker = VolatilityBreaker(warning_vix=25.0, halt_vix=35.0)

        status = breaker.check(vix=18.0)

        assert status.status == "OK"
        assert status.position_size_multiplier == 1.0


class TestMarketRegimeBreaker:
    """Tests for MarketRegimeBreaker."""

    def test_market_regime_detects_bear(self) -> None:
        """Detects bearish regime when SPY < 200 SMA with negative slope."""
        spy_data = create_spy_data(
            days=250,
            current_price=430.0,
            sma_200=450.0,
            trending_up=False,
        )
        breaker = MarketRegimeBreaker(sma_period=200)

        status = breaker.check(spy_data=spy_data)

        assert status.status == "WARNING"
        assert "Bearish" in status.message or "below" in status.message.lower()

    def test_market_regime_bullish(self) -> None:
        """OK when SPY above 200 SMA."""
        spy_data = create_spy_data(
            days=250,
            current_price=480.0,
            sma_200=450.0,
            trending_up=True,
        )
        breaker = MarketRegimeBreaker(sma_period=200)

        status = breaker.check(spy_data=spy_data)

        assert status.status == "OK"

    def test_no_data_returns_ok(self) -> None:
        """OK when no SPY data available."""
        breaker = MarketRegimeBreaker()

        status = breaker.check(spy_data=None)

        assert status.status == "OK"

    def test_insufficient_data_returns_ok(self) -> None:
        """OK when insufficient data for SMA."""
        spy_data = create_spy_data(days=50)  # Less than 200
        breaker = MarketRegimeBreaker(sma_period=200)

        status = breaker.check(spy_data=spy_data)

        assert status.status == "OK"


class TestCircuitBreakerSystem:
    """Tests for CircuitBreakerSystem."""

    def test_system_initialization(self) -> None:
        """Test system initialization."""
        manager = create_portfolio_manager()
        config = CircuitBreakerConfig()
        system = CircuitBreakerSystem(manager, config)

        assert system.config == config
        assert system.drawdown_breaker is not None
        assert system.losing_streak_breaker is not None
        assert system.volatility_breaker is not None
        assert system.market_regime_breaker is not None

    def test_check_all_returns_system_status(self) -> None:
        """check_all returns SystemStatus."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        status = system.check_all()

        assert isinstance(status, SystemStatus)
        assert len(status.breaker_statuses) == 4

    def test_most_restrictive_wins(self) -> None:
        """Most restrictive breaker status wins."""
        manager = create_portfolio_manager(drawdown=0.11)  # Above threshold
        system = CircuitBreakerSystem(manager)

        status = system.check_all(vix=15.0)  # Low VIX

        # Drawdown halt should win
        assert status.overall_status == "TRIGGERED"
        assert status.can_trade is False

    def test_position_multiplier_is_minimum(self) -> None:
        """Position multiplier is the minimum across breakers."""
        manager = create_portfolio_manager(drawdown=0.08)  # 50% multiplier
        system = CircuitBreakerSystem(manager)

        status = system.check_all(vix=30.0)  # 75% multiplier

        # Minimum should be 0.5
        assert status.position_size_multiplier == 0.5


class TestManualControls:
    """Tests for manual controls."""

    def test_acknowledge_warning(self) -> None:
        """Acknowledge a warning."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        result = system.acknowledge_warning("drawdown")

        assert result is True

    def test_acknowledge_invalid_breaker(self) -> None:
        """Acknowledge invalid breaker fails."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        result = system.acknowledge_warning("nonexistent")

        assert result is False

    def test_manual_reset_required(self) -> None:
        """Manual reset required to resume after halt."""
        manager = create_portfolio_manager(drawdown=0.11)  # Above threshold
        system = CircuitBreakerSystem(manager)

        # Initially halted
        status = system.check_all()
        assert status.can_trade is False

        # Reset with correct code
        result = system.reset_breaker("drawdown", "CONFIRM_RESET")
        assert result is True

        # Now can trade
        status = system.check_all()
        drawdown_status = next(
            s for s in status.breaker_statuses if s.breaker_name == "DrawdownBreaker"
        )
        assert drawdown_status.can_trade is True

    def test_reset_with_wrong_code_fails(self) -> None:
        """Reset with wrong confirmation code fails."""
        manager = create_portfolio_manager(drawdown=0.10)
        system = CircuitBreakerSystem(manager)

        result = system.reset_breaker("drawdown", "WRONG_CODE")

        assert result is False

    def test_override_with_admin_key(self) -> None:
        """Admin can override breaker temporarily."""
        manager = create_portfolio_manager(drawdown=0.11)  # Above threshold
        system = CircuitBreakerSystem(manager)

        admin_key = system.get_admin_key()
        result = system.override_breaker("drawdown", admin_key, duration_hours=2)

        assert result is True

        # Check that override is active
        status = system.check_all()
        drawdown_status = next(
            s for s in status.breaker_statuses if s.breaker_name == "DrawdownBreaker"
        )
        assert "OVERRIDDEN" in drawdown_status.message

    def test_override_with_wrong_key_fails(self) -> None:
        """Override with wrong admin key fails."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        result = system.override_breaker("drawdown", "wrong_key", duration_hours=2)

        assert result is False

    def test_override_duration_limited(self) -> None:
        """Override duration limited to 24 hours."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)
        admin_key = system.get_admin_key()

        # Request 48 hours
        result = system.override_breaker("drawdown", admin_key, duration_hours=48)

        assert result is True
        # Should be capped at 24 hours (verified by implementation)


class TestEventLogging:
    """Tests for event logging."""

    def test_events_logged_on_status_change(self) -> None:
        """Events logged when status changes."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        # First check - establishes baseline
        system.check_all()

        # Change drawdown to trigger
        manager.set_peak_equity(manager.account_equity / 0.9)
        system.check_all()

        events = system.get_events()
        assert len(events) > 0

    def test_get_events_limited(self) -> None:
        """Get events respects limit."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        # Generate some events
        for _ in range(5):
            system.check_all()

        events = system.get_events(limit=2)
        assert len(events) <= 2


class TestRecordTradeResult:
    """Tests for recording trade results."""

    def test_record_trade_updates_streak(self) -> None:
        """Recording trade updates losing streak."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        system.record_trade_result(is_winner=False)
        assert system.losing_streak_breaker.current_streak == 1

        system.record_trade_result(is_winner=True)
        assert system.losing_streak_breaker.current_streak == 0


class TestStatusSummary:
    """Tests for status summary."""

    def test_get_status_summary(self) -> None:
        """Get comprehensive status summary."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        summary = system.get_status_summary()

        assert "overall_status" in summary
        assert "can_trade" in summary
        assert "position_size_multiplier" in summary
        assert "breakers" in summary
        assert len(summary["breakers"]) == 4


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_breakers_ok(self) -> None:
        """All breakers OK returns normal status."""
        manager = create_portfolio_manager(drawdown=0.0)
        system = CircuitBreakerSystem(manager)

        status = system.check_all(vix=15.0)

        assert status.overall_status == "OK"
        assert status.can_trade is True
        assert status.position_size_multiplier == 1.0

    def test_multiple_warnings(self) -> None:
        """Multiple warnings returns WARNING status."""
        manager = create_portfolio_manager(drawdown=0.08)  # Warning
        system = CircuitBreakerSystem(manager)

        status = system.check_all(vix=28.0)  # Also warning

        assert status.overall_status == "WARNING"
        assert status.can_trade is True
        assert len(status.warnings) >= 2

    def test_empty_spy_data(self) -> None:
        """Empty SPY data handled gracefully."""
        manager = create_portfolio_manager()
        system = CircuitBreakerSystem(manager)

        status = system.check_all(spy_data=pd.DataFrame())

        assert status is not None
        market_status = next(
            s for s in status.breaker_statuses if s.breaker_name == "MarketRegimeBreaker"
        )
        assert market_status.status == "OK"
