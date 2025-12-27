"""Unit tests for Portfolio Risk Manager."""

from __future__ import annotations

from datetime import date

import pytest

from src.risk.portfolio import (
    PortfolioRiskManager,
    Position,
    RiskConfig,
    ValidationResult,
)
from src.strategies.base import Signal


# Test fixtures
def create_signal(
    symbol: str = "AAPL",
    sector: str = "Technology",
    risk_percent: float = 1.0,
) -> Signal:
    """Create a test signal."""
    return Signal(
        symbol=symbol,
        strategy="TestStrategy",
        direction="LONG",
        entry_price=100.0,
        stop_loss=99.0,
        take_profit=105.0,
        score=75,
        reasoning="Test signal",
        metadata={"sector": sector, "risk_percent": risk_percent},
    )


def create_position(
    symbol: str = "AAPL",
    sector: str = "Technology",
    risk_percent: float = 1.0,
) -> Position:
    """Create a test position."""
    return Position(
        symbol=symbol,
        sector=sector,
        entry_price=100.0,
        current_price=101.0,
        shares=100,
        stop_loss=95.0,
        direction="LONG",
        entry_date=date.today(),
        risk_amount=500.0,
        risk_percent=risk_percent,
    )


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RiskConfig()
        assert config.max_positions == 5
        assert config.max_sector_positions == 2
        assert config.max_correlated_positions == 3
        assert config.correlation_threshold == 0.7
        assert config.max_portfolio_heat == 0.05
        assert config.daily_loss_limit == 0.03
        assert config.weekly_loss_limit == 0.05
        assert config.max_drawdown == 0.10

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = RiskConfig(
            max_positions=10,
            daily_loss_limit=0.05,
        )
        assert config.max_positions == 10
        assert config.daily_loss_limit == 0.05


class TestPosition:
    """Tests for Position dataclass."""

    def test_unrealized_pnl_long_profit(self) -> None:
        """Test unrealized P&L for profitable long position."""
        pos = Position(
            symbol="AAPL",
            sector="Technology",
            entry_price=100.0,
            current_price=110.0,
            shares=100,
            stop_loss=95.0,
            direction="LONG",
            entry_date=date.today(),
            risk_amount=500.0,
            risk_percent=1.0,
        )
        assert pos.unrealized_pnl == 1000.0  # (110 - 100) × 100

    def test_unrealized_pnl_long_loss(self) -> None:
        """Test unrealized P&L for losing long position."""
        pos = Position(
            symbol="AAPL",
            sector="Technology",
            entry_price=100.0,
            current_price=95.0,
            shares=100,
            stop_loss=90.0,
            direction="LONG",
            entry_date=date.today(),
            risk_amount=500.0,
            risk_percent=1.0,
        )
        assert pos.unrealized_pnl == -500.0

    def test_unrealized_pnl_short_profit(self) -> None:
        """Test unrealized P&L for profitable short position."""
        pos = Position(
            symbol="AAPL",
            sector="Technology",
            entry_price=100.0,
            current_price=90.0,
            shares=100,
            stop_loss=105.0,
            direction="SHORT",
            entry_date=date.today(),
            risk_amount=500.0,
            risk_percent=1.0,
        )
        assert pos.unrealized_pnl == 1000.0  # (100 - 90) × 100

    def test_unrealized_pnl_percent(self) -> None:
        """Test unrealized P&L percentage calculation."""
        pos = Position(
            symbol="AAPL",
            sector="Technology",
            entry_price=100.0,
            current_price=110.0,
            shares=100,
            stop_loss=95.0,
            direction="LONG",
            entry_date=date.today(),
            risk_amount=500.0,
            risk_percent=1.0,
        )
        assert pos.unrealized_pnl_percent == pytest.approx(10.0, rel=0.01)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_allowed_result(self) -> None:
        """Test allowed validation result."""
        result = ValidationResult(
            allowed=True,
            checks_passed=["Position count OK", "Sector limit OK"],
            checks_failed=[],
            warnings=["Approaching position limit"],
        )
        assert result.allowed is True
        assert len(result.checks_passed) == 2
        assert len(result.checks_failed) == 0
        assert len(result.warnings) == 1

    def test_rejected_result(self) -> None:
        """Test rejected validation result."""
        result = ValidationResult(
            allowed=False,
            checks_passed=["Sector limit OK"],
            checks_failed=["Max positions reached"],
            warnings=[],
        )
        assert result.allowed is False
        assert len(result.checks_failed) == 1


class TestPortfolioRiskManagerInit:
    """Tests for PortfolioRiskManager initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        assert manager.config == config
        assert manager.account_equity == 100000.0
        assert manager.get_position_count() == 0

    def test_init_custom_equity(self) -> None:
        """Test initialization with custom equity."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=50000.0)

        assert manager.account_equity == 50000.0
        assert manager.get_peak_equity() == 50000.0


class TestMaxPositionsLimit:
    """Tests for maximum positions limit."""

    def test_rejects_when_max_positions_reached(self) -> None:
        """5 positions = reject 6th."""
        config = RiskConfig(max_positions=5)
        manager = PortfolioRiskManager(config)

        # Add 5 positions
        positions = [
            create_position(symbol=f"STOCK{i}", sector="Different", risk_percent=0.5)
            for i in range(5)
        ]
        manager.set_positions(positions)

        # Try to add 6th
        signal = create_signal(symbol="STOCK5", sector="NewSector")
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "Max positions reached" in reason

    def test_allows_within_position_limit(self) -> None:
        """4 positions = allow 5th."""
        config = RiskConfig(max_positions=5)
        manager = PortfolioRiskManager(config)

        # Add 4 positions
        positions = [
            create_position(symbol=f"STOCK{i}", sector="Different", risk_percent=0.5)
            for i in range(4)
        ]
        manager.set_positions(positions)

        # Try to add 5th
        signal = create_signal(symbol="STOCK4", sector="NewSector", risk_percent=0.5)
        can_open, reason = manager.can_open_position(signal)

        assert can_open is True


class TestSectorLimit:
    """Tests for sector position limit."""

    def test_rejects_when_sector_limit_reached(self) -> None:
        """2 tech positions = reject 3rd tech."""
        config = RiskConfig(max_sector_positions=2)
        manager = PortfolioRiskManager(config)

        # Add 2 tech positions
        positions = [
            create_position(symbol="AAPL", sector="Technology", risk_percent=0.5),
            create_position(symbol="MSFT", sector="Technology", risk_percent=0.5),
        ]
        manager.set_positions(positions)

        # Try to add 3rd tech
        signal = create_signal(symbol="GOOGL", sector="Technology")
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "Technology" in reason
        assert "positions reached" in reason

    def test_allows_different_sector(self) -> None:
        """2 tech positions + 1 healthcare = allowed."""
        config = RiskConfig(max_sector_positions=2)
        manager = PortfolioRiskManager(config)

        # Add 2 tech positions
        positions = [
            create_position(symbol="AAPL", sector="Technology", risk_percent=0.5),
            create_position(symbol="MSFT", sector="Technology", risk_percent=0.5),
        ]
        manager.set_positions(positions)

        # Try to add healthcare
        signal = create_signal(symbol="JNJ", sector="Healthcare", risk_percent=0.5)
        can_open, reason = manager.can_open_position(signal)

        assert can_open is True


class TestCorrelatedPositions:
    """Tests for correlated positions limit."""

    def test_rejects_correlated_positions(self) -> None:
        """Too many correlated positions = reject."""
        config = RiskConfig(max_correlated_positions=2)
        manager = PortfolioRiskManager(config)

        # Add 2 positions in same sector (sector correlation)
        positions = [
            create_position(symbol="AAPL", sector="Technology", risk_percent=0.5),
            create_position(symbol="MSFT", sector="Technology", risk_percent=0.5),
        ]
        manager.set_positions(positions)

        # Try to add 3rd in same sector
        signal = create_signal(symbol="GOOGL", sector="Technology")
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "correlated" in reason.lower()

    def test_get_correlated_positions_same_sector(self) -> None:
        """Get correlated positions based on sector."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        positions = [
            create_position(symbol="AAPL", sector="Technology"),
            create_position(symbol="MSFT", sector="Technology"),
            create_position(symbol="JNJ", sector="Healthcare"),
        ]
        manager.set_positions(positions)

        correlated = manager.get_correlated_positions("GOOGL", "Technology")
        assert "AAPL" in correlated
        assert "MSFT" in correlated
        assert "JNJ" not in correlated


class TestPortfolioHeat:
    """Tests for portfolio heat calculation."""

    def test_portfolio_heat_calculation(self) -> None:
        """Sum of individual risks."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        positions = [
            create_position(symbol="AAPL", risk_percent=1.0),
            create_position(symbol="MSFT", risk_percent=1.5),
            create_position(symbol="GOOGL", risk_percent=0.5),
        ]
        manager.set_positions(positions)

        heat = manager.get_portfolio_heat()
        # Total = (1.0 + 1.5 + 0.5) / 100 = 0.03
        assert heat == pytest.approx(0.03, rel=0.01)

    def test_empty_portfolio_heat(self) -> None:
        """Empty portfolio has zero heat."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        assert manager.get_portfolio_heat() == 0.0

    def test_rejects_when_heat_exceeded(self) -> None:
        """Reject when portfolio heat would exceed limit."""
        config = RiskConfig(max_portfolio_heat=0.03)
        manager = PortfolioRiskManager(config)

        # Add positions with 2.5% heat
        positions = [
            create_position(symbol="AAPL", risk_percent=1.0, sector="Tech1"),
            create_position(symbol="MSFT", risk_percent=1.5, sector="Tech2"),
        ]
        manager.set_positions(positions)

        # Try to add position with 1% risk (would exceed 3% limit)
        signal = create_signal(symbol="GOOGL", sector="Tech3", risk_percent=1.0)
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "heat" in reason.lower()


class TestDailyLossLimit:
    """Tests for daily loss limit."""

    def test_daily_loss_limit_blocks_trades(self) -> None:
        """Daily limit hit = no new trades."""
        config = RiskConfig(daily_loss_limit=0.03)  # 3%
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        # Set daily loss at limit
        manager.set_daily_pnl(-3000.0)  # -3% of 100k

        signal = create_signal()
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "daily" in reason.lower()
        assert manager.is_daily_limit_hit() is True

    def test_daily_loss_below_limit_allows_trades(self) -> None:
        """Daily loss below limit = trades allowed."""
        config = RiskConfig(daily_loss_limit=0.03)
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        # Set daily loss below limit
        manager.set_daily_pnl(-2000.0)  # -2% < 3% limit

        assert manager.is_daily_limit_hit() is False


class TestWeeklyLossLimit:
    """Tests for weekly loss limit."""

    def test_weekly_loss_limit_blocks_trades(self) -> None:
        """Weekly limit hit = no new trades."""
        config = RiskConfig(weekly_loss_limit=0.05)  # 5%
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        # Set weekly loss at limit
        manager.set_weekly_pnl(-5000.0)  # -5% of 100k

        signal = create_signal()
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "weekly" in reason.lower()
        assert manager.is_weekly_limit_hit() is True

    def test_weekly_loss_below_limit_allows_trades(self) -> None:
        """Weekly loss below limit = trades allowed."""
        config = RiskConfig(weekly_loss_limit=0.05)
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        # Set weekly loss below limit
        manager.set_weekly_pnl(-3000.0)  # -3% < 5% limit

        assert manager.is_weekly_limit_hit() is False


class TestDrawdown:
    """Tests for drawdown tracking."""

    def test_drawdown_calculation_correct(self) -> None:
        """Peak to trough calculation."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        # Set peak and current equity
        manager.set_peak_equity(100000.0)
        manager.account_equity = 92000.0

        drawdown = manager.get_current_drawdown()
        assert drawdown == pytest.approx(0.08, rel=0.01)  # 8% drawdown

    def test_max_drawdown_triggers_halt(self) -> None:
        """10% drawdown = halt trading."""
        config = RiskConfig(max_drawdown=0.10)
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        # Set 10% drawdown
        manager.set_peak_equity(100000.0)
        manager.account_equity = 90000.0

        assert manager.is_max_drawdown_hit() is True

        signal = create_signal()
        can_open, reason = manager.can_open_position(signal)

        assert can_open is False
        assert "drawdown" in reason.lower()

    def test_drawdown_below_limit(self) -> None:
        """Drawdown below limit = trading allowed."""
        config = RiskConfig(max_drawdown=0.10)
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        manager.set_peak_equity(100000.0)
        manager.account_equity = 95000.0  # 5% drawdown

        assert manager.is_max_drawdown_hit() is False

    def test_update_equity_tracks_peak(self) -> None:
        """Update equity should track new highs."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        manager.update_equity(110000.0)  # New high
        assert manager.get_peak_equity() == 110000.0

        manager.update_equity(105000.0)  # Pullback
        assert manager.get_peak_equity() == 110000.0  # Peak unchanged


class TestValidationPipeline:
    """Tests for validation pipeline."""

    def test_validation_pipeline_returns_details(self) -> None:
        """All checks should be reported."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        signal = create_signal()
        result = manager.validate_trade(signal)

        # Should have multiple checks passed
        assert len(result.checks_passed) > 0
        assert result.allowed is True
        assert isinstance(result.checks_failed, list)
        assert isinstance(result.warnings, list)

    def test_all_checks_in_result(self) -> None:
        """Validation should include all check types."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        signal = create_signal()
        result = manager.validate_trade(signal)

        # Check that key validations are performed
        all_checks = " ".join(result.checks_passed + result.checks_failed)
        assert "Position" in all_checks or "position" in all_checks
        assert "Sector" in all_checks or "sector" in all_checks
        assert "heat" in all_checks.lower()


class TestWarnings:
    """Tests for approaching limit warnings."""

    def test_warnings_for_approaching_limits(self) -> None:
        """80% of limit = warning."""
        config = RiskConfig(max_positions=5)
        manager = PortfolioRiskManager(config)

        # Add 4 positions (80% of 5)
        positions = [
            create_position(symbol=f"STOCK{i}", sector=f"Sector{i}", risk_percent=0.5)
            for i in range(4)
        ]
        manager.set_positions(positions)

        signal = create_signal(symbol="NEWSTOCK", sector="NewSector", risk_percent=0.5)
        result = manager.validate_trade(signal)

        assert len(result.warnings) > 0
        assert any("position" in w.lower() for w in result.warnings)

    def test_no_warnings_when_well_under_limits(self) -> None:
        """Well under limits = no warnings."""
        config = RiskConfig(max_positions=10)
        manager = PortfolioRiskManager(config)

        # Add 2 positions (20% of 10)
        positions = [
            create_position(symbol=f"STOCK{i}", sector=f"Sector{i}", risk_percent=0.2)
            for i in range(2)
        ]
        manager.set_positions(positions)

        signal = create_signal(symbol="NEWSTOCK", sector="NewSector", risk_percent=0.2)
        result = manager.validate_trade(signal)

        # Should have no position limit warnings
        position_warnings = [w for w in result.warnings if "position" in w.lower()]
        assert len(position_warnings) == 0


class TestSectorExposure:
    """Tests for sector exposure tracking."""

    def test_get_sector_exposure(self) -> None:
        """Count positions per sector."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        positions = [
            create_position(symbol="AAPL", sector="Technology"),
            create_position(symbol="MSFT", sector="Technology"),
            create_position(symbol="JNJ", sector="Healthcare"),
            create_position(symbol="PFE", sector="Healthcare"),
            create_position(symbol="UNH", sector="Healthcare"),
        ]
        manager.set_positions(positions)

        exposure = manager.get_sector_exposure()
        assert exposure["Technology"] == 2
        assert exposure["Healthcare"] == 3


class TestPositionManagement:
    """Tests for position management methods."""

    def test_add_position(self) -> None:
        """Add a position."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        pos = create_position(symbol="AAPL")
        manager.add_position(pos)

        assert manager.get_position_count() == 1
        assert manager.positions[0].symbol == "AAPL"

    def test_remove_position(self) -> None:
        """Remove a position."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        positions = [
            create_position(symbol="AAPL"),
            create_position(symbol="MSFT"),
        ]
        manager.set_positions(positions)

        removed = manager.remove_position("AAPL")
        assert removed is True
        assert manager.get_position_count() == 1
        assert manager.positions[0].symbol == "MSFT"

    def test_remove_nonexistent_position(self) -> None:
        """Remove nonexistent position returns False."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        removed = manager.remove_position("NOTFOUND")
        assert removed is False


class TestPnLTracking:
    """Tests for P&L tracking."""

    def test_record_trade_pnl(self) -> None:
        """Record P&L updates daily and weekly."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        manager.record_trade_pnl(500.0)
        assert manager.get_daily_pnl() == 500.0
        assert manager.get_weekly_pnl() == 500.0
        assert manager.account_equity == 100500.0

    def test_record_trade_pnl_updates_peak(self) -> None:
        """Record profit updates peak equity."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        manager.record_trade_pnl(5000.0)
        assert manager.get_peak_equity() == 105000.0

    def test_record_trade_loss_no_peak_update(self) -> None:
        """Record loss doesn't update peak."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        manager.record_trade_pnl(-2000.0)
        assert manager.get_peak_equity() == 100000.0
        assert manager.account_equity == 98000.0


class TestPortfolioSummary:
    """Tests for portfolio summary."""

    def test_get_portfolio_summary(self) -> None:
        """Get complete portfolio summary."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)

        positions = [
            create_position(symbol="AAPL", sector="Technology", risk_percent=1.0),
            create_position(symbol="MSFT", sector="Technology", risk_percent=1.0),
        ]
        manager.set_positions(positions)
        manager.set_daily_pnl(-500.0)
        manager.set_weekly_pnl(-1000.0)

        summary = manager.get_portfolio_summary()

        assert summary["positions_count"] == 2
        assert summary["max_positions"] == 5
        assert summary["portfolio_heat"] == pytest.approx(0.02, rel=0.01)
        assert summary["daily_pnl"] == -500.0
        assert summary["weekly_pnl"] == -1000.0
        assert summary["current_equity"] == 100000.0
        assert "Technology" in summary["sector_exposure"]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_peak_equity(self) -> None:
        """Handle zero peak equity gracefully."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config, account_equity=100000.0)
        manager._peak_equity = 0  # Force zero peak

        drawdown = manager.get_current_drawdown()
        assert drawdown == 0.0

    def test_empty_correlation_matrix(self) -> None:
        """Handle empty symbol list for correlation."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        matrix = manager.calculate_correlation_matrix([])
        assert matrix.empty

        matrix = manager.calculate_correlation_matrix(["AAPL"])
        assert matrix.empty

    def test_unknown_sector(self) -> None:
        """Handle unknown sector gracefully."""
        config = RiskConfig()
        manager = PortfolioRiskManager(config)

        signal = Signal(
            symbol="XYZ",
            strategy="Test",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            score=75,
            reasoning="Test",
            metadata={},  # No sector
        )

        result = manager.validate_trade(signal)
        assert "Unknown" in str(result.checks_passed) or len(result.checks_passed) > 0
