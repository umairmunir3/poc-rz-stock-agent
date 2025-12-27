"""Unit tests for position sizing calculator."""

from __future__ import annotations

import pytest

from src.risk.position_sizing import (
    OptionsPosition,
    PositionSize,
    PositionSizer,
    PositionSizingError,
)


class TestPositionSizeDataclass:
    """Tests for PositionSize dataclass."""

    def test_attributes(self) -> None:
        """Test PositionSize has correct attributes."""
        pos = PositionSize(
            shares=100,
            dollar_amount=5000.0,
            risk_amount=250.0,
            risk_percent=2.5,
            position_percent=50.0,
        )
        assert pos.shares == 100
        assert pos.dollar_amount == 5000.0
        assert pos.risk_amount == 250.0
        assert pos.risk_percent == 2.5
        assert pos.position_percent == 50.0


class TestOptionsPositionDataclass:
    """Tests for OptionsPosition dataclass."""

    def test_attributes_viable(self) -> None:
        """Test OptionsPosition with viable position."""
        pos = OptionsPosition(
            contracts=2,
            premium_cost=500.0,
            risk_amount=500.0,
            position_percent=5.0,
            is_viable=True,
            message="Position viable: 2 contract(s) for $500.00",
        )
        assert pos.contracts == 2
        assert pos.premium_cost == 500.0
        assert pos.risk_amount == 500.0
        assert pos.position_percent == 5.0
        assert pos.is_viable is True
        assert "viable" in pos.message

    def test_attributes_not_viable(self) -> None:
        """Test OptionsPosition with non-viable position."""
        pos = OptionsPosition(
            contracts=0,
            premium_cost=0.0,
            risk_amount=0.0,
            position_percent=0.0,
            is_viable=False,
            message="Option too expensive",
        )
        assert pos.contracts == 0
        assert pos.is_viable is False


class TestPositionSizerInit:
    """Tests for PositionSizer initialization."""

    def test_init_defaults(self) -> None:
        """Test initialization with defaults."""
        sizer = PositionSizer(account_equity=10000.0)
        assert sizer.account_equity == 10000.0
        assert sizer.risk_per_trade == 0.01

    def test_init_custom_risk(self) -> None:
        """Test initialization with custom risk."""
        sizer = PositionSizer(account_equity=25000.0, risk_per_trade=0.02)
        assert sizer.account_equity == 25000.0
        assert sizer.risk_per_trade == 0.02

    def test_init_zero_equity_raises(self) -> None:
        """Test that zero equity raises error."""
        with pytest.raises(ValueError, match="Account equity must be positive"):
            PositionSizer(account_equity=0)

    def test_init_negative_equity_raises(self) -> None:
        """Test that negative equity raises error."""
        with pytest.raises(ValueError, match="Account equity must be positive"):
            PositionSizer(account_equity=-1000)

    def test_init_zero_risk_raises(self) -> None:
        """Test that zero risk raises error."""
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            PositionSizer(account_equity=10000, risk_per_trade=0)

    def test_init_excessive_risk_raises(self) -> None:
        """Test that risk > 1 raises error."""
        with pytest.raises(ValueError, match="Risk per trade must be between 0 and 1"):
            PositionSizer(account_equity=10000, risk_per_trade=1.5)


class TestBasicCalculation:
    """Tests for basic position size calculation."""

    def test_basic_calculation(self) -> None:
        """Test basic calculation: $10k account, 1% risk, $100 entry, $95 stop."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.calculate_position_size(entry_price=100.0, stop_price=95.0)

        # Risk per share: $5, Max risk: $100, Shares: 100/5 = 20
        assert result.shares == 20
        assert result.dollar_amount == 2000.0
        assert result.risk_amount == 100.0  # 20 × $5
        assert result.risk_percent == pytest.approx(1.0, rel=0.01)
        assert result.position_percent == pytest.approx(20.0, rel=0.01)

    def test_higher_risk_tolerance(self) -> None:
        """Test with 2% risk tolerance."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_position_size(entry_price=100.0, stop_price=95.0)

        # Risk per share: $5, Max risk: $200, Shares: 200/5 = 40
        assert result.shares == 40
        assert result.dollar_amount == 4000.0
        assert result.risk_amount == 200.0

    def test_smaller_account(self) -> None:
        """Test with smaller account."""
        sizer = PositionSizer(account_equity=5000.0, risk_per_trade=0.01)
        result = sizer.calculate_position_size(entry_price=100.0, stop_price=95.0)

        # Risk per share: $5, Max risk: $50, Shares: 50/5 = 10
        assert result.shares == 10
        assert result.dollar_amount == 1000.0

    def test_override_equity(self) -> None:
        """Test overriding account equity in method."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.calculate_position_size(
            entry_price=100.0,
            stop_price=95.0,
            account_equity=20000.0,  # Override with larger account
        )

        # With $20k account: Max risk: $200, Shares: 200/5 = 40
        assert result.shares == 40
        assert result.dollar_amount == 4000.0


class TestAccountLimitRespected:
    """Tests that position size never exceeds account."""

    def test_respects_account_limit(self) -> None:
        """Position should never exceed account size."""
        sizer = PositionSizer(account_equity=1000.0, risk_per_trade=0.10)  # 10% risk
        result = sizer.calculate_position_size(
            entry_price=100.0,
            stop_price=99.0,  # Only $1 risk per share
        )

        # Max risk: $100, Risk per share: $1, Would be 100 shares = $10,000
        # But account only has $1000, so limited to 10 shares
        assert result.dollar_amount <= 1000.0
        assert result.shares == 10

    def test_high_priced_stock_limited(self) -> None:
        """Test that high-priced stock is limited by account."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.05)
        result = sizer.calculate_position_size(
            entry_price=5000.0,
            stop_price=4900.0,
        )

        # Max risk: $500, Risk per share: $100, Would be 5 shares = $25,000
        # But account only has $10,000, so limited to 2 shares
        assert result.shares <= 2
        assert result.dollar_amount <= 10000.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_handles_zero_risk_raises(self) -> None:
        """Stop = Entry should raise error."""
        sizer = PositionSizer(account_equity=10000.0)
        with pytest.raises(PositionSizingError, match="Stop price cannot equal entry"):
            sizer.calculate_position_size(entry_price=100.0, stop_price=100.0)

    def test_negative_entry_raises(self) -> None:
        """Negative entry price should raise error."""
        sizer = PositionSizer(account_equity=10000.0)
        with pytest.raises(PositionSizingError, match="Entry price must be positive"):
            sizer.calculate_position_size(entry_price=-100.0, stop_price=95.0)

    def test_negative_stop_raises(self) -> None:
        """Negative stop price should raise error."""
        sizer = PositionSizer(account_equity=10000.0)
        with pytest.raises(PositionSizingError, match="Stop price must be positive"):
            sizer.calculate_position_size(entry_price=100.0, stop_price=-95.0)

    def test_zero_entry_raises(self) -> None:
        """Zero entry price should raise error."""
        sizer = PositionSizer(account_equity=10000.0)
        with pytest.raises(PositionSizingError, match="Entry price must be positive"):
            sizer.calculate_position_size(entry_price=0.0, stop_price=95.0)

    def test_short_position_sizing(self) -> None:
        """Test position sizing for short trade (stop above entry)."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.calculate_position_size(
            entry_price=100.0,
            stop_price=105.0,  # Stop above entry for short
        )

        # Risk per share: $5, Max risk: $100, Shares: 20
        assert result.shares == 20
        assert result.risk_amount == 100.0


class TestOptionsSingleContract:
    """Tests for options position sizing with single contract scenarios."""

    def test_options_not_viable_expensive(self) -> None:
        """Test when option is too expensive for risk budget."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.calculate_options_position(option_price=2.50)

        # Cost per contract: $250, Max risk: $100
        # Cannot afford even 1 contract within risk budget
        assert result.contracts == 0
        assert result.is_viable is False
        assert "too expensive" in result.message.lower()

    def test_options_single_contract_viable(self) -> None:
        """Test when exactly 1 contract is viable."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.015)
        result = sizer.calculate_options_position(option_price=1.00)

        # Cost per contract: $100, Max risk: $150
        # Can afford 1 contract
        assert result.contracts == 1
        assert result.premium_cost == 100.0
        assert result.is_viable is True


class TestOptionsViable:
    """Tests for viable options positions."""

    def test_options_viable_multiple_contracts(self) -> None:
        """Test viable options position with multiple contracts."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_options_position(option_price=1.00)

        # Cost per contract: $100, Max risk: $200
        # Can afford 2 contracts
        assert result.contracts == 2
        assert result.premium_cost == 200.0
        assert result.risk_amount == 200.0  # Full premium at risk
        assert result.is_viable is True
        assert result.position_percent == pytest.approx(2.0, rel=0.01)

    def test_options_override_equity(self) -> None:
        """Test options with overridden equity."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)
        result = sizer.calculate_options_position(option_price=1.00, account_equity=20000.0)

        # With $20k: Max risk: $400, Contracts: 4
        assert result.contracts == 4
        assert result.premium_cost == 400.0

    def test_options_zero_price_not_viable(self) -> None:
        """Test options with zero price."""
        sizer = PositionSizer(account_equity=10000.0)
        result = sizer.calculate_options_position(option_price=0.0)

        assert result.is_viable is False
        assert result.contracts == 0

    def test_options_negative_price_not_viable(self) -> None:
        """Test options with negative price."""
        sizer = PositionSizer(account_equity=10000.0)
        result = sizer.calculate_options_position(option_price=-1.0)

        assert result.is_viable is False
        assert result.contracts == 0


class TestKellyCriterion:
    """Tests for Kelly Criterion calculation."""

    def test_kelly_criterion_basic(self) -> None:
        """Test Kelly calculation with 60% win rate."""
        # 60% win rate, 1.5 avg win, 1 avg loss
        # Odds = 1.5, p = 0.6, q = 0.4
        # Kelly = (1.5 × 0.6 - 0.4) / 1.5 = (0.9 - 0.4) / 1.5 = 0.5 / 1.5 = 0.333
        result = PositionSizer.calculate_kelly_size(
            win_rate=0.6,
            avg_win=1.5,
            avg_loss=1.0,
        )
        assert result == pytest.approx(0.333, rel=0.01)

    def test_kelly_50_50_even_odds(self) -> None:
        """Test Kelly with 50/50 odds and even payoff."""
        # 50% win rate, 1:1 odds = 0 Kelly (no edge)
        result = PositionSizer.calculate_kelly_size(
            win_rate=0.5,
            avg_win=1.0,
            avg_loss=1.0,
        )
        assert result == pytest.approx(0.0, abs=0.01)

    def test_kelly_high_win_rate(self) -> None:
        """Test Kelly with high win rate."""
        # 80% win rate, 1:1 odds
        # Kelly = (1 × 0.8 - 0.2) / 1 = 0.6
        result = PositionSizer.calculate_kelly_size(
            win_rate=0.8,
            avg_win=1.0,
            avg_loss=1.0,
        )
        assert result == pytest.approx(0.6, rel=0.01)

    def test_kelly_negative_edge_returns_zero(self) -> None:
        """Test Kelly returns 0 for negative edge."""
        # 30% win rate, 1:1 odds = negative Kelly
        result = PositionSizer.calculate_kelly_size(
            win_rate=0.3,
            avg_win=1.0,
            avg_loss=1.0,
        )
        assert result == 0.0

    def test_kelly_invalid_win_rate_raises(self) -> None:
        """Test invalid win rate raises error."""
        with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
            PositionSizer.calculate_kelly_size(win_rate=1.5, avg_win=1.0, avg_loss=1.0)

        with pytest.raises(ValueError, match="Win rate must be between 0 and 1"):
            PositionSizer.calculate_kelly_size(win_rate=-0.1, avg_win=1.0, avg_loss=1.0)

    def test_kelly_invalid_avg_win_raises(self) -> None:
        """Test invalid avg win raises error."""
        with pytest.raises(ValueError, match="Average win must be positive"):
            PositionSizer.calculate_kelly_size(win_rate=0.6, avg_win=0.0, avg_loss=1.0)

        with pytest.raises(ValueError, match="Average win must be positive"):
            PositionSizer.calculate_kelly_size(win_rate=0.6, avg_win=-1.0, avg_loss=1.0)

    def test_kelly_invalid_avg_loss_raises(self) -> None:
        """Test invalid avg loss raises error."""
        with pytest.raises(ValueError, match="Average loss must be positive"):
            PositionSizer.calculate_kelly_size(win_rate=0.6, avg_win=1.0, avg_loss=0.0)

        with pytest.raises(ValueError, match="Average loss must be positive"):
            PositionSizer.calculate_kelly_size(win_rate=0.6, avg_win=1.0, avg_loss=-1.0)


class TestHalfKelly:
    """Tests for half-Kelly calculation."""

    def test_half_kelly_is_half(self) -> None:
        """Verify half-Kelly is exactly half of full Kelly."""
        win_rate, avg_win, avg_loss = 0.6, 1.5, 1.0

        full_kelly = PositionSizer.calculate_kelly_size(win_rate, avg_win, avg_loss)
        half_kelly = PositionSizer.calculate_half_kelly(win_rate, avg_win, avg_loss)

        assert half_kelly == pytest.approx(full_kelly / 2, rel=0.001)

    def test_half_kelly_safer(self) -> None:
        """Verify half-Kelly recommends smaller position."""
        win_rate, avg_win, avg_loss = 0.7, 2.0, 1.0

        full_kelly = PositionSizer.calculate_kelly_size(win_rate, avg_win, avg_loss)
        half_kelly = PositionSizer.calculate_half_kelly(win_rate, avg_win, avg_loss)

        assert half_kelly < full_kelly
        assert half_kelly > 0


class TestPositionForSignal:
    """Tests for position_for_signal helper method."""

    def test_basic_signal(self) -> None:
        """Test basic signal without take profit."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.position_for_signal(entry_price=100.0, stop_price=95.0)

        assert result["shares"] == 20
        assert result["entry_price"] == 100.0
        assert result["stop_price"] == 95.0
        assert "take_profit" not in result
        assert "risk_reward_ratio" not in result

    def test_signal_with_take_profit(self) -> None:
        """Test signal with take profit target."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.position_for_signal(
            entry_price=100.0,
            stop_price=95.0,
            take_profit=110.0,
        )

        assert result["shares"] == 20
        assert result["take_profit"] == 110.0
        assert result["risk_reward_ratio"] == pytest.approx(2.0, rel=0.01)  # $10/$5
        assert result["reward_amount"] == 200.0  # 20 × $10

    def test_signal_override_equity(self) -> None:
        """Test signal with overridden equity."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.position_for_signal(
            entry_price=100.0,
            stop_price=95.0,
            account_equity=20000.0,
        )

        assert result["shares"] == 40  # Double due to doubled equity


class TestPropertyBasedBehavior:
    """Property-based tests for invariants."""

    @pytest.mark.parametrize(
        "entry,stop,expected_shares",
        [
            (100.0, 95.0, 20),  # $5 risk
            (50.0, 48.0, 50),  # $2 risk
            (200.0, 190.0, 10),  # $10 risk
        ],
    )
    def test_position_risk_never_exceeds_budget(
        self, entry: float, stop: float, expected_shares: int
    ) -> None:
        """Position risk should never exceed risk_per_trade × account."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        result = sizer.calculate_position_size(entry_price=entry, stop_price=stop)

        max_risk = 10000.0 * 0.01  # $100
        assert result.risk_amount <= max_risk + 0.01  # Small tolerance for rounding
        assert result.shares == expected_shares

    def test_shares_always_non_negative(self) -> None:
        """Shares should always be non-negative integer."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)

        # Various scenarios
        for entry in [10.0, 50.0, 100.0, 500.0]:
            for stop_offset in [0.5, 1.0, 5.0, 10.0]:
                stop = entry - stop_offset
                if stop > 0:
                    result = sizer.calculate_position_size(entry_price=entry, stop_price=stop)
                    assert result.shares >= 0
                    assert isinstance(result.shares, int)

    def test_dollar_amount_never_exceeds_equity(self) -> None:
        """Dollar amount should never exceed account equity."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.05)

        # High risk tolerance + small stop should not exceed account
        result = sizer.calculate_position_size(entry_price=1000.0, stop_price=990.0)
        assert result.dollar_amount <= 10000.0

    def test_options_risk_matches_premium(self) -> None:
        """For long options, risk should equal premium cost."""
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.03)
        result = sizer.calculate_options_position(option_price=1.50)

        if result.is_viable:
            assert result.risk_amount == result.premium_cost
