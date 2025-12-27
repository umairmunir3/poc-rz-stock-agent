"""Position Sizing Calculator - calculates position sizes based on risk parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PositionSize:
    """Result of position size calculation for stocks.

    Attributes:
        shares: Number of shares to purchase.
        dollar_amount: Total cost of position in dollars.
        risk_amount: Amount at risk (entry - stop) × shares.
        risk_percent: Percentage of account at risk.
        position_percent: Position size as percentage of account.
    """

    shares: int
    dollar_amount: float
    risk_amount: float
    risk_percent: float
    position_percent: float


@dataclass
class OptionsPosition:
    """Result of options position size calculation.

    Attributes:
        contracts: Number of contracts to purchase.
        premium_cost: Total premium cost for position.
        risk_amount: Amount at risk (full premium for long options).
        position_percent: Position size as percentage of account.
        is_viable: Whether the position is viable (can afford at least 1 contract).
        message: Explanation if position is not viable.
    """

    contracts: int
    premium_cost: float
    risk_amount: float
    position_percent: float
    is_viable: bool
    message: str


class PositionSizingError(Exception):
    """Raised when position sizing calculation fails."""


class PositionSizer:
    """Calculates position sizes based on risk parameters.

    Uses fixed fractional position sizing to determine the number of shares
    or contracts based on account equity and risk tolerance.

    Example:
        >>> sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.01)
        >>> result = sizer.calculate_position_size(entry_price=100.0, stop_price=95.0)
        >>> print(f"Buy {result.shares} shares for ${result.dollar_amount}")
        Buy 20 shares for $2000.0
    """

    def __init__(
        self,
        account_equity: float,
        risk_per_trade: float = 0.01,
    ) -> None:
        """Initialize the position sizer.

        Args:
            account_equity: Total account equity in dollars.
            risk_per_trade: Maximum risk per trade as decimal (0.01 = 1%).

        Raises:
            ValueError: If account_equity is <= 0 or risk_per_trade is not between 0 and 1.
        """
        if account_equity <= 0:
            raise ValueError("Account equity must be positive")
        if not 0 < risk_per_trade <= 1:
            raise ValueError("Risk per trade must be between 0 and 1 (exclusive of 0)")

        self.account_equity = account_equity
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        account_equity: float | None = None,
    ) -> PositionSize:
        """Calculate position size for a stock trade using fixed fractional sizing.

        Args:
            entry_price: Planned entry price per share.
            stop_price: Stop loss price per share.
            account_equity: Override account equity (uses init value if None).

        Returns:
            PositionSize with calculated values.

        Raises:
            PositionSizingError: If stop equals entry (zero risk) or entry/stop are invalid.
        """
        equity = account_equity if account_equity is not None else self.account_equity

        if entry_price <= 0:
            raise PositionSizingError("Entry price must be positive")
        if stop_price <= 0:
            raise PositionSizingError("Stop price must be positive")
        if entry_price == stop_price:
            raise PositionSizingError("Stop price cannot equal entry price (zero risk)")

        # Calculate risk per share (absolute value for both long and short)
        risk_per_share = abs(entry_price - stop_price)

        # Maximum risk amount based on account and risk tolerance
        max_risk_amount = equity * self.risk_per_trade

        # Calculate maximum shares based on risk
        max_shares_by_risk = max_risk_amount / risk_per_share

        # Calculate maximum shares based on account size (can't exceed equity)
        max_shares_by_equity = equity / entry_price

        # Use the smaller of the two limits
        max_shares = min(max_shares_by_risk, max_shares_by_equity)

        # Round down to whole shares
        shares = max(0, int(math.floor(max_shares)))

        # Calculate actual values
        dollar_amount = shares * entry_price
        risk_amount = shares * risk_per_share
        risk_percent = (risk_amount / equity) * 100 if equity > 0 else 0.0
        position_percent = (dollar_amount / equity) * 100 if equity > 0 else 0.0

        return PositionSize(
            shares=shares,
            dollar_amount=dollar_amount,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            position_percent=position_percent,
        )

    def calculate_options_position(
        self,
        option_price: float,
        account_equity: float | None = None,
    ) -> OptionsPosition:
        """Calculate position size for an options trade.

        For long options, the full premium is at risk.

        Args:
            option_price: Price per contract (premium × 100 shares).
            account_equity: Override account equity (uses init value if None).

        Returns:
            OptionsPosition with calculated values.
        """
        equity = account_equity if account_equity is not None else self.account_equity

        if option_price <= 0:
            return OptionsPosition(
                contracts=0,
                premium_cost=0.0,
                risk_amount=0.0,
                position_percent=0.0,
                is_viable=False,
                message="Option price must be positive",
            )

        # Premium cost per contract (option price × 100 shares per contract)
        cost_per_contract = option_price * 100

        # Maximum risk amount
        max_risk_amount = equity * self.risk_per_trade

        # Calculate contracts based on risk budget
        max_contracts = max_risk_amount / cost_per_contract

        # Round down to whole contracts
        contracts = max(0, int(math.floor(max_contracts)))

        if contracts == 0:
            # Check if we can't even afford one contract within risk budget
            if cost_per_contract > max_risk_amount:
                return OptionsPosition(
                    contracts=0,
                    premium_cost=0.0,
                    risk_amount=0.0,
                    position_percent=0.0,
                    is_viable=False,
                    message=f"Option too expensive: ${cost_per_contract:.2f} per contract "
                    f"exceeds risk budget of ${max_risk_amount:.2f}",
                )
            return OptionsPosition(
                contracts=0,
                premium_cost=0.0,
                risk_amount=0.0,
                position_percent=0.0,
                is_viable=False,
                message="Cannot afford any contracts",
            )

        # Calculate actual values
        premium_cost = contracts * cost_per_contract
        risk_amount = premium_cost  # Full premium at risk for long options
        position_percent = (premium_cost / equity) * 100 if equity > 0 else 0.0

        return OptionsPosition(
            contracts=contracts,
            premium_cost=premium_cost,
            risk_amount=risk_amount,
            position_percent=position_percent,
            is_viable=True,
            message=f"Position viable: {contracts} contract(s) for ${premium_cost:.2f}",
        )

    @staticmethod
    def calculate_kelly_size(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate optimal position size using Kelly Criterion.

        The Kelly Criterion determines the optimal fraction of capital to bet
        to maximize long-term growth rate. In practice, half-Kelly is often
        used for a more conservative approach.

        Formula: f* = (bp - q) / b
        Where:
            - f* = fraction of capital to bet
            - b = odds received (avg_win / avg_loss)
            - p = probability of winning (win_rate)
            - q = probability of losing (1 - win_rate)

        Args:
            win_rate: Probability of winning (0 to 1).
            avg_win: Average profit per winning trade (positive number).
            avg_loss: Average loss per losing trade (positive number).

        Returns:
            Optimal fraction of capital to risk (0 to 1).
            Returns 0 if Kelly suggests not betting.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not 0 <= win_rate <= 1:
            raise ValueError("Win rate must be between 0 and 1")
        if avg_win <= 0:
            raise ValueError("Average win must be positive")
        if avg_loss <= 0:
            raise ValueError("Average loss must be positive")

        # Calculate odds ratio (b in Kelly formula)
        odds = avg_win / avg_loss

        # Calculate probability of losing
        loss_rate = 1 - win_rate

        # Kelly formula: f* = (bp - q) / b = (odds × win_rate - loss_rate) / odds
        kelly_fraction = (odds * win_rate - loss_rate) / odds

        # Kelly can be negative if edge is negative (don't bet)
        return max(0.0, kelly_fraction)

    @staticmethod
    def calculate_half_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Calculate half-Kelly position size.

        Half-Kelly is a more conservative approach that uses half the
        recommended Kelly fraction, reducing volatility while still
        capturing most of the growth.

        Args:
            win_rate: Probability of winning (0 to 1).
            avg_win: Average profit per winning trade.
            avg_loss: Average loss per losing trade.

        Returns:
            Half of the optimal Kelly fraction.
        """
        return PositionSizer.calculate_kelly_size(win_rate, avg_win, avg_loss) / 2

    def position_for_signal(
        self,
        entry_price: float,
        stop_price: float,
        take_profit: float | None = None,
        account_equity: float | None = None,
    ) -> dict:
        """Calculate complete position details for a trade signal.

        Args:
            entry_price: Planned entry price.
            stop_price: Stop loss price.
            take_profit: Target price (optional).
            account_equity: Override account equity.

        Returns:
            Dictionary with position details and risk/reward analysis.
        """
        position = self.calculate_position_size(entry_price, stop_price, account_equity)
        equity = account_equity if account_equity is not None else self.account_equity

        result = {
            "shares": position.shares,
            "dollar_amount": position.dollar_amount,
            "risk_amount": position.risk_amount,
            "risk_percent": position.risk_percent,
            "position_percent": position.position_percent,
            "entry_price": entry_price,
            "stop_price": stop_price,
        }

        if take_profit is not None:
            reward_per_share = abs(take_profit - entry_price)
            risk_per_share = abs(entry_price - stop_price)
            reward_amount = position.shares * reward_per_share

            result.update(
                {
                    "take_profit": take_profit,
                    "reward_amount": reward_amount,
                    "risk_reward_ratio": (
                        reward_per_share / risk_per_share if risk_per_share > 0 else 0
                    ),
                    "reward_percent": (reward_amount / equity) * 100 if equity > 0 else 0.0,
                }
            )

        return result
