"""Backtesting Engine - simulates strategy performance on historical data."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from src.risk.position_sizing import PositionSizer
    from src.strategies.base import Signal, Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Represents a single trade executed during backtesting.

    Attributes:
        trade_id: Unique trade identifier.
        symbol: Stock ticker symbol.
        direction: Trade direction (LONG or SHORT).
        entry_date: Date of entry.
        entry_price: Entry price after slippage.
        exit_date: Date of exit (None if still open).
        exit_price: Exit price after slippage (None if still open).
        shares: Number of shares traded.
        stop_loss: Stop loss price level.
        take_profit: Take profit price level.
        exit_reason: Why the trade was exited.
        pnl: Realized profit/loss in dollars.
        pnl_percent: Realized profit/loss as percentage.
        commission: Total commission paid.
        signal_score: Original signal score.
        strategy: Name of the strategy that generated the signal.
    """

    trade_id: int
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_date: date
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    strategy: str
    signal_score: int
    exit_date: date | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    commission: float = 0.0

    @property
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def hold_days(self) -> int:
        """Calculate holding period in days."""
        if self.exit_date is None:
            return 0
        return (self.exit_date - self.entry_date).days

    @property
    def r_multiple(self) -> float:
        """Calculate R-multiple (PnL relative to initial risk).

        Returns:
            R-multiple where 1R = initial risk amount.
        """
        risk_per_share = abs(self.entry_price - self.stop_loss)
        if risk_per_share == 0:
            return 0.0
        pnl_per_share = self.pnl / self.shares if self.shares > 0 else 0.0
        return pnl_per_share / risk_per_share


@dataclass
class PerformanceMetrics:
    """Performance metrics for a backtest.

    Attributes:
        total_return: Total return as decimal (0.10 = 10%).
        cagr: Compound annual growth rate.
        sharpe_ratio: Sharpe ratio (risk-adjusted return).
        sortino_ratio: Sortino ratio (downside risk-adjusted return).
        max_drawdown: Maximum drawdown as decimal.
        win_rate: Winning trade percentage (0-1).
        avg_win: Average winning trade percentage.
        avg_loss: Average losing trade percentage.
        profit_factor: Gross profits / Gross losses.
        expectancy: Expected R-multiple per trade.
        total_trades: Total number of trades.
        avg_hold_days: Average holding period in days.
    """

    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0

    @classmethod
    def from_trades(
        cls,
        trades: list[BacktestTrade],
        equity_curve: pd.DataFrame,
        initial_capital: float,
        trading_days: int,
    ) -> PerformanceMetrics:
        """Calculate metrics from trades and equity curve.

        Args:
            trades: List of completed trades.
            equity_curve: DataFrame with equity values.
            initial_capital: Starting capital.
            trading_days: Number of trading days in backtest.

        Returns:
            Calculated performance metrics.
        """
        if not trades or equity_curve.empty:
            return cls()

        closed_trades = [t for t in trades if not t.is_open]
        if not closed_trades:
            return cls()

        # Basic metrics
        total_trades = len(closed_trades)
        winners = [t for t in closed_trades if t.is_winner]
        losers = [t for t in closed_trades if not t.is_winner]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0.0

        # Average win/loss percentages
        avg_win = (
            sum(t.pnl_percent for t in winners) / len(winners) if winners else 0.0
        )
        avg_loss = (
            abs(sum(t.pnl_percent for t in losers) / len(losers)) if losers else 0.0
        )

        # Profit factor
        gross_profits = sum(t.pnl for t in winners)
        gross_losses = abs(sum(t.pnl for t in losers))
        profit_factor = (
            gross_profits / gross_losses if gross_losses > 0 else float("inf")
        )

        # Expectancy (average R-multiple)
        r_multiples = [t.r_multiple for t in closed_trades]
        expectancy = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0

        # Average hold days
        hold_days_list = [t.hold_days for t in closed_trades if t.hold_days > 0]
        avg_hold_days = (
            sum(hold_days_list) / len(hold_days_list) if hold_days_list else 0.0
        )

        # Total return
        final_equity = equity_curve["equity"].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital

        # CAGR
        years = trading_days / 252 if trading_days > 0 else 1.0
        cagr = (
            ((final_equity / initial_capital) ** (1 / years) - 1) if years > 0 else 0.0
        )

        # Daily returns for Sharpe/Sortino
        equity_values = equity_curve["equity"].values
        daily_returns = np.diff(equity_values) / equity_values[:-1]  # type: ignore[arg-type]

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(
                252
            )
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (using downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1 and np.std(negative_returns) > 0:
            sortino_ratio = (
                np.mean(daily_returns) / np.std(negative_returns)
            ) * np.sqrt(252)
        else:
            sortino_ratio = 0.0

        # Max drawdown
        max_drawdown = equity_curve["drawdown"].max() if "drawdown" in equity_curve else 0.0

        return cls(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_trades=total_trades,
            avg_hold_days=avg_hold_days,
        )


@dataclass
class BacktestResult:
    """Results from running a backtest.

    Attributes:
        equity_curve: DataFrame with date, equity, drawdown.
        trades: List of all trades executed.
        metrics: Calculated performance metrics.
        signals_generated: Total signals generated during backtest.
        trades_executed: Number of trades actually executed.
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital.
        final_equity: Ending equity.
    """

    equity_curve: pd.DataFrame
    trades: list[BacktestTrade]
    metrics: PerformanceMetrics
    signals_generated: int = 0
    trades_executed: int = 0
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = 0.0
    final_equity: float = 0.0


@dataclass
class WalkForwardWindow:
    """A single window in walk-forward optimization.

    Attributes:
        train_start: Training period start date.
        train_end: Training period end date.
        test_start: Test period start date.
        test_end: Test period end date.
        train_result: Backtest result on training period.
        test_result: Backtest result on test period.
    """

    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_result: BacktestResult | None = None
    test_result: BacktestResult | None = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization.

    Attributes:
        windows: List of walk-forward windows.
        combined_metrics: Combined metrics across all test periods.
        in_sample_metrics: Combined metrics for all training periods.
        out_of_sample_metrics: Combined metrics for all test periods.
        robustness_ratio: OOS return / IS return (closer to 1 = better).
    """

    windows: list[WalkForwardWindow]
    combined_metrics: PerformanceMetrics
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics
    robustness_ratio: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtest execution.

    Attributes:
        slippage_percent: Slippage as percentage of price (0.001 = 0.1%).
        commission_per_trade: Commission per trade in dollars.
        use_next_bar_open: Whether to use next bar open for fills.
        max_positions: Maximum concurrent positions.
        min_signal_score: Minimum signal score to execute.
    """

    slippage_percent: float = 0.001  # 0.1%
    commission_per_trade: float = 1.0
    use_next_bar_open: bool = True
    max_positions: int = 10
    min_signal_score: int = 0


class BacktestEngine:
    """Engine for backtesting trading strategies.

    Simulates strategy performance on historical data with realistic
    execution modeling including slippage, commissions, and position sizing.

    Example:
        >>> engine = BacktestEngine(strategies=[my_strategy], position_sizer=sizer)
        >>> result = engine.run(["AAPL", "MSFT"], start_date, end_date)
        >>> print(f"Total return: {result.metrics.total_return:.2%}")
    """

    def __init__(
        self,
        strategies: list[Strategy],
        position_sizer: PositionSizer,
        config: BacktestConfig | None = None,
        data_provider: Any | None = None,
    ) -> None:
        """Initialize the backtest engine.

        Args:
            strategies: List of strategies to run.
            position_sizer: Position sizing calculator.
            config: Backtest configuration.
            data_provider: Optional data provider for historical data.
        """
        self.strategies = strategies
        self.position_sizer = position_sizer
        self.config = config or BacktestConfig()
        self.data_provider = data_provider

        # Internal state
        self._trades: list[BacktestTrade] = []
        self._open_positions: dict[str, BacktestTrade] = {}
        self._equity: float = 0.0
        self._trade_counter: int = 0
        self._signals_generated: int = 0

    def run(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 10000.0,
        historical_data: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            symbols: List of symbols to backtest.
            start_date: Backtest start date.
            end_date: Backtest end date.
            initial_capital: Starting capital.
            historical_data: Pre-loaded historical data (symbol -> DataFrame).

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        # Initialize state
        self._trades = []
        self._open_positions = {}
        self._equity = initial_capital
        self._trade_counter = 0
        self._signals_generated = 0

        # Get historical data
        if historical_data is None:
            historical_data = self._load_historical_data(symbols, start_date, end_date)

        # Build date index from available data
        all_dates = self._get_trading_dates(historical_data, start_date, end_date)
        if not all_dates:
            return self._empty_result(initial_capital, start_date, end_date)

        # Track equity curve
        equity_history: list[dict[str, Any]] = []

        # Run simulation day by day
        for current_date in all_dates:
            # Update position values and check exits
            self._update_positions(current_date, historical_data)

            # Check for new signals if we have capacity
            if len(self._open_positions) < self.config.max_positions:
                self._scan_for_signals(current_date, symbols, historical_data)

            # Record equity
            equity_history.append(
                {
                    "date": current_date,
                    "equity": self._equity,
                }
            )

        # Close any remaining open positions at end
        self._close_all_positions(all_dates[-1], historical_data)

        # Build equity curve DataFrame
        equity_df = pd.DataFrame(equity_history)
        equity_df = self._calculate_drawdown(equity_df)

        # Calculate metrics
        trading_days = len(all_dates)
        metrics = PerformanceMetrics.from_trades(
            self._trades, equity_df, initial_capital, trading_days
        )

        return BacktestResult(
            equity_curve=equity_df,
            trades=self._trades,
            metrics=metrics,
            signals_generated=self._signals_generated,
            trades_executed=len(self._trades),
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_equity=self._equity,
        )

    def walk_forward(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        train_months: int = 12,
        test_months: int = 3,
        initial_capital: float = 10000.0,
        historical_data: dict[str, pd.DataFrame] | None = None,
    ) -> WalkForwardResult:
        """Run walk-forward optimization.

        Divides the data into train/test windows and rolls forward.

        Args:
            symbols: List of symbols to backtest.
            start_date: Start date.
            end_date: End date.
            train_months: Training period in months.
            test_months: Test period in months.
            initial_capital: Starting capital.
            historical_data: Pre-loaded historical data.

        Returns:
            WalkForwardResult with all window results.
        """
        # Load data if not provided
        if historical_data is None:
            historical_data = self._load_historical_data(symbols, start_date, end_date)

        # Generate windows
        windows = self._generate_walk_forward_windows(
            start_date, end_date, train_months, test_months
        )

        if not windows:
            return WalkForwardResult(
                windows=[],
                combined_metrics=PerformanceMetrics(),
                in_sample_metrics=PerformanceMetrics(),
                out_of_sample_metrics=PerformanceMetrics(),
                robustness_ratio=0.0,
            )

        # Run backtest on each window
        all_test_trades: list[BacktestTrade] = []
        all_train_trades: list[BacktestTrade] = []
        test_equity_curves: list[pd.DataFrame] = []
        train_equity_curves: list[pd.DataFrame] = []

        for window in windows:
            # Training period
            train_result = self.run(
                symbols,
                window.train_start,
                window.train_end,
                initial_capital,
                historical_data,
            )
            window.train_result = train_result
            all_train_trades.extend(train_result.trades)
            train_equity_curves.append(train_result.equity_curve)

            # Test period
            test_result = self.run(
                symbols,
                window.test_start,
                window.test_end,
                initial_capital,
                historical_data,
            )
            window.test_result = test_result
            all_test_trades.extend(test_result.trades)
            test_equity_curves.append(test_result.equity_curve)

        # Combine metrics
        combined_test_equity = pd.concat(test_equity_curves, ignore_index=True)
        combined_train_equity = pd.concat(train_equity_curves, ignore_index=True)

        total_test_days = len(combined_test_equity)
        total_train_days = len(combined_train_equity)

        out_of_sample_metrics = PerformanceMetrics.from_trades(
            all_test_trades, combined_test_equity, initial_capital, total_test_days
        )
        in_sample_metrics = PerformanceMetrics.from_trades(
            all_train_trades, combined_train_equity, initial_capital, total_train_days
        )

        # Robustness ratio
        robustness_ratio = 0.0
        if in_sample_metrics.total_return > 0:
            robustness_ratio = (
                out_of_sample_metrics.total_return / in_sample_metrics.total_return
            )

        return WalkForwardResult(
            windows=windows,
            combined_metrics=out_of_sample_metrics,
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=out_of_sample_metrics,
            robustness_ratio=robustness_ratio,
        )

    # -------------------------------------------------------------------------
    # Reporting Methods
    # -------------------------------------------------------------------------

    def generate_report(self, result: BacktestResult) -> str:
        """Generate a markdown report of backtest results.

        Args:
            result: BacktestResult to report on.

        Returns:
            Markdown formatted report string.
        """
        m = result.metrics
        report = f"""# Backtest Report

## Summary
- **Period**: {result.start_date} to {result.end_date}
- **Initial Capital**: ${result.initial_capital:,.2f}
- **Final Equity**: ${result.final_equity:,.2f}
- **Total Return**: {m.total_return:.2%}

## Performance Metrics

| Metric | Value |
|--------|-------|
| CAGR | {m.cagr:.2%} |
| Sharpe Ratio | {m.sharpe_ratio:.2f} |
| Sortino Ratio | {m.sortino_ratio:.2f} |
| Max Drawdown | {m.max_drawdown:.2%} |

## Trade Statistics

| Metric | Value |
|--------|-------|
| Total Trades | {m.total_trades} |
| Win Rate | {m.win_rate:.1%} |
| Average Win | {m.avg_win:.2%} |
| Average Loss | {m.avg_loss:.2%} |
| Profit Factor | {m.profit_factor:.2f} |
| Expectancy (R) | {m.expectancy:.2f} |
| Avg Hold Days | {m.avg_hold_days:.1f} |

## Signals
- **Signals Generated**: {result.signals_generated}
- **Trades Executed**: {result.trades_executed}
"""
        return report

    def generate_html_report(self, result: BacktestResult) -> str:
        """Generate an HTML report of backtest results.

        Args:
            result: BacktestResult to report on.

        Returns:
            HTML formatted report string.
        """
        m = result.metrics
        report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>Backtest Report</h1>

    <h2>Summary</h2>
    <table>
        <tr><td>Period</td><td>{result.start_date} to {result.end_date}</td></tr>
        <tr><td>Initial Capital</td><td>${result.initial_capital:,.2f}</td></tr>
        <tr><td>Final Equity</td><td>${result.final_equity:,.2f}</td></tr>
        <tr><td>Total Return</td><td class="{'positive' if m.total_return >= 0 else 'negative'}">{m.total_return:.2%}</td></tr>
    </table>

    <h2>Performance Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>CAGR</td><td>{m.cagr:.2%}</td></tr>
        <tr><td>Sharpe Ratio</td><td>{m.sharpe_ratio:.2f}</td></tr>
        <tr><td>Sortino Ratio</td><td>{m.sortino_ratio:.2f}</td></tr>
        <tr><td>Max Drawdown</td><td class="negative">{m.max_drawdown:.2%}</td></tr>
    </table>

    <h2>Trade Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Trades</td><td>{m.total_trades}</td></tr>
        <tr><td>Win Rate</td><td>{m.win_rate:.1%}</td></tr>
        <tr><td>Average Win</td><td class="positive">{m.avg_win:.2%}</td></tr>
        <tr><td>Average Loss</td><td class="negative">{m.avg_loss:.2%}</td></tr>
        <tr><td>Profit Factor</td><td>{m.profit_factor:.2f}</td></tr>
        <tr><td>Expectancy (R)</td><td>{m.expectancy:.2f}</td></tr>
        <tr><td>Avg Hold Days</td><td>{m.avg_hold_days:.1f}</td></tr>
    </table>
</body>
</html>"""
        return report

    def plot_equity_curve(self, result: BacktestResult) -> Figure:
        """Plot the equity curve.

        Args:
            result: BacktestResult containing equity curve.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(result.equity_curve["date"], result.equity_curve["equity"], label="Equity")
        ax.axhline(y=result.initial_capital, color="gray", linestyle="--", label="Initial Capital")

        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.set_title("Equity Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_drawdown(self, result: BacktestResult) -> Figure:
        """Plot the drawdown chart.

        Args:
            result: BacktestResult containing equity curve with drawdown.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.fill_between(
            result.equity_curve["date"],
            -result.equity_curve["drawdown"] * 100,
            0,
            alpha=0.5,
            color="red",
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Drawdown")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_returns(self, result: BacktestResult) -> Figure:
        """Plot monthly returns heatmap.

        Args:
            result: BacktestResult containing equity curve.

        Returns:
            Matplotlib figure.
        """
        import matplotlib.pyplot as plt

        # Calculate monthly returns
        df = result.equity_curve.copy()
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        monthly = df["equity"].resample("ME").last()
        monthly_returns = monthly.pct_change().dropna()

        if monthly_returns.empty:
            # Return empty figure if no data
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title("Monthly Returns (No Data)")
            return fig

        # Create pivot table
        returns_df = pd.DataFrame(
            {
                "year": monthly_returns.index.year,  # type: ignore[attr-defined]
                "month": monthly_returns.index.month,  # type: ignore[attr-defined]
                "return": monthly_returns.values,
            }
        )

        pivot = returns_df.pivot(index="year", columns="month", values="return")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 6))

        n_cols = len(pivot.columns)
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.15, vmax=0.15)

        # Labels - only show months that exist in data
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_labels = [month_names[int(m) - 1] for m in pivot.columns]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(month_labels)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Return")

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(n_cols):
                val = pivot.values[i, j]
                if not math.isnan(val):
                    ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8)

        ax.set_title("Monthly Returns")
        plt.tight_layout()
        return fig

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _load_historical_data(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> dict[str, pd.DataFrame]:
        """Load historical data for symbols.

        Args:
            symbols: List of symbols.
            start_date: Start date.
            end_date: End date.

        Returns:
            Dictionary mapping symbols to DataFrames.
        """
        if self.data_provider is None:
            return {}

        data = {}
        for symbol in symbols:
            try:
                df = self.data_provider.get_daily_ohlcv(symbol, start_date, end_date)
                if df is not None and not df.empty:
                    data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")

        return data

    def _get_trading_dates(
        self, historical_data: dict[str, pd.DataFrame], start_date: date, end_date: date
    ) -> list[date]:
        """Get list of trading dates from historical data.

        Args:
            historical_data: Historical data by symbol.
            start_date: Start date.
            end_date: End date.

        Returns:
            Sorted list of trading dates.
        """
        all_dates: set[date] = set()

        for df in historical_data.values():
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"]).dt.date
            else:
                dates = df.index.date if hasattr(df.index, "date") else []  # type: ignore[assignment]

            for d in dates:
                if start_date <= d <= end_date:
                    all_dates.add(d)

        return sorted(all_dates)

    def _get_price_on_date(
        self, symbol: str, target_date: date, historical_data: dict[str, pd.DataFrame]
    ) -> dict[str, float] | None:
        """Get OHLC prices for a symbol on a specific date.

        Args:
            symbol: Stock symbol.
            target_date: Target date.
            historical_data: Historical data.

        Returns:
            Dictionary with open, high, low, close or None.
        """
        if symbol not in historical_data:
            return None

        df = historical_data[symbol]

        # Get the row for this date
        if "date" in df.columns:
            mask = pd.to_datetime(df["date"]).dt.date == target_date
            rows = df[mask]
        else:
            try:
                rows = df.loc[[pd.Timestamp(target_date)]]
            except KeyError:
                return None

        if rows.empty:
            return None

        row = rows.iloc[0]
        return {
            "open": float(row.get("open", row.get("Open", 0))),
            "high": float(row.get("high", row.get("High", 0))),
            "low": float(row.get("low", row.get("Low", 0))),
            "close": float(row.get("close", row.get("Close", 0))),
        }

    def _update_positions(
        self, current_date: date, historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Update open positions and check for exits.

        Args:
            current_date: Current simulation date.
            historical_data: Historical price data.
        """
        symbols_to_close = []

        for symbol, trade in self._open_positions.items():
            prices = self._get_price_on_date(symbol, current_date, historical_data)
            if prices is None:
                continue

            high = prices["high"]
            low = prices["low"]

            # Check stop loss
            if trade.direction == "LONG":
                if low <= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, current_date, "stop_loss")
                    symbols_to_close.append(symbol)
                    continue
                if high >= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, current_date, "take_profit")
                    symbols_to_close.append(symbol)
                    continue
            else:  # SHORT
                if high >= trade.stop_loss:
                    self._close_trade(trade, trade.stop_loss, current_date, "stop_loss")
                    symbols_to_close.append(symbol)
                    continue
                if low <= trade.take_profit:
                    self._close_trade(trade, trade.take_profit, current_date, "take_profit")
                    symbols_to_close.append(symbol)
                    continue

            # Note: Unrealized P&L could be tracked here but equity updates on close

        # Remove closed positions
        for symbol in symbols_to_close:
            del self._open_positions[symbol]

    def _scan_for_signals(
        self,
        current_date: date,
        symbols: list[str],
        historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Scan for new trading signals.

        Args:
            current_date: Current simulation date.
            symbols: List of symbols to scan.
            historical_data: Historical price data.
        """
        for symbol in symbols:
            # Skip if already in position
            if symbol in self._open_positions:
                continue

            # Skip if at max positions
            if len(self._open_positions) >= self.config.max_positions:
                break

            if symbol not in historical_data:
                continue

            df = historical_data[symbol]

            # Get data up to current date for signal generation
            if "date" in df.columns:
                mask = pd.to_datetime(df["date"]).dt.date <= current_date
                current_df = df[mask].copy()
            else:
                current_df = df[df.index <= pd.Timestamp(current_date)].copy()

            if len(current_df) < 20:  # Need minimum data
                continue

            # Run each strategy
            for strategy in self.strategies:
                try:
                    signal = strategy.scan(current_df)
                    if signal is not None:
                        self._signals_generated += 1

                        # Check minimum score
                        if signal.score < self.config.min_signal_score:
                            continue

                        # Execute the signal
                        self._execute_signal(signal, current_date, historical_data)
                        break  # One signal per symbol per day

                except Exception as e:
                    logger.warning(f"Error scanning {symbol} with {strategy.name}: {e}")

    def _execute_signal(
        self,
        signal: Signal,
        signal_date: date,
        historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Execute a trading signal.

        Args:
            signal: Signal to execute.
            signal_date: Date signal was generated.
            historical_data: Historical price data.
        """
        symbol = signal.symbol

        # Get next day's open for realistic fill
        if self.config.use_next_bar_open:
            next_date = signal_date + timedelta(days=1)
            # Find next available trading day
            for _ in range(5):  # Look up to 5 days ahead
                prices = self._get_price_on_date(symbol, next_date, historical_data)
                if prices is not None:
                    break
                next_date += timedelta(days=1)
            else:
                return  # No data available

            entry_price = prices["open"]
            entry_date = next_date
        else:
            entry_price = signal.entry_price
            entry_date = signal_date

        # Apply slippage
        if signal.direction == "LONG":
            entry_price *= 1 + self.config.slippage_percent
        else:
            entry_price *= 1 - self.config.slippage_percent

        # Calculate position size using current equity
        position_size = self.position_sizer.calculate_position_size(
            entry_price=entry_price,
            stop_price=signal.stop_loss,
            account_equity=self._equity,
        )

        if position_size.shares <= 0:
            return

        # Check if we can afford the position
        cost = position_size.dollar_amount + self.config.commission_per_trade
        if cost > self._equity:
            return

        # Create trade
        self._trade_counter += 1
        trade = BacktestTrade(
            trade_id=self._trade_counter,
            symbol=symbol,
            direction=signal.direction,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=position_size.shares,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            signal_score=signal.score,
            commission=self.config.commission_per_trade,
        )

        # Deduct from equity
        self._equity -= cost

        # Track position
        self._open_positions[symbol] = trade
        self._trades.append(trade)

    def _close_trade(
        self, trade: BacktestTrade, exit_price: float, exit_date: date, exit_reason: str
    ) -> None:
        """Close a trade and update equity.

        Args:
            trade: Trade to close.
            exit_price: Exit price.
            exit_date: Exit date.
            exit_reason: Reason for exit.
        """
        # Apply slippage to exit
        if trade.direction == "LONG":
            exit_price *= 1 - self.config.slippage_percent
        else:
            exit_price *= 1 + self.config.slippage_percent

        # Calculate P&L
        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.shares
        else:
            pnl = (trade.entry_price - exit_price) * trade.shares

        # Subtract exit commission
        pnl -= self.config.commission_per_trade
        trade.commission += self.config.commission_per_trade

        pnl_percent = (pnl / (trade.entry_price * trade.shares)) * 100

        # Update trade
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent

        # Update equity
        self._equity += trade.entry_price * trade.shares + pnl

    def _close_all_positions(
        self, final_date: date, historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Close all open positions at end of backtest.

        Args:
            final_date: Final date of backtest.
            historical_data: Historical price data.
        """
        for symbol, trade in list(self._open_positions.items()):
            prices = self._get_price_on_date(symbol, final_date, historical_data)
            if prices is not None:
                self._close_trade(trade, prices["close"], final_date, "end_of_backtest")
            del self._open_positions[symbol]

    def _calculate_drawdown(self, equity_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown for equity curve.

        Args:
            equity_df: DataFrame with equity column.

        Returns:
            DataFrame with added drawdown column.
        """
        equity = equity_df["equity"]
        peak = equity.expanding(min_periods=1).max()
        drawdown = (peak - equity) / peak
        equity_df["drawdown"] = drawdown
        return equity_df

    def _empty_result(
        self, initial_capital: float, start_date: date, end_date: date
    ) -> BacktestResult:
        """Create an empty result when no data available.

        Args:
            initial_capital: Starting capital.
            start_date: Start date.
            end_date: End date.

        Returns:
            Empty BacktestResult.
        """
        return BacktestResult(
            equity_curve=pd.DataFrame(columns=["date", "equity", "drawdown"]),
            trades=[],
            metrics=PerformanceMetrics(),
            signals_generated=0,
            trades_executed=0,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_equity=initial_capital,
        )

    def _generate_walk_forward_windows(
        self, start_date: date, end_date: date, train_months: int, test_months: int
    ) -> list[WalkForwardWindow]:
        """Generate walk-forward windows.

        Args:
            start_date: Start date.
            end_date: End date.
            train_months: Training period in months.
            test_months: Test period in months.

        Returns:
            List of WalkForwardWindow objects.
        """
        windows = []
        current_start = start_date

        while True:
            # Training period
            train_start = current_start
            train_end = self._add_months(train_start, train_months) - timedelta(days=1)

            # Test period
            test_start = train_end + timedelta(days=1)
            test_end = self._add_months(test_start, test_months) - timedelta(days=1)

            # Stop if test period goes beyond end date
            if test_start > end_date:
                break

            # Clip test end to end date
            test_end = min(test_end, end_date)

            windows.append(
                WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            # Move to next window
            current_start = test_start

        return windows

    @staticmethod
    def _add_months(d: date, months: int) -> date:
        """Add months to a date.

        Args:
            d: Starting date.
            months: Number of months to add.

        Returns:
            New date.
        """
        month = d.month + months
        year = d.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(d.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return date(year, month, day)
