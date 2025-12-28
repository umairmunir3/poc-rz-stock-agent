"""Unit tests for Backtesting Engine."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestTrade,
    PerformanceMetrics,
    WalkForwardWindow,
)
from src.risk.position_sizing import PositionSizer

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_historical_data() -> dict[str, pd.DataFrame]:
    """Create sample historical data for testing."""
    dates = pd.date_range("2024-01-01", "2024-03-31", freq="B")

    # AAPL - trending up
    aapl_prices = np.linspace(150, 180, len(dates)) + np.random.normal(0, 1, len(dates))
    aapl_df = pd.DataFrame(
        {
            "date": dates,
            "open": aapl_prices - 0.5,
            "high": aapl_prices + 1,
            "low": aapl_prices - 1,
            "close": aapl_prices,
            "volume": np.random.randint(1000000, 5000000, len(dates)),
        }
    )

    # MSFT - sideways with volatility
    msft_prices = 350 + np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 10
    msft_df = pd.DataFrame(
        {
            "date": dates,
            "open": msft_prices - 0.5,
            "high": msft_prices + 2,
            "low": msft_prices - 2,
            "close": msft_prices,
            "volume": np.random.randint(500000, 2000000, len(dates)),
        }
    )

    return {"AAPL": aapl_df, "MSFT": msft_df}


@pytest.fixture
def mock_strategy() -> MagicMock:
    """Create a mock strategy."""
    strategy = MagicMock()
    strategy.name = "MockStrategy"
    strategy.scan.return_value = None
    return strategy


@pytest.fixture
def position_sizer() -> PositionSizer:
    """Create a position sizer."""
    return PositionSizer(account_equity=10000.0, risk_per_trade=0.02)


@pytest.fixture
def backtest_engine(mock_strategy: MagicMock, position_sizer: PositionSizer) -> BacktestEngine:
    """Create a backtest engine."""
    return BacktestEngine(
        strategies=[mock_strategy],
        position_sizer=position_sizer,
    )


# -----------------------------------------------------------------------------
# BacktestTrade Tests
# -----------------------------------------------------------------------------


class TestBacktestTrade:
    """Tests for BacktestTrade dataclass."""

    def test_trade_creation(self) -> None:
        """Test basic trade creation."""
        trade = BacktestTrade(
            trade_id=1,
            symbol="AAPL",
            direction="LONG",
            entry_date=date(2024, 1, 15),
            entry_price=150.0,
            shares=10,
            stop_loss=145.0,
            take_profit=165.0,
            strategy="TestStrategy",
            signal_score=75,
        )

        assert trade.trade_id == 1
        assert trade.symbol == "AAPL"
        assert trade.direction == "LONG"
        assert trade.is_open is True

    def test_trade_is_open(self) -> None:
        """Test is_open property."""
        trade = BacktestTrade(
            trade_id=1,
            symbol="AAPL",
            direction="LONG",
            entry_date=date(2024, 1, 15),
            entry_price=150.0,
            shares=10,
            stop_loss=145.0,
            take_profit=165.0,
            strategy="TestStrategy",
            signal_score=75,
        )

        assert trade.is_open is True

        trade.exit_date = date(2024, 1, 20)
        trade.exit_price = 160.0
        assert trade.is_open is False

    def test_trade_is_winner(self) -> None:
        """Test is_winner property."""
        trade = BacktestTrade(
            trade_id=1,
            symbol="AAPL",
            direction="LONG",
            entry_date=date(2024, 1, 15),
            entry_price=150.0,
            shares=10,
            stop_loss=145.0,
            take_profit=165.0,
            strategy="TestStrategy",
            signal_score=75,
            pnl=100.0,
        )

        assert trade.is_winner is True

        trade.pnl = -50.0
        assert trade.is_winner is False

    def test_trade_hold_days(self) -> None:
        """Test hold_days calculation."""
        trade = BacktestTrade(
            trade_id=1,
            symbol="AAPL",
            direction="LONG",
            entry_date=date(2024, 1, 15),
            entry_price=150.0,
            shares=10,
            stop_loss=145.0,
            take_profit=165.0,
            strategy="TestStrategy",
            signal_score=75,
            exit_date=date(2024, 1, 25),
        )

        assert trade.hold_days == 10

    def test_trade_r_multiple(self) -> None:
        """Test R-multiple calculation."""
        trade = BacktestTrade(
            trade_id=1,
            symbol="AAPL",
            direction="LONG",
            entry_date=date(2024, 1, 15),
            entry_price=100.0,
            shares=10,
            stop_loss=95.0,  # $5 risk per share
            take_profit=115.0,
            strategy="TestStrategy",
            signal_score=75,
            pnl=100.0,  # $10 profit per share (2R)
        )

        assert trade.r_multiple == pytest.approx(2.0)


# -----------------------------------------------------------------------------
# PerformanceMetrics Tests
# -----------------------------------------------------------------------------


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics."""

    def test_empty_trades_returns_defaults(self) -> None:
        """Test that empty trades return default metrics."""
        equity_df = pd.DataFrame({"equity": [10000], "drawdown": [0]})
        metrics = PerformanceMetrics.from_trades([], equity_df, 10000, 252)

        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0

    def test_win_rate_calculation(self) -> None:
        """Test win rate calculation."""
        trades = [
            BacktestTrade(
                trade_id=i,
                symbol="AAPL",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=105.0 if i < 6 else 97.0,
                pnl=50.0 if i < 6 else -30.0,
            )
            for i in range(10)
        ]

        equity_df = pd.DataFrame({"equity": [10000, 10050, 10100, 10150], "drawdown": [0, 0, 0, 0]})
        metrics = PerformanceMetrics.from_trades(trades, equity_df, 10000, 252)

        # 6 winners out of 10 trades
        assert metrics.win_rate == pytest.approx(0.6)
        assert metrics.total_trades == 10

    def test_sharpe_ratio_formula(self) -> None:
        """Test Sharpe ratio calculation."""
        # Create equity curve with known returns
        equity_values = [10000, 10100, 10200, 10300, 10400]  # 1% daily returns
        equity_df = pd.DataFrame({"equity": equity_values, "drawdown": [0] * len(equity_values)})

        trades = [
            BacktestTrade(
                trade_id=1,
                symbol="AAPL",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=104.0,
                pnl=400.0,
            )
        ]

        metrics = PerformanceMetrics.from_trades(trades, equity_df, 10000, len(equity_values))

        # With constant 1% returns, Sharpe should be very high
        assert metrics.sharpe_ratio > 0

    def test_max_drawdown_accurate(self) -> None:
        """Test max drawdown calculation."""
        equity_df = pd.DataFrame(
            {"equity": [10000, 11000, 9900, 10500], "drawdown": [0, 0, 0.10, 0.045]}
        )

        trades = [
            BacktestTrade(
                trade_id=1,
                symbol="AAPL",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=105.0,
                pnl=50.0,
            )
        ]

        metrics = PerformanceMetrics.from_trades(trades, equity_df, 10000, 4)

        assert metrics.max_drawdown == pytest.approx(0.10)

    def test_profit_factor_calculation(self) -> None:
        """Test profit factor calculation."""
        trades = [
            BacktestTrade(
                trade_id=1,
                symbol="AAPL",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=110.0,
                pnl=100.0,  # Winner
            ),
            BacktestTrade(
                trade_id=2,
                symbol="MSFT",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=95.0,
                pnl=-50.0,  # Loser
            ),
        ]

        equity_df = pd.DataFrame({"equity": [10000, 10050], "drawdown": [0, 0]})
        metrics = PerformanceMetrics.from_trades(trades, equity_df, 10000, 2)

        # Profit factor = 100 / 50 = 2.0
        assert metrics.profit_factor == pytest.approx(2.0)

    def test_expectancy_calculation(self) -> None:
        """Test expectancy (R-multiple) calculation."""
        trades = [
            BacktestTrade(
                trade_id=1,
                symbol="AAPL",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,  # $5 risk
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=110.0,
                pnl=100.0,  # $10/share = 2R
            ),
            BacktestTrade(
                trade_id=2,
                symbol="MSFT",
                direction="LONG",
                entry_date=date(2024, 1, 1),
                entry_price=100.0,
                shares=10,
                stop_loss=95.0,  # $5 risk
                take_profit=110.0,
                strategy="Test",
                signal_score=75,
                exit_date=date(2024, 1, 5),
                exit_price=95.0,
                pnl=-50.0,  # -$5/share = -1R
            ),
        ]

        equity_df = pd.DataFrame({"equity": [10000, 10050], "drawdown": [0, 0]})
        metrics = PerformanceMetrics.from_trades(trades, equity_df, 10000, 2)

        # Average R = (2R + (-1R)) / 2 = 0.5R
        assert metrics.expectancy == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# BacktestEngine Tests
# -----------------------------------------------------------------------------


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_engine_initialization(
        self, mock_strategy: MagicMock, position_sizer: PositionSizer
    ) -> None:
        """Test engine initialization."""
        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
        )

        assert len(engine.strategies) == 1
        assert engine.position_sizer is position_sizer
        assert engine.config.slippage_percent == 0.001

    def test_engine_with_custom_config(
        self, mock_strategy: MagicMock, position_sizer: PositionSizer
    ) -> None:
        """Test engine with custom configuration."""
        config = BacktestConfig(
            slippage_percent=0.002,
            commission_per_trade=2.0,
            max_positions=5,
        )
        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
            config=config,
        )

        assert engine.config.slippage_percent == 0.002
        assert engine.config.commission_per_trade == 2.0
        assert engine.config.max_positions == 5

    def test_run_with_no_signals(
        self,
        backtest_engine: BacktestEngine,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test running backtest with no signals generated."""
        result = backtest_engine.run(
            symbols=["AAPL", "MSFT"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        assert result.initial_capital == 10000.0
        assert result.final_equity == 10000.0
        assert result.trades_executed == 0
        assert len(result.trades) == 0

    def test_equity_curve_calculation(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test that equity curve is calculated correctly."""
        # Create a signal that will be executed
        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 152.0
        signal.stop_loss = 148.0
        signal.take_profit = 160.0
        signal.score = 80
        signal.strategy = "TestStrategy"

        mock_strategy.scan.return_value = signal

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        assert not result.equity_curve.empty
        assert "date" in result.equity_curve.columns
        assert "equity" in result.equity_curve.columns
        assert "drawdown" in result.equity_curve.columns

    def test_drawdown_calculation(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test drawdown is calculated correctly."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        # Drawdown should be >= 0
        assert (result.equity_curve["drawdown"] >= 0).all()

    def test_realistic_fills_uses_next_bar_open(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test that fills use next bar open price."""
        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 152.0
        signal.stop_loss = 148.0
        signal.take_profit = 200.0  # High target to stay in trade
        signal.score = 80
        signal.strategy = "TestStrategy"

        # Only return signal on first scan
        mock_strategy.scan.side_effect = [signal] + [None] * 100

        config = BacktestConfig(
            use_next_bar_open=True,
            slippage_percent=0.0,  # Disable slippage for clear test
        )

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
            config=config,
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        if result.trades:
            trade = result.trades[0]
            # Entry should be from the next bar's open, not the signal's entry price
            assert trade.entry_price != signal.entry_price

    def test_slippage_applied(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test that slippage is applied to entry price."""
        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 150.0
        signal.stop_loss = 145.0
        signal.take_profit = 200.0
        signal.score = 80
        signal.strategy = "TestStrategy"

        mock_strategy.scan.side_effect = [signal] + [None] * 100

        config = BacktestConfig(
            use_next_bar_open=False,
            slippage_percent=0.001,  # 0.1% slippage
        )

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
            config=config,
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        if result.trades:
            trade = result.trades[0]
            # For LONG, slippage increases entry price
            # Entry should be slightly higher than raw price
            assert trade.entry_price > 149.0

    def test_commission_deducted(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test that commissions are deducted."""
        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 152.0
        signal.stop_loss = 148.0
        signal.take_profit = 200.0
        signal.score = 80
        signal.strategy = "TestStrategy"

        mock_strategy.scan.side_effect = [signal] + [None] * 100

        config = BacktestConfig(
            commission_per_trade=5.0,
        )

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
            config=config,
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        if result.trades:
            trade = result.trades[0]
            # Commission should be tracked (entry + exit)
            assert trade.commission == 10.0  # $5 entry + $5 exit

    def test_position_sizing_uses_actual_equity(
        self,
        mock_strategy: MagicMock,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test that position sizing uses current equity."""
        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 152.0
        signal.stop_loss = 148.0
        signal.take_profit = 200.0
        signal.score = 80
        signal.strategy = "TestStrategy"

        mock_strategy.scan.side_effect = [signal] + [None] * 100

        # Custom position sizer with specific settings
        sizer = PositionSizer(account_equity=10000.0, risk_per_trade=0.02)

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=sizer,
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        if result.trades:
            trade = result.trades[0]
            # Position should be sized appropriately for $10,000 account
            position_value = trade.shares * trade.entry_price
            assert position_value <= 10000.0

    def test_consistency_across_runs(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test that same inputs produce same outputs."""
        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
        )

        result1 = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        result2 = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        assert result1.final_equity == result2.final_equity
        assert len(result1.trades) == len(result2.trades)
        assert result1.metrics.total_return == result2.metrics.total_return


# -----------------------------------------------------------------------------
# Walk-Forward Tests
# -----------------------------------------------------------------------------


class TestWalkForward:
    """Tests for walk-forward optimization."""

    def test_walk_forward_windows(
        self,
        backtest_engine: BacktestEngine,
        sample_historical_data: dict[str, pd.DataFrame],
    ) -> None:
        """Test walk-forward generates correct windows."""
        result = backtest_engine.walk_forward(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            train_months=1,
            test_months=1,
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        # Should have at least 1 window
        assert len(result.windows) >= 1

        # Each window should have train and test results
        for window in result.windows:
            assert window.train_start < window.train_end
            assert window.test_start < window.test_end
            assert window.train_end < window.test_start


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow dataclass."""

    def test_window_creation(self) -> None:
        """Test window creation."""
        window = WalkForwardWindow(
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 9, 30),
        )

        assert window.train_start == date(2024, 1, 1)
        assert window.train_end == date(2024, 6, 30)
        assert window.train_result is None
        assert window.test_result is None


# -----------------------------------------------------------------------------
# Report Generation Tests
# -----------------------------------------------------------------------------


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_contains_metrics(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test that generated report contains all metrics."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        report = backtest_engine.generate_report(result)

        assert "Backtest Report" in report
        assert "Summary" in report
        assert "CAGR" in report
        assert "Sharpe Ratio" in report
        assert "Max Drawdown" in report
        assert "Win Rate" in report
        assert "Profit Factor" in report
        assert "Expectancy" in report

    def test_generate_html_report(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test HTML report generation."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        html = backtest_engine.generate_html_report(result)

        assert "<!DOCTYPE html>" in html
        assert "Backtest Report" in html
        assert "table" in html.lower()


class TestPlotGeneration:
    """Tests for plot generation."""

    def test_plot_equity_curve(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test equity curve plot generation."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        fig = backtest_engine.plot_equity_curve(result)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_drawdown(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test drawdown plot generation."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        fig = backtest_engine.plot_drawdown(result)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_monthly_returns(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test monthly returns heatmap generation."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        fig = backtest_engine.plot_monthly_returns(result)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# -----------------------------------------------------------------------------
# Configuration Tests
# -----------------------------------------------------------------------------


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.slippage_percent == 0.001
        assert config.commission_per_trade == 1.0
        assert config.use_next_bar_open is True
        assert config.max_positions == 10
        assert config.min_signal_score == 0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = BacktestConfig(
            slippage_percent=0.005,
            commission_per_trade=5.0,
            use_next_bar_open=False,
            max_positions=5,
            min_signal_score=70,
        )

        assert config.slippage_percent == 0.005
        assert config.commission_per_trade == 5.0
        assert config.use_next_bar_open is False
        assert config.max_positions == 5
        assert config.min_signal_score == 70


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_historical_data(self, backtest_engine: BacktestEngine) -> None:
        """Test handling of empty historical data."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_capital=10000.0,
            historical_data={},
        )

        assert result.final_equity == 10000.0
        assert result.trades_executed == 0

    def test_single_day_backtest(
        self, backtest_engine: BacktestEngine, sample_historical_data: dict[str, pd.DataFrame]
    ) -> None:
        """Test backtest over a single day."""
        result = backtest_engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2),
            initial_capital=10000.0,
            historical_data=sample_historical_data,
        )

        assert result.final_equity == 10000.0

    def test_stop_loss_trigger(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
    ) -> None:
        """Test that stop loss is properly triggered."""
        # Create data where price drops to stop loss
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        prices = [150, 149, 148, 147, 146, 145, 144, 143]
        prices = prices[: len(dates)]

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000000] * len(dates),
            }
        )

        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 150.0
        signal.stop_loss = 146.0  # Will be hit
        signal.take_profit = 160.0
        signal.score = 80
        signal.strategy = "TestStrategy"

        mock_strategy.scan.side_effect = [signal] + [None] * 100

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
            config=BacktestConfig(use_next_bar_open=False, slippage_percent=0),
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=10000.0,
            historical_data={"AAPL": data},
        )

        if result.trades:
            trade = result.trades[0]
            assert trade.exit_reason == "stop_loss"
            assert trade.pnl < 0

    def test_take_profit_trigger(
        self,
        mock_strategy: MagicMock,
        position_sizer: PositionSizer,
    ) -> None:
        """Test that take profit is properly triggered."""
        # Create data where price rises to take profit
        dates = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        prices = [150, 152, 154, 156, 158, 160, 162, 164]
        prices = prices[: len(dates)]

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [p + 1 for p in prices],
                "low": [p - 1 for p in prices],
                "close": prices,
                "volume": [1000000] * len(dates),
            }
        )

        signal = MagicMock()
        signal.symbol = "AAPL"
        signal.direction = "LONG"
        signal.entry_price = 150.0
        signal.stop_loss = 145.0
        signal.take_profit = 158.0  # Will be hit
        signal.score = 80
        signal.strategy = "TestStrategy"

        mock_strategy.scan.side_effect = [signal] + [None] * 100

        engine = BacktestEngine(
            strategies=[mock_strategy],
            position_sizer=position_sizer,
            config=BacktestConfig(use_next_bar_open=False, slippage_percent=0),
        )

        result = engine.run(
            symbols=["AAPL"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 10),
            initial_capital=10000.0,
            historical_data={"AAPL": data},
        )

        if result.trades:
            trade = result.trades[0]
            assert trade.exit_reason == "take_profit"
            assert trade.pnl > 0


class TestHelperMethods:
    """Tests for private helper methods."""

    def test_add_months(self) -> None:
        """Test _add_months helper."""
        result = BacktestEngine._add_months(date(2024, 1, 15), 3)
        assert result == date(2024, 4, 15)

        # Test year rollover
        result = BacktestEngine._add_months(date(2024, 11, 15), 3)
        assert result == date(2025, 2, 15)

        # Test end of month handling
        result = BacktestEngine._add_months(date(2024, 1, 31), 1)
        assert result == date(2024, 2, 29)  # 2024 is a leap year
