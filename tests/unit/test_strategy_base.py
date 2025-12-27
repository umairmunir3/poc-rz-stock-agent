"""Unit tests for strategy base classes and interfaces."""

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from src.strategies.base import (
    Direction,
    ExitSignal,
    ExitType,
    Signal,
    Strategy,
    StrategyParameters,
    StrategyRegistry,
    StrategyResult,
    strategy_registry,
)


class TestDirectionEnum:
    """Tests for Direction enum."""

    def test_long_value(self) -> None:
        """Test LONG direction value."""
        assert Direction.LONG.value == "LONG"

    def test_short_value(self) -> None:
        """Test SHORT direction value."""
        assert Direction.SHORT.value == "SHORT"

    def test_string_inheritance(self) -> None:
        """Test that Direction inherits from str."""
        assert isinstance(Direction.LONG, str)
        assert isinstance(Direction.SHORT, str)


class TestExitTypeEnum:
    """Tests for ExitType enum."""

    def test_stop_loss_value(self) -> None:
        """Test STOP_LOSS exit type."""
        assert ExitType.STOP_LOSS.value == "STOP_LOSS"

    def test_take_profit_value(self) -> None:
        """Test TAKE_PROFIT exit type."""
        assert ExitType.TAKE_PROFIT.value == "TAKE_PROFIT"

    def test_trailing_stop_value(self) -> None:
        """Test TRAILING_STOP exit type."""
        assert ExitType.TRAILING_STOP.value == "TRAILING_STOP"

    def test_strategy_exit_value(self) -> None:
        """Test STRATEGY_EXIT exit type."""
        assert ExitType.STRATEGY_EXIT.value == "STRATEGY_EXIT"

    def test_time_exit_value(self) -> None:
        """Test TIME_EXIT exit type."""
        assert ExitType.TIME_EXIT.value == "TIME_EXIT"


class TestSignal:
    """Tests for Signal dataclass."""

    def test_valid_long_signal(self) -> None:
        """Test creating a valid LONG signal."""
        signal = Signal(
            symbol="AAPL",
            strategy="TestStrategy",
            direction="LONG",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            score=75,
            reasoning="Test signal",
        )
        assert signal.symbol == "AAPL"
        assert signal.direction == "LONG"
        assert signal.entry_price == 150.0
        assert signal.stop_loss == 145.0
        assert signal.take_profit == 160.0
        assert signal.score == 75

    def test_valid_short_signal(self) -> None:
        """Test creating a valid SHORT signal."""
        signal = Signal(
            symbol="MSFT",
            strategy="TestStrategy",
            direction="SHORT",
            entry_price=300.0,
            stop_loss=310.0,
            take_profit=280.0,
            score=65,
            reasoning="Test short signal",
        )
        assert signal.symbol == "MSFT"
        assert signal.direction == "SHORT"
        assert signal.stop_loss == 310.0
        assert signal.take_profit == 280.0

    def test_signal_with_metadata(self) -> None:
        """Test signal with custom metadata."""
        metadata = {"indicator": "RSI", "value": 30.5}
        signal = Signal(
            symbol="GOOGL",
            strategy="RSIStrategy",
            direction="LONG",
            entry_price=2800.0,
            stop_loss=2750.0,
            take_profit=2900.0,
            score=80,
            reasoning="RSI oversold",
            metadata=metadata,
        )
        assert signal.metadata == metadata
        assert signal.metadata["indicator"] == "RSI"

    def test_signal_timestamp_default(self) -> None:
        """Test that timestamp defaults to now."""
        before = datetime.now()
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="LONG",
            entry_price=150.0,
            stop_loss=145.0,
            take_profit=160.0,
            score=50,
            reasoning="Test",
        )
        after = datetime.now()
        assert before <= signal.timestamp <= after

    def test_invalid_score_too_high(self) -> None:
        """Test that score over 100 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0 and 100"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=101,
                reasoning="Test",
            )

    def test_invalid_score_negative(self) -> None:
        """Test that negative score raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0 and 100"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=-1,
                reasoning="Test",
            )

    def test_invalid_entry_price_zero(self) -> None:
        """Test that zero entry price raises ValueError."""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=0.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=50,
                reasoning="Test",
            )

    def test_invalid_entry_price_negative(self) -> None:
        """Test that negative entry price raises ValueError."""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=-50.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=50,
                reasoning="Test",
            )

    def test_invalid_stop_loss_zero(self) -> None:
        """Test that zero stop loss raises ValueError."""
        with pytest.raises(ValueError, match="Stop loss must be positive"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=0.0,
                take_profit=160.0,
                score=50,
                reasoning="Test",
            )

    def test_invalid_take_profit_zero(self) -> None:
        """Test that zero take profit raises ValueError."""
        with pytest.raises(ValueError, match="Take profit must be positive"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=0.0,
                score=50,
                reasoning="Test",
            )

    def test_long_stop_loss_above_entry(self) -> None:
        """Test that LONG stop loss above entry raises ValueError."""
        with pytest.raises(ValueError, match="Stop loss must be below entry price for LONG"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=155.0,
                take_profit=160.0,
                score=50,
                reasoning="Test",
            )

    def test_long_take_profit_below_entry(self) -> None:
        """Test that LONG take profit below entry raises ValueError."""
        with pytest.raises(ValueError, match="Take profit must be above entry price for LONG"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=140.0,
                score=50,
                reasoning="Test",
            )

    def test_short_stop_loss_below_entry(self) -> None:
        """Test that SHORT stop loss below entry raises ValueError."""
        with pytest.raises(ValueError, match="Stop loss must be above entry price for SHORT"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="SHORT",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=140.0,
                score=50,
                reasoning="Test",
            )

    def test_short_take_profit_above_entry(self) -> None:
        """Test that SHORT take profit above entry raises ValueError."""
        with pytest.raises(ValueError, match="Take profit must be below entry price for SHORT"):
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="SHORT",
                entry_price=150.0,
                stop_loss=155.0,
                take_profit=160.0,
                score=50,
                reasoning="Test",
            )

    def test_risk_reward_ratio_long(self) -> None:
        """Test risk reward ratio calculation for LONG."""
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            score=50,
            reasoning="Test",
        )
        # Risk = 100 - 95 = 5, Reward = 110 - 100 = 10
        assert signal.risk_reward_ratio == 2.0

    def test_risk_reward_ratio_short(self) -> None:
        """Test risk reward ratio calculation for SHORT."""
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="SHORT",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            score=50,
            reasoning="Test",
        )
        # Risk = 105 - 100 = 5, Reward = 100 - 90 = 10
        assert signal.risk_reward_ratio == 2.0

    def test_risk_percent_long(self) -> None:
        """Test risk percentage calculation for LONG."""
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            score=50,
            reasoning="Test",
        )
        assert signal.risk_percent == 5.0

    def test_risk_percent_short(self) -> None:
        """Test risk percentage calculation for SHORT."""
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="SHORT",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            score=50,
            reasoning="Test",
        )
        assert signal.risk_percent == 5.0

    def test_reward_percent_long(self) -> None:
        """Test reward percentage calculation for LONG."""
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            score=50,
            reasoning="Test",
        )
        assert signal.reward_percent == 10.0

    def test_reward_percent_short(self) -> None:
        """Test reward percentage calculation for SHORT."""
        signal = Signal(
            symbol="AAPL",
            strategy="Test",
            direction="SHORT",
            entry_price=100.0,
            stop_loss=105.0,
            take_profit=90.0,
            score=50,
            reasoning="Test",
        )
        assert signal.reward_percent == 10.0

    def test_to_dict(self) -> None:
        """Test converting signal to dictionary."""
        timestamp = datetime(2024, 1, 15, 10, 30, 0)
        signal = Signal(
            symbol="AAPL",
            strategy="TestStrategy",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            score=75,
            reasoning="Test reasoning",
            timestamp=timestamp,
            metadata={"key": "value"},
        )
        result = signal.to_dict()
        assert result["symbol"] == "AAPL"
        assert result["strategy"] == "TestStrategy"
        assert result["direction"] == "LONG"
        assert result["entry_price"] == 100.0
        assert result["stop_loss"] == 95.0
        assert result["take_profit"] == 110.0
        assert result["score"] == 75
        assert result["reasoning"] == "Test reasoning"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["metadata"] == {"key": "value"}
        assert result["risk_reward_ratio"] == 2.0
        assert result["risk_percent"] == 5.0
        assert result["reward_percent"] == 10.0


class TestExitSignal:
    """Tests for ExitSignal dataclass."""

    def test_valid_exit_signal(self) -> None:
        """Test creating a valid exit signal."""
        exit_signal = ExitSignal(
            trade_id=123,
            exit_type="STOP_LOSS",
            exit_price=145.0,
            reasoning="Stop loss triggered",
        )
        assert exit_signal.trade_id == 123
        assert exit_signal.exit_type == "STOP_LOSS"
        assert exit_signal.exit_price == 145.0

    def test_exit_signal_timestamp_default(self) -> None:
        """Test that timestamp defaults to now."""
        before = datetime.now()
        exit_signal = ExitSignal(
            trade_id=1,
            exit_type="TAKE_PROFIT",
            exit_price=100.0,
            reasoning="Test",
        )
        after = datetime.now()
        assert before <= exit_signal.timestamp <= after

    def test_invalid_exit_price_zero(self) -> None:
        """Test that zero exit price raises ValueError."""
        with pytest.raises(ValueError, match="Exit price must be positive"):
            ExitSignal(
                trade_id=1,
                exit_type="STOP_LOSS",
                exit_price=0.0,
                reasoning="Test",
            )

    def test_invalid_exit_price_negative(self) -> None:
        """Test that negative exit price raises ValueError."""
        with pytest.raises(ValueError, match="Exit price must be positive"):
            ExitSignal(
                trade_id=1,
                exit_type="STOP_LOSS",
                exit_price=-10.0,
                reasoning="Test",
            )

    def test_to_dict(self) -> None:
        """Test converting exit signal to dictionary."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        exit_signal = ExitSignal(
            trade_id=456,
            exit_type="TRAILING_STOP",
            exit_price=155.0,
            reasoning="Trailing stop hit",
            timestamp=timestamp,
        )
        result = exit_signal.to_dict()
        assert result["trade_id"] == 456
        assert result["exit_type"] == "TRAILING_STOP"
        assert result["exit_price"] == 155.0
        assert result["reasoning"] == "Trailing stop hit"
        assert result["timestamp"] == timestamp.isoformat()


class TestStrategyResult:
    """Tests for StrategyResult dataclass."""

    def test_empty_result(self) -> None:
        """Test creating an empty strategy result."""
        result = StrategyResult(strategy_name="TestStrategy")
        assert result.strategy_name == "TestStrategy"
        assert result.signals == []
        assert result.scanned_count == 0
        assert result.signal_count == 0

    def test_result_with_signals(self) -> None:
        """Test creating a result with signals."""
        signals = [
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=80,
                reasoning="Test",
            ),
            Signal(
                symbol="MSFT",
                strategy="Test",
                direction="SHORT",
                entry_price=300.0,
                stop_loss=310.0,
                take_profit=280.0,
                score=70,
                reasoning="Test",
            ),
        ]
        result = StrategyResult(
            strategy_name="TestStrategy",
            signals=signals,
            scanned_count=100,
        )
        assert result.signal_count == 2
        assert len(result.signals) == 2

    def test_hit_rate(self) -> None:
        """Test hit rate calculation."""
        signals = [
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=80,
                reasoning="Test",
            ),
        ]
        result = StrategyResult(
            strategy_name="TestStrategy",
            signals=signals,
            scanned_count=10,
        )
        assert result.hit_rate == 10.0

    def test_hit_rate_zero_scanned(self) -> None:
        """Test hit rate when nothing scanned."""
        result = StrategyResult(strategy_name="TestStrategy")
        assert result.hit_rate == 0.0

    def test_avg_score(self) -> None:
        """Test average score calculation."""
        signals = [
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=150.0,
                stop_loss=145.0,
                take_profit=160.0,
                score=80,
                reasoning="Test",
            ),
            Signal(
                symbol="MSFT",
                strategy="Test",
                direction="LONG",
                entry_price=300.0,
                stop_loss=295.0,
                take_profit=310.0,
                score=60,
                reasoning="Test",
            ),
        ]
        result = StrategyResult(strategy_name="TestStrategy", signals=signals)
        assert result.avg_score == 70.0

    def test_avg_score_no_signals(self) -> None:
        """Test average score with no signals."""
        result = StrategyResult(strategy_name="TestStrategy")
        assert result.avg_score == 0.0

    def test_get_top_signals(self) -> None:
        """Test getting top signals by score."""
        signals = [
            Signal(
                symbol="A",
                strategy="Test",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=50,
                reasoning="Test",
            ),
            Signal(
                symbol="B",
                strategy="Test",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=90,
                reasoning="Test",
            ),
            Signal(
                symbol="C",
                strategy="Test",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=70,
                reasoning="Test",
            ),
        ]
        result = StrategyResult(strategy_name="TestStrategy", signals=signals)
        top = result.get_top_signals(2)
        assert len(top) == 2
        assert top[0].symbol == "B"
        assert top[0].score == 90
        assert top[1].symbol == "C"
        assert top[1].score == 70

    def test_filter_by_direction_long(self) -> None:
        """Test filtering signals by LONG direction."""
        signals = [
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=80,
                reasoning="Test",
            ),
            Signal(
                symbol="MSFT",
                strategy="Test",
                direction="SHORT",
                entry_price=100.0,
                stop_loss=105.0,
                take_profit=90.0,
                score=70,
                reasoning="Test",
            ),
        ]
        result = StrategyResult(strategy_name="TestStrategy", signals=signals)
        long_signals = result.filter_by_direction("LONG")
        assert len(long_signals) == 1
        assert long_signals[0].symbol == "AAPL"

    def test_filter_by_direction_short(self) -> None:
        """Test filtering signals by SHORT direction."""
        signals = [
            Signal(
                symbol="AAPL",
                strategy="Test",
                direction="LONG",
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
                score=80,
                reasoning="Test",
            ),
            Signal(
                symbol="MSFT",
                strategy="Test",
                direction="SHORT",
                entry_price=100.0,
                stop_loss=105.0,
                take_profit=90.0,
                score=70,
                reasoning="Test",
            ),
        ]
        result = StrategyResult(strategy_name="TestStrategy", signals=signals)
        short_signals = result.filter_by_direction("SHORT")
        assert len(short_signals) == 1
        assert short_signals[0].symbol == "MSFT"


class TestStrategyParameters:
    """Tests for StrategyParameters dataclass."""

    def test_valid_numeric_parameter(self) -> None:
        """Test creating a valid numeric parameter."""
        param = StrategyParameters(
            name="rsi_period",
            value=14,
            min_value=5,
            max_value=50,
            description="RSI calculation period",
        )
        assert param.name == "rsi_period"
        assert param.value == 14
        assert param.validate()

    def test_valid_float_parameter(self) -> None:
        """Test creating a valid float parameter."""
        param = StrategyParameters(
            name="risk_percent",
            value=0.02,
            min_value=0.01,
            max_value=0.10,
        )
        assert param.validate()

    def test_valid_bool_parameter(self) -> None:
        """Test creating a valid boolean parameter."""
        param = StrategyParameters(
            name="use_trailing_stop",
            value=True,
        )
        assert param.validate()

    def test_valid_string_parameter(self) -> None:
        """Test creating a valid string parameter."""
        param = StrategyParameters(
            name="timeframe",
            value="1D",
        )
        assert param.validate()

    def test_invalid_below_minimum(self) -> None:
        """Test parameter below minimum fails validation."""
        param = StrategyParameters(
            name="period",
            value=3,
            min_value=5,
            max_value=50,
        )
        assert not param.validate()

    def test_invalid_above_maximum(self) -> None:
        """Test parameter above maximum fails validation."""
        param = StrategyParameters(
            name="period",
            value=100,
            min_value=5,
            max_value=50,
        )
        assert not param.validate()

    def test_valid_at_minimum(self) -> None:
        """Test parameter at minimum is valid."""
        param = StrategyParameters(
            name="period",
            value=5,
            min_value=5,
            max_value=50,
        )
        assert param.validate()

    def test_valid_at_maximum(self) -> None:
        """Test parameter at maximum is valid."""
        param = StrategyParameters(
            name="period",
            value=50,
            min_value=5,
            max_value=50,
        )
        assert param.validate()


class ConcreteStrategy(Strategy):
    """Concrete implementation of Strategy for testing."""

    name = "ConcreteTestStrategy"
    description = "A concrete strategy for testing"

    def __init__(self, param1: int = 10, param2: float = 0.5) -> None:
        """Initialize with test parameters."""
        self.param1 = param1
        self.param2 = param2

    def scan(self, df: pd.DataFrame) -> Signal | None:
        """Return None for testing."""
        return None

    def check_exit(self, df: pd.DataFrame, trade: Any) -> ExitSignal | None:
        """Return None for testing."""
        return None

    def get_parameters(self) -> dict[str, Any]:
        """Return current parameters."""
        return {"param1": self.param1, "param2": self.param2}

    def set_parameters(self, params: dict[str, Any]) -> None:
        """Set parameters from dict."""
        if "param1" in params:
            self.param1 = params["param1"]
        if "param2" in params:
            self.param2 = params["param2"]


class TestStrategy:
    """Tests for Strategy abstract base class."""

    @pytest.fixture
    def strategy(self) -> ConcreteStrategy:
        """Create a concrete strategy for testing."""
        return ConcreteStrategy()

    @pytest.fixture
    def valid_df(self) -> pd.DataFrame:
        """Create a valid OHLCV DataFrame."""
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000, 1100, 1200],
            }
        )

    def test_strategy_name(self, strategy: ConcreteStrategy) -> None:
        """Test strategy name attribute."""
        assert strategy.name == "ConcreteTestStrategy"

    def test_strategy_description(self, strategy: ConcreteStrategy) -> None:
        """Test strategy description attribute."""
        assert strategy.description == "A concrete strategy for testing"

    def test_validate_dataframe_valid(
        self, strategy: ConcreteStrategy, valid_df: pd.DataFrame
    ) -> None:
        """Test DataFrame validation with valid data."""
        assert strategy.validate_dataframe(valid_df)

    def test_validate_dataframe_missing_column(self, strategy: ConcreteStrategy) -> None:
        """Test DataFrame validation with missing column."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                # Missing volume
            }
        )
        assert not strategy.validate_dataframe(df)

    def test_validate_dataframe_case_insensitive(self, strategy: ConcreteStrategy) -> None:
        """Test DataFrame validation is case-insensitive."""
        df = pd.DataFrame(
            {
                "Open": [100.0],
                "HIGH": [102.0],
                "Low": [99.0],
                "CLOSE": [101.0],
                "Volume": [1000],
            }
        )
        assert strategy.validate_dataframe(df)

    def test_calculate_stop_loss_long(self, strategy: ConcreteStrategy) -> None:
        """Test stop loss calculation for LONG position."""
        stop = strategy.calculate_stop_loss(
            entry_price=100.0,
            atr=2.0,
            direction="LONG",
            multiplier=2.0,
        )
        assert stop == 96.0

    def test_calculate_stop_loss_short(self, strategy: ConcreteStrategy) -> None:
        """Test stop loss calculation for SHORT position."""
        stop = strategy.calculate_stop_loss(
            entry_price=100.0,
            atr=2.0,
            direction="SHORT",
            multiplier=2.0,
        )
        assert stop == 104.0

    def test_calculate_stop_loss_custom_multiplier(self, strategy: ConcreteStrategy) -> None:
        """Test stop loss with custom ATR multiplier."""
        stop = strategy.calculate_stop_loss(
            entry_price=100.0,
            atr=2.0,
            direction="LONG",
            multiplier=3.0,
        )
        assert stop == 94.0

    def test_calculate_take_profit_long(self, strategy: ConcreteStrategy) -> None:
        """Test take profit calculation for LONG position."""
        take_profit = strategy.calculate_take_profit(
            entry_price=100.0,
            stop_loss=95.0,
            direction="LONG",
            risk_reward=2.0,
        )
        assert take_profit == 110.0

    def test_calculate_take_profit_short(self, strategy: ConcreteStrategy) -> None:
        """Test take profit calculation for SHORT position."""
        take_profit = strategy.calculate_take_profit(
            entry_price=100.0,
            stop_loss=105.0,
            direction="SHORT",
            risk_reward=2.0,
        )
        assert take_profit == 90.0

    def test_calculate_take_profit_custom_rr(self, strategy: ConcreteStrategy) -> None:
        """Test take profit with custom risk-reward ratio."""
        take_profit = strategy.calculate_take_profit(
            entry_price=100.0,
            stop_loss=95.0,
            direction="LONG",
            risk_reward=3.0,
        )
        assert take_profit == 115.0

    def test_score_signal_all_aligned(self, strategy: ConcreteStrategy) -> None:
        """Test signal scoring with all factors aligned."""
        score = strategy.score_signal(
            trend_aligned=True,
            momentum_aligned=True,
            volume_confirmed=True,
            at_support_resistance=True,
            risk_reward_good=True,
        )
        assert score == 100

    def test_score_signal_none_aligned(self, strategy: ConcreteStrategy) -> None:
        """Test signal scoring with no factors aligned."""
        score = strategy.score_signal(
            trend_aligned=False,
            momentum_aligned=False,
            volume_confirmed=False,
            at_support_resistance=False,
            risk_reward_good=False,
        )
        assert score == 0

    def test_score_signal_partial(self, strategy: ConcreteStrategy) -> None:
        """Test signal scoring with partial alignment."""
        score = strategy.score_signal(
            trend_aligned=True,  # 30
            momentum_aligned=True,  # 25
            volume_confirmed=False,
            at_support_resistance=False,
            risk_reward_good=True,  # 10
        )
        assert score == 65

    def test_get_parameters(self, strategy: ConcreteStrategy) -> None:
        """Test getting strategy parameters."""
        params = strategy.get_parameters()
        assert params["param1"] == 10
        assert params["param2"] == 0.5

    def test_set_parameters(self, strategy: ConcreteStrategy) -> None:
        """Test setting strategy parameters."""
        strategy.set_parameters({"param1": 20, "param2": 0.8})
        assert strategy.param1 == 20
        assert strategy.param2 == 0.8


class TestStrategyRegistry:
    """Tests for StrategyRegistry class."""

    @pytest.fixture
    def registry(self) -> StrategyRegistry:
        """Create a fresh registry for testing."""
        return StrategyRegistry()

    def test_register_strategy(self, registry: StrategyRegistry) -> None:
        """Test registering a strategy."""
        registry.register(ConcreteStrategy)
        assert "ConcreteTestStrategy" in registry.list_strategies()

    def test_get_registered_strategy(self, registry: StrategyRegistry) -> None:
        """Test getting a registered strategy."""
        registry.register(ConcreteStrategy)
        strategy_class = registry.get("ConcreteTestStrategy")
        assert strategy_class is ConcreteStrategy

    def test_get_unregistered_strategy(self, registry: StrategyRegistry) -> None:
        """Test getting an unregistered strategy returns None."""
        result = registry.get("NonExistentStrategy")
        assert result is None

    def test_list_strategies_empty(self, registry: StrategyRegistry) -> None:
        """Test listing strategies when empty."""
        assert registry.list_strategies() == []

    def test_list_strategies_multiple(self, registry: StrategyRegistry) -> None:
        """Test listing multiple strategies."""

        class AnotherStrategy(ConcreteStrategy):
            name = "AnotherTestStrategy"

        registry.register(ConcreteStrategy)
        registry.register(AnotherStrategy)
        strategies = registry.list_strategies()
        assert len(strategies) == 2
        assert "ConcreteTestStrategy" in strategies
        assert "AnotherTestStrategy" in strategies

    def test_create_instance(self, registry: StrategyRegistry) -> None:
        """Test creating a strategy instance."""
        registry.register(ConcreteStrategy)
        instance = registry.create_instance("ConcreteTestStrategy")
        assert isinstance(instance, ConcreteStrategy)
        assert instance.param1 == 10

    def test_create_instance_with_kwargs(self, registry: StrategyRegistry) -> None:
        """Test creating a strategy instance with custom kwargs."""
        registry.register(ConcreteStrategy)
        instance = registry.create_instance("ConcreteTestStrategy", param1=25)
        assert instance is not None
        assert instance.param1 == 25

    def test_create_instance_unregistered(self, registry: StrategyRegistry) -> None:
        """Test creating instance of unregistered strategy returns None."""
        result = registry.create_instance("NonExistentStrategy")
        assert result is None


class TestGlobalRegistry:
    """Tests for the global strategy registry."""

    def test_global_registry_exists(self) -> None:
        """Test that global registry is available."""
        assert strategy_registry is not None
        assert isinstance(strategy_registry, StrategyRegistry)
