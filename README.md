# Swing Trader

A comprehensive swing trading system with technical analysis, risk management, options analysis, and automated execution capabilities.

## Overview

Swing Trader is a Python-based trading system designed for semi-automated swing trading strategies. It integrates with Interactive Brokers for execution, uses Alpha Vantage for market data, and provides real-time notifications via Telegram.

### Key Features

- **Technical Analysis**: Built-in indicators using TA-Lib and pandas-ta
- **Risk Management**: Position sizing, max drawdown controls, and portfolio risk limits
- **Options Analysis**: Options chain analysis and Greeks calculations
- **Signal Generation**: Customizable trading signals and strategy framework
- **Backtesting**: Historical strategy testing with performance metrics
- **Automated Execution**: Integration with Interactive Brokers via ib_insync
- **Real-time Notifications**: Telegram alerts for trades and signals
- **Data Pipeline**: AWS S3 storage and PostgreSQL for historical data

## Architecture

```
swing-trader/
├── src/
│   ├── data/           # Data fetching, storage, and preprocessing
│   ├── indicators/     # Technical indicators (RSI, MACD, Bollinger, etc.)
│   ├── strategies/     # Trading strategy implementations
│   ├── risk/           # Risk management and position sizing
│   ├── options/        # Options analysis and Greeks
│   ├── signals/        # Signal generation and filtering
│   └── backtest/       # Backtesting engine and metrics
├── dashboard/          # Web dashboard (FastAPI)
├── infrastructure/     # AWS CDK / Terraform configs
├── tests/              # Unit and integration tests
├── config/             # Configuration management
└── scripts/            # Utility scripts
```

## Prerequisites

- Python 3.11+
- PostgreSQL 15+
- TA-Lib C library (see installation below)
- Interactive Brokers TWS or Gateway (for live trading)

## Installation

### 1. Install TA-Lib C Library

**macOS:**
```bash
brew install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```

### 2. Clone and Setup Project

```bash
git clone https://github.com/your-org/swing-trader.git
cd swing-trader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
vim .env
```

### 4. Setup Database

```bash
# Create PostgreSQL database
createdb swing_trader

# Run migrations (when available)
# alembic upgrade head
```

## Configuration

All configuration is managed via environment variables. See `.env.example` for required variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | Required |
| `IB_HOST` | Interactive Brokers host | `127.0.0.1` |
| `IB_PORT` | Interactive Brokers port | `7497` |
| `IB_CLIENT_ID` | IB client ID | `1` |
| `DATABASE_URL` | PostgreSQL connection URL | Required |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | Required |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | Required |
| `RISK_PER_TRADE` | Max risk per trade (decimal) | `0.01` |
| `MAX_POSITIONS` | Max concurrent positions | `5` |
| `MAX_DRAWDOWN` | Max drawdown threshold | `0.10` |

## Usage

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run only unit tests
pytest tests/unit/

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_indicators.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/
```

### Starting the Dashboard

```bash
uvicorn dashboard.main:app --reload --port 8000
```

## Development

### Project Structure

- **src/data/**: Data providers, caching, and normalization
- **src/indicators/**: Technical indicator calculations
- **src/strategies/**: Trading strategy base class and implementations
- **src/risk/**: Position sizing, stop-loss, and portfolio risk
- **src/options/**: Options pricing and Greeks
- **src/signals/**: Signal generation and aggregation
- **src/backtest/**: Historical testing framework

### Adding a New Strategy

1. Create a new file in `src/strategies/`
2. Inherit from the base `Strategy` class
3. Implement required methods: `generate_signals()`, `calculate_position_size()`
4. Register the strategy in the strategy factory

### Running Backtests

```python
from src.backtest import Backtester
from src.strategies import MomentumStrategy

backtester = Backtester(
    strategy=MomentumStrategy(),
    start_date="2023-01-01",
    end_date="2023-12-31",
    initial_capital=100000,
)
results = backtester.run()
print(results.summary())
```

## Testing

The project uses pytest with the following plugins:

- **pytest-cov**: Code coverage reporting
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking utilities
- **hypothesis**: Property-based testing
- **freezegun**: Time mocking
- **responses**: HTTP request mocking
- **factory-boy**: Test data factories

Coverage requirement: **80% minimum**

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest && ruff check .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading stocks and options involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and consider consulting a financial advisor before making investment decisions.
