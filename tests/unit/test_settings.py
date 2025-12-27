"""Tests for configuration settings."""

import os
from unittest.mock import patch

import pytest

from config.settings import Settings


class TestSettings:
    """Test suite for Settings class."""

    def test_settings_loads_from_environment(self) -> None:
        """Test that settings load correctly from environment variables."""
        env_vars = {
            "ALPHA_VANTAGE_API_KEY": "test_key",
            "IB_HOST": "192.168.1.1",
            "IB_PORT": "7496",
            "IB_CLIENT_ID": "2",
            "AWS_REGION": "eu-west-1",
            "S3_BUCKET": "my-bucket",
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
            "TELEGRAM_BOT_TOKEN": "bot_token",
            "TELEGRAM_CHAT_ID": "chat_123",
            "RISK_PER_TRADE": "0.02",
            "MAX_POSITIONS": "10",
            "MAX_DRAWDOWN": "0.15",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.alpha_vantage_api_key == "test_key"
            assert settings.ib_host == "192.168.1.1"
            assert settings.ib_port == 7496
            assert settings.ib_client_id == 2
            assert settings.aws_region == "eu-west-1"
            assert settings.s3_bucket == "my-bucket"
            assert settings.telegram_bot_token == "bot_token"
            assert settings.telegram_chat_id == "chat_123"
            assert settings.risk_per_trade == 0.02
            assert settings.max_positions == 10
            assert settings.max_drawdown == 0.15

    def test_settings_default_values(self) -> None:
        """Test that default values are applied correctly."""
        env_vars = {
            "ALPHA_VANTAGE_API_KEY": "test_key",
            "S3_BUCKET": "my-bucket",
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
            "TELEGRAM_BOT_TOKEN": "bot_token",
            "TELEGRAM_CHAT_ID": "chat_123",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.ib_host == "127.0.0.1"
            assert settings.ib_port == 7497
            assert settings.ib_client_id == 1
            assert settings.aws_region == "us-east-1"
            assert settings.risk_per_trade == 0.01
            assert settings.max_positions == 5
            assert settings.max_drawdown == 0.10
            assert settings.environment == "development"
            assert settings.log_level == "INFO"

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        env_vars = {
            "ALPHA_VANTAGE_API_KEY": "test_key",
            "S3_BUCKET": "my-bucket",
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
            "TELEGRAM_BOT_TOKEN": "bot_token",
            "TELEGRAM_CHAT_ID": "chat_123",
            "ENVIRONMENT": "production",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.is_production is True

        env_vars["ENVIRONMENT"] = "development"
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.is_production is False

    def test_risk_per_trade_validation(self) -> None:
        """Test that risk_per_trade is validated within bounds."""
        env_vars = {
            "ALPHA_VANTAGE_API_KEY": "test_key",
            "S3_BUCKET": "my-bucket",
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
            "TELEGRAM_BOT_TOKEN": "bot_token",
            "TELEGRAM_CHAT_ID": "chat_123",
            "RISK_PER_TRADE": "0.50",  # Too high
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValueError):
                Settings()

    def test_database_url_string_property(self) -> None:
        """Test database_url_string property returns string."""
        env_vars = {
            "ALPHA_VANTAGE_API_KEY": "test_key",
            "S3_BUCKET": "my-bucket",
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/db",
            "TELEGRAM_BOT_TOKEN": "bot_token",
            "TELEGRAM_CHAT_ID": "chat_123",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert isinstance(settings.database_url_string, str)
            assert "postgresql" in settings.database_url_string
