"""Application settings using Pydantic BaseSettings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Alpha Vantage API
    alpha_vantage_api_key: str = Field(
        ...,
        description="API key for Alpha Vantage market data",
    )

    # Interactive Brokers Connection
    ib_host: str = Field(
        default="127.0.0.1",
        description="Interactive Brokers TWS/Gateway host",
    )
    ib_port: int = Field(
        default=7497,
        description="Interactive Brokers TWS/Gateway port (7497 for TWS paper, 7496 for TWS live)",
    )
    ib_client_id: int = Field(
        default=1,
        description="Interactive Brokers client ID",
    )

    # AWS Configuration
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for services",
    )
    s3_bucket: str = Field(
        ...,
        description="S3 bucket for data storage",
    )
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key ID (optional if using IAM roles)",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret access key (optional if using IAM roles)",
    )

    # Database
    database_url: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection URL",
    )

    # Telegram Notifications
    telegram_bot_token: str = Field(
        ...,
        description="Telegram bot token for notifications",
    )
    telegram_chat_id: str = Field(
        ...,
        description="Telegram chat ID for notifications",
    )

    # Risk Management
    risk_per_trade: float = Field(
        default=0.01,
        ge=0.001,
        le=0.10,
        description="Maximum risk per trade as decimal (0.01 = 1%)",
    )
    max_positions: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of concurrent positions",
    )
    max_drawdown: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Maximum drawdown threshold as decimal (0.10 = 10%)",
    )

    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment",
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def database_url_string(self) -> str:
        """Get database URL as string."""
        return str(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Application settings loaded from environment variables.
    """
    return Settings()  # type: ignore[call-arg]
