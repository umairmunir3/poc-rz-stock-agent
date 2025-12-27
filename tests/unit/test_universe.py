"""Unit tests for stock universe filtering."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from freezegun import freeze_time

from src.data.alpha_vantage import AlphaVantageClient
from src.data.exceptions import AlphaVantageError
from src.data.universe import (
    StockUniverse,
    UniverseCache,
    UniverseFilters,
)
from tests.fixtures.universe_data import (
    SAMPLE_LISTINGS,
    SAMPLE_OVERVIEWS,
)


class TestUniverseFilters:
    """Tests for UniverseFilters dataclass."""

    def test_default_values(self) -> None:
        """Test default filter values."""
        filters = UniverseFilters()

        assert filters.min_market_cap == 10_000_000_000
        assert filters.min_avg_volume == 1_000_000
        assert filters.min_price == 10.0
        assert filters.max_spread_pct == 0.05
        assert filters.exclude_adrs is True
        assert filters.exclude_etfs is True
        assert filters.exclude_reits is True
        assert filters.exclude_spacs is True

    def test_custom_values(self) -> None:
        """Test custom filter values."""
        filters = UniverseFilters(
            min_market_cap=5_000_000_000,
            min_avg_volume=500_000,
            exclude_reits=False,
        )

        assert filters.min_market_cap == 5_000_000_000
        assert filters.min_avg_volume == 500_000
        assert filters.exclude_reits is False

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        filters = UniverseFilters()
        result = filters.to_dict()

        assert isinstance(result, dict)
        assert result["min_market_cap"] == 10_000_000_000
        assert result["exclude_etfs"] is True


class TestUniverseCache:
    """Tests for UniverseCache."""

    def test_cache_creation(self) -> None:
        """Test cache creation and count calculation."""
        cache = UniverseCache(
            symbols=["AAPL", "MSFT", "GOOGL"],
            metadata={},
            last_updated=datetime.utcnow(),
            filters_used={},
        )

        assert cache.count == 3
        assert len(cache.symbols) == 3

    def test_cache_expiry_not_expired(self) -> None:
        """Test cache is not expired within 24 hours."""
        cache = UniverseCache(
            symbols=["AAPL"],
            metadata={},
            last_updated=datetime.utcnow() - timedelta(hours=12),
            filters_used={},
        )

        assert cache.is_expired(max_age_hours=24) is False

    def test_cache_expiry_expired(self) -> None:
        """Test cache is expired after 24 hours."""
        cache = UniverseCache(
            symbols=["AAPL"],
            metadata={},
            last_updated=datetime.utcnow() - timedelta(hours=25),
            filters_used={},
        )

        assert cache.is_expired(max_age_hours=24) is True

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization and deserialization."""
        original = UniverseCache(
            symbols=["AAPL", "MSFT"],
            metadata={"AAPL": {"sector": "Technology"}},
            last_updated=datetime(2024, 1, 15, 12, 0, 0),
            filters_used={"min_market_cap": 10000000000},
        )

        data = original.to_dict()
        restored = UniverseCache.from_dict(data)

        assert restored.symbols == original.symbols
        assert restored.metadata == original.metadata
        assert restored.filters_used == original.filters_used
        assert restored.count == original.count


class TestStockUniverse:
    """Tests for StockUniverse class."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock Alpha Vantage client."""
        client = AsyncMock(spec=AlphaVantageClient)

        async def mock_overview(symbol: str) -> dict:
            if symbol in SAMPLE_OVERVIEWS:
                return SAMPLE_OVERVIEWS[symbol]
            raise AlphaVantageError(f"Unknown symbol: {symbol}")

        client.get_company_overview = AsyncMock(side_effect=mock_overview)
        return client

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def universe(self, mock_client: AsyncMock, temp_cache_dir: Path) -> StockUniverse:
        """Create a StockUniverse instance with mocked dependencies."""
        return StockUniverse(
            alpha_vantage_client=mock_client,
            cache_dir=temp_cache_dir,
        )

    @pytest.mark.asyncio
    async def test_filters_by_market_cap(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that stocks below $10B market cap are excluded."""
        # Create listings with only market cap test stocks
        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "SMALL1", "name": "Small Cap Corp 1", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "SMALL2", "name": "Small Cap Corp 2", "exchange": "NASDAQ", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        # AAPL should be included (>$10B), SMALL1/SMALL2 should be excluded
        assert "AAPL" in result
        assert "SMALL1" not in result
        assert "SMALL2" not in result

    @pytest.mark.asyncio
    async def test_filters_by_price(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that penny stocks below $10 are excluded."""
        test_listings = [
            {"symbol": "MSFT", "name": "Microsoft", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "PENNY1", "name": "Penny Stock 1", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "PENNY2", "name": "Penny Stock 2", "exchange": "NASDAQ", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        # MSFT should be included, PENNY stocks should be excluded
        assert "MSFT" in result
        assert "PENNY1" not in result
        assert "PENNY2" not in result

    @pytest.mark.asyncio
    async def test_excludes_adrs(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that ADRs (symbols ending in .Y/.F or on OTC) are excluded."""
        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "TSM.Y", "name": "Taiwan Semi ADR", "exchange": "NYSE", "assetType": "Stock"},
            {"symbol": "NIO.F", "name": "NIO ADR", "exchange": "NYSE", "assetType": "Stock"},
            {"symbol": "SONY", "name": "Sony ADR", "exchange": "OTC", "assetType": "Stock"},
            {"symbol": "TCEHY", "name": "Tencent ADR", "exchange": "PINK", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        assert "AAPL" in result
        assert "TSM.Y" not in result
        assert "NIO.F" not in result
        assert "SONY" not in result
        assert "TCEHY" not in result

    @pytest.mark.asyncio
    async def test_excludes_etfs(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that ETF types are excluded."""
        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSE", "assetType": "ETF"},
            {"symbol": "QQQ", "name": "Invesco QQQ ETF", "exchange": "NASDAQ", "assetType": "ETF"},
            {"symbol": "GLD", "name": "SPDR Gold ETF", "exchange": "NYSE", "assetType": "ETN"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        assert "AAPL" in result
        assert "SPY" not in result
        assert "QQQ" not in result
        assert "GLD" not in result

    @pytest.mark.asyncio
    async def test_excludes_reits(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that REITs are excluded based on name and sector."""
        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "SPG", "name": "Simon Property REIT", "exchange": "NYSE", "assetType": "Stock"},
            {"symbol": "O", "name": "Realty Income REIT", "exchange": "NYSE", "assetType": "Stock"},
            {"symbol": "PLD", "name": "Prologis Real Estate Investment Trust", "exchange": "NYSE", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        assert "AAPL" in result
        assert "SPG" not in result
        assert "O" not in result
        assert "PLD" not in result

    @pytest.mark.asyncio
    async def test_excludes_spacs(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that SPACs are excluded based on name patterns."""
        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "PSTH", "name": "Pershing Square Acquisition Corp", "exchange": "NYSE", "assetType": "Stock"},
            {"symbol": "CCIV", "name": "Churchill Capital Acquisition Company", "exchange": "NYSE", "assetType": "Stock"},
            {"symbol": "DKNG", "name": "DraftKings Blank Check Company", "exchange": "NASDAQ", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        assert "AAPL" in result
        assert "PSTH" not in result
        assert "CCIV" not in result
        assert "DKNG" not in result

    @pytest.mark.asyncio
    async def test_combined_filters(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that all filters work together correctly."""
        # Use subset of sample listings
        test_listings = SAMPLE_LISTINGS[:35]  # Include various types

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        # Should only include valid large-cap stocks
        for symbol in result:
            # Verify each passed symbol is a valid large-cap stock
            assert symbol in SAMPLE_OVERVIEWS or symbol in [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
                "JPM", "V", "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",
                "BAC", "WFC", "GS", "MS", "C", "CAT", "BA", "HON", "UPS", "GE",
                "PG", "KO", "PEP", "WMT", "COST",
            ]

    @pytest.mark.asyncio
    @freeze_time("2024-01-15 12:00:00")
    async def test_cache_expiry(
        self, universe: StockUniverse, mock_client: AsyncMock, temp_cache_dir: Path
    ) -> None:
        """Test that cache refreshes after 24 hours."""
        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
        ]

        # Create an old cache
        old_cache = UniverseCache(
            symbols=["OLD_SYMBOL"],
            metadata={},
            last_updated=datetime(2024, 1, 14, 10, 0, 0),  # 26 hours ago
            filters_used=UniverseFilters().to_dict(),
        )
        cache_file = temp_cache_dir / "universe_cache.json"
        with open(cache_file, "w") as f:
            json.dump(old_cache.to_dict(), f)

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe()

        # Should have refreshed due to expiry
        assert "OLD_SYMBOL" not in result
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_uses_valid_cache(
        self, universe: StockUniverse, mock_client: AsyncMock, temp_cache_dir: Path
    ) -> None:
        """Test that valid cache is used without API calls."""
        # Create a valid cache
        valid_cache = UniverseCache(
            symbols=["CACHED_AAPL", "CACHED_MSFT"],
            metadata={},
            last_updated=datetime.utcnow() - timedelta(hours=1),
            filters_used=UniverseFilters().to_dict(),
        )
        cache_file = temp_cache_dir / "universe_cache.json"
        with open(cache_file, "w") as f:
            json.dump(valid_cache.to_dict(), f)

        result = await universe.build_universe()

        # Should use cache without calling fetch_listings
        assert result == ["CACHED_AAPL", "CACHED_MSFT"]

    @pytest.mark.asyncio
    async def test_refresh_ignores_cache(
        self, universe: StockUniverse, mock_client: AsyncMock, temp_cache_dir: Path
    ) -> None:
        """Test that refresh_universe ignores existing cache."""
        # Create a valid cache
        valid_cache = UniverseCache(
            symbols=["CACHED_SYMBOL"],
            metadata={},
            last_updated=datetime.utcnow() - timedelta(hours=1),
            filters_used=UniverseFilters().to_dict(),
        )
        cache_file = temp_cache_dir / "universe_cache.json"
        with open(cache_file, "w") as f:
            json.dump(valid_cache.to_dict(), f)

        test_listings = [
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.refresh_universe()

        # Should have fetched fresh data
        assert "CACHED_SYMBOL" not in result
        assert "AAPL" in result

    @pytest.mark.asyncio
    async def test_get_symbol_metadata(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test fetching metadata for a symbol."""
        metadata = await universe.get_symbol_metadata("AAPL")

        assert metadata.symbol == "AAPL"
        assert metadata.name == "Apple Inc"
        assert metadata.sector == "Technology"
        assert metadata.industry == "Consumer Electronics"
        assert metadata.market_cap == 3000000000000

    @pytest.mark.asyncio
    async def test_get_symbol_metadata_caches(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test that metadata is cached after first fetch."""
        # First call
        await universe.get_symbol_metadata("AAPL")

        # Second call should use cache
        await universe.get_symbol_metadata("AAPL")

        # API should only be called once
        assert mock_client.get_company_overview.call_count == 1

    @pytest.mark.asyncio
    async def test_get_symbol_metadata_handles_error(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test graceful handling of API errors for metadata."""
        mock_client.get_company_overview.side_effect = AlphaVantageError("API error")

        metadata = await universe.get_symbol_metadata("UNKNOWN")

        # Should return minimal metadata
        assert metadata.symbol == "UNKNOWN"
        assert metadata.name == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_filter_by_sector(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test filtering symbols by sector."""
        # Pre-populate cache with tech stocks
        test_listings = [
            {"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "MSFT", "name": "Microsoft", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "JPM", "name": "JPMorgan", "exchange": "NYSE", "assetType": "Stock"},
        ]

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            await universe.build_universe()

        tech_stocks = await universe.filter_by_sector("Technology")

        assert "AAPL" in tech_stocks
        assert "MSFT" in tech_stocks
        assert "JPM" not in tech_stocks  # Financial Services

    @pytest.mark.asyncio
    async def test_custom_filters(
        self, universe: StockUniverse, mock_client: AsyncMock
    ) -> None:
        """Test using custom filter criteria."""
        test_listings = [
            {"symbol": "AAPL", "name": "Apple", "exchange": "NASDAQ", "assetType": "Stock"},
            {"symbol": "SMALL3", "name": "Small Cap", "exchange": "NYSE", "assetType": "Stock"},
        ]

        # Use lower market cap threshold
        custom_filters = UniverseFilters(min_market_cap=5_000_000_000)

        with patch.object(universe, "_fetch_listings", return_value=test_listings):
            result = await universe.build_universe(filters=custom_filters)

        # Both should be included with lower threshold
        assert "AAPL" in result
        assert "SMALL3" in result  # $8B > $5B threshold


class TestQuickFilters:
    """Tests for quick filter methods."""

    @pytest.fixture
    def universe(self) -> StockUniverse:
        """Create a universe with mock client."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        return StockUniverse(alpha_vantage_client=mock_client)

    def test_adr_suffix_detection(self, universe: StockUniverse) -> None:
        """Test ADR detection by symbol suffix."""
        filters = UniverseFilters(exclude_adrs=True)

        # Should fail for .Y suffix
        result = universe._passes_quick_filters(
            {"symbol": "TSM.Y", "name": "Taiwan Semi", "exchange": "NYSE", "assetType": "Stock"},
            filters,
        )
        assert result is False

        # Should fail for .F suffix
        result = universe._passes_quick_filters(
            {"symbol": "NIO.F", "name": "NIO", "exchange": "NYSE", "assetType": "Stock"},
            filters,
        )
        assert result is False

    def test_adr_exchange_detection(self, universe: StockUniverse) -> None:
        """Test ADR detection by exchange."""
        filters = UniverseFilters(exclude_adrs=True)

        # Should fail for OTC exchange
        result = universe._passes_quick_filters(
            {"symbol": "SONY", "name": "Sony", "exchange": "OTC", "assetType": "Stock"},
            filters,
        )
        assert result is False

        # Should fail for PINK exchange
        result = universe._passes_quick_filters(
            {"symbol": "TCEHY", "name": "Tencent", "exchange": "PINK", "assetType": "Stock"},
            filters,
        )
        assert result is False

    def test_etf_asset_type_detection(self, universe: StockUniverse) -> None:
        """Test ETF detection by asset type."""
        filters = UniverseFilters(exclude_etfs=True)

        # Should fail for ETF type
        result = universe._passes_quick_filters(
            {"symbol": "SPY", "name": "SPDR S&P 500", "exchange": "NYSE", "assetType": "ETF"},
            filters,
        )
        assert result is False

        # Should fail for ETN type
        result = universe._passes_quick_filters(
            {"symbol": "GLD", "name": "SPDR Gold", "exchange": "NYSE", "assetType": "ETN"},
            filters,
        )
        assert result is False

    def test_etf_name_detection(self, universe: StockUniverse) -> None:
        """Test ETF detection by name."""
        filters = UniverseFilters(exclude_etfs=True)

        result = universe._passes_quick_filters(
            {"symbol": "TEST", "name": "Test ETF Fund", "exchange": "NYSE", "assetType": "Stock"},
            filters,
        )
        assert result is False

    def test_spac_detection(self, universe: StockUniverse) -> None:
        """Test SPAC detection by name patterns."""
        filters = UniverseFilters(exclude_spacs=True)

        # "Acquisition Corp"
        result = universe._passes_quick_filters(
            {"symbol": "TEST", "name": "Test Acquisition Corp", "exchange": "NYSE", "assetType": "Stock"},
            filters,
        )
        assert result is False

        # "Blank Check"
        result = universe._passes_quick_filters(
            {"symbol": "TEST", "name": "Test Blank Check Company", "exchange": "NYSE", "assetType": "Stock"},
            filters,
        )
        assert result is False

    def test_reit_name_detection(self, universe: StockUniverse) -> None:
        """Test REIT detection by name."""
        filters = UniverseFilters(exclude_reits=True)

        result = universe._passes_quick_filters(
            {"symbol": "TEST", "name": "Test REIT Corp", "exchange": "NYSE", "assetType": "Stock"},
            filters,
        )
        assert result is False

    def test_valid_stock_passes(self, universe: StockUniverse) -> None:
        """Test that valid stocks pass quick filters."""
        filters = UniverseFilters()

        result = universe._passes_quick_filters(
            {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
            filters,
        )
        assert result is True


class TestCacheOperations:
    """Tests for cache save/load operations."""

    @pytest.fixture
    def temp_cache_dir(self) -> Path:
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def universe(self, temp_cache_dir: Path) -> StockUniverse:
        """Create a universe with mock client."""
        mock_client = AsyncMock(spec=AlphaVantageClient)
        return StockUniverse(
            alpha_vantage_client=mock_client,
            cache_dir=temp_cache_dir,
        )

    @pytest.mark.asyncio
    async def test_save_and_load_local_cache(
        self, universe: StockUniverse, temp_cache_dir: Path
    ) -> None:
        """Test saving and loading cache from local file."""
        cache = UniverseCache(
            symbols=["AAPL", "MSFT"],
            metadata={"AAPL": {"sector": "Technology"}},
            last_updated=datetime.utcnow(),
            filters_used={"min_market_cap": 10000000000},
        )

        await universe._save_cache_local(cache)

        loaded = await universe._load_cache_local()

        assert loaded is not None
        assert loaded.symbols == cache.symbols
        assert loaded.metadata == cache.metadata

    @pytest.mark.asyncio
    async def test_load_missing_cache(
        self, universe: StockUniverse, temp_cache_dir: Path
    ) -> None:
        """Test loading when cache file doesn't exist."""
        loaded = await universe._load_cache_local()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_corrupted_cache(
        self, universe: StockUniverse, temp_cache_dir: Path
    ) -> None:
        """Test loading corrupted cache file."""
        cache_file = temp_cache_dir / "universe_cache.json"
        with open(cache_file, "w") as f:
            f.write("invalid json{")

        loaded = await universe._load_cache_local()
        assert loaded is None
