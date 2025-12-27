"""Stock universe filtering for tradeable large-cap US equities."""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.data.alpha_vantage import AlphaVantageClient
from src.data.exceptions import AlphaVantageError

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path(".cache")


@dataclass
class UniverseFilters:
    """Configurable filtering criteria for stock universe."""

    min_market_cap: float = 10_000_000_000  # $10B
    min_avg_volume: int = 1_000_000  # 1M shares daily
    min_price: float = 10.0  # Minimum share price
    max_spread_pct: float = 0.05  # 5% max bid-ask spread for options
    exclude_adrs: bool = True  # Exclude American Depositary Receipts
    exclude_etfs: bool = True  # Exclude Exchange Traded Funds
    exclude_reits: bool = True  # Exclude Real Estate Investment Trusts
    exclude_spacs: bool = True  # Exclude Special Purpose Acquisition Companies

    def to_dict(self) -> dict[str, Any]:
        """Convert filters to dictionary for caching."""
        return {
            "min_market_cap": self.min_market_cap,
            "min_avg_volume": self.min_avg_volume,
            "min_price": self.min_price,
            "max_spread_pct": self.max_spread_pct,
            "exclude_adrs": self.exclude_adrs,
            "exclude_etfs": self.exclude_etfs,
            "exclude_reits": self.exclude_reits,
            "exclude_spacs": self.exclude_spacs,
        }


class SymbolMetadata(BaseModel):
    """Metadata for a stock symbol."""

    symbol: str
    name: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    avg_volume: int | None = None
    price: float | None = None
    exchange: str | None = None
    asset_type: str | None = None
    ipo_date: str | None = None
    is_adr: bool = False
    is_etf: bool = False
    is_reit: bool = False
    is_spac: bool = False


@dataclass
class UniverseCache:
    """Cached universe data with metadata."""

    symbols: list[str]
    metadata: dict[str, dict[str, Any]]
    last_updated: datetime
    filters_used: dict[str, Any]
    count: int = field(init=False)

    def __post_init__(self) -> None:
        """Calculate count after initialization."""
        self.count = len(self.symbols)

    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if cache has expired."""
        age = datetime.utcnow() - self.last_updated
        return age > timedelta(hours=max_age_hours)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbols": self.symbols,
            "metadata": self.metadata,
            "last_updated": self.last_updated.isoformat(),
            "filters_used": self.filters_used,
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniverseCache":
        """Create from dictionary."""
        return cls(
            symbols=data["symbols"],
            metadata=data["metadata"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
            filters_used=data["filters_used"],
        )


# ADR suffixes and patterns
ADR_SUFFIXES = {".Y", ".F"}
ADR_EXCHANGE_PATTERNS = {"OTC", "PINK"}

# ETF asset types
ETF_ASSET_TYPES = {"ETF", "ETN"}

# REIT keywords in industry/name
REIT_KEYWORDS = {"REIT", "Real Estate Investment Trust", "REAL ESTATE INVESTMENT"}

# SPAC keywords in name
SPAC_KEYWORDS = {"SPAC", "Acquisition Corp", "Acquisition Company", "Blank Check"}


class StockUniverse:
    """Filters US equity market to tradeable large-cap stocks.

    Maintains a universe of ~500-800 stocks meeting liquidity,
    market cap, and other trading criteria.
    """

    def __init__(
        self,
        alpha_vantage_client: AlphaVantageClient,
        cache_dir: Path | None = None,
        s3_bucket: str | None = None,
    ) -> None:
        """Initialize the stock universe.

        Args:
            alpha_vantage_client: Client for Alpha Vantage API.
            cache_dir: Local directory for cache files.
            s3_bucket: Optional S3 bucket for cache storage.
        """
        self.client = alpha_vantage_client
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.s3_bucket = s3_bucket
        self._cache: UniverseCache | None = None
        self._metadata_cache: dict[str, SymbolMetadata] = {}

    async def build_universe(
        self,
        filters: UniverseFilters | None = None,
    ) -> list[str]:
        """Build the stock universe based on filtering criteria.

        Args:
            filters: Filtering criteria. Uses defaults if not provided.

        Returns:
            List of symbols meeting all criteria.
        """
        if filters is None:
            filters = UniverseFilters()

        # Try to load from cache first
        cached = await self._load_cache()
        if cached and not cached.is_expired() and cached.filters_used == filters.to_dict():
            logger.info(f"Using cached universe with {cached.count} symbols")
            self._cache = cached
            return cached.symbols

        # Build fresh universe
        logger.info("Building fresh stock universe...")
        symbols = await self._fetch_and_filter(filters)

        # Cache the result - convert SymbolMetadata to dicts for JSON serialization
        metadata_dict = {}
        for s in symbols:
            meta = self._metadata_cache.get(s)
            if meta is not None:
                metadata_dict[s] = meta.model_dump() if hasattr(meta, "model_dump") else meta
            else:
                metadata_dict[s] = {}

        self._cache = UniverseCache(
            symbols=symbols,
            metadata=metadata_dict,
            last_updated=datetime.utcnow(),
            filters_used=filters.to_dict(),
        )
        await self._save_cache(self._cache)

        logger.info(f"Built universe with {len(symbols)} symbols")
        return symbols

    async def refresh_universe(
        self,
        filters: UniverseFilters | None = None,
    ) -> list[str]:
        """Force rebuild of the universe, ignoring cache.

        Args:
            filters: Filtering criteria. Uses defaults if not provided.

        Returns:
            List of symbols meeting all criteria.
        """
        # Clear cache to force rebuild
        self._cache = None
        if filters is None:
            filters = UniverseFilters()

        logger.info("Forcing universe refresh...")
        symbols = await self._fetch_and_filter(filters)

        # Convert SymbolMetadata to dicts for JSON serialization
        metadata_dict = {}
        for s in symbols:
            meta = self._metadata_cache.get(s)
            if meta is not None:
                metadata_dict[s] = meta.model_dump() if hasattr(meta, "model_dump") else meta
            else:
                metadata_dict[s] = {}

        self._cache = UniverseCache(
            symbols=symbols,
            metadata=metadata_dict,
            last_updated=datetime.utcnow(),
            filters_used=filters.to_dict(),
        )
        await self._save_cache(self._cache)

        return symbols

    async def get_symbol_metadata(self, symbol: str) -> SymbolMetadata:
        """Get metadata for a specific symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            SymbolMetadata with sector, industry, market_cap, etc.
        """
        # Check memory cache first
        if symbol in self._metadata_cache:
            return self._metadata_cache[symbol]

        # Fetch from API
        try:
            overview = await self.client.get_company_overview(symbol)
            metadata = SymbolMetadata(
                symbol=symbol,
                name=overview.get("name", ""),
                sector=overview.get("sector"),
                industry=overview.get("industry"),
                market_cap=overview.get("market_capitalization"),
                exchange=overview.get("exchange"),
                asset_type=overview.get("asset_type"),
            )
            self._metadata_cache[symbol] = metadata
            return metadata
        except AlphaVantageError as e:
            logger.warning(f"Failed to get metadata for {symbol}: {e}")
            # Return minimal metadata
            return SymbolMetadata(symbol=symbol, name=symbol)

    async def filter_by_sector(self, sector: str) -> list[str]:
        """Get symbols in a specific sector.

        Args:
            sector: Sector name (e.g., "Technology", "Healthcare").

        Returns:
            List of symbols in the specified sector.
        """
        if self._cache is None:
            await self.build_universe()

        result = []
        for symbol in self._cache.symbols:
            metadata = await self.get_symbol_metadata(symbol)
            if metadata.sector and metadata.sector.upper() == sector.upper():
                result.append(symbol)

        return result

    async def _fetch_and_filter(self, filters: UniverseFilters) -> list[str]:
        """Fetch stock listings and apply filters.

        Args:
            filters: Filtering criteria to apply.

        Returns:
            List of symbols meeting all criteria.
        """
        # Fetch active listings
        listings = await self._fetch_listings()

        # Apply filters progressively
        filtered = []
        batch_size = 5  # Process in batches to respect rate limits

        for i in range(0, len(listings), batch_size):
            batch = listings[i : i + batch_size]

            for listing in batch:
                symbol = listing.get("symbol", "")
                if not symbol:
                    continue

                # Quick filters that don't require API calls
                if not self._passes_quick_filters(listing, filters):
                    continue

                # Detailed filters requiring API enrichment
                try:
                    if await self._passes_detailed_filters(symbol, listing, filters):
                        filtered.append(symbol)
                except AlphaVantageError as e:
                    logger.debug(f"Skipping {symbol} due to API error: {e}")
                    continue

        return sorted(filtered)

    async def _fetch_listings(self) -> list[dict[str, Any]]:
        """Fetch active stock listings from Alpha Vantage.

        Returns:
            List of listing dictionaries with symbol, name, exchange, etc.
        """
        # Alpha Vantage listing status endpoint
        params = {
            "function": "LISTING_STATUS",
            "state": "active",
        }

        try:
            # Use the client's _request method directly for this endpoint
            data = await self.client._request(params)

            # Parse CSV-like response
            if isinstance(data, str):
                return self._parse_listing_csv(data)
            elif isinstance(data, list):
                return data
            else:
                logger.warning(f"Unexpected listing response format: {type(data)}")
                return []

        except AlphaVantageError as e:
            logger.error(f"Failed to fetch listings: {e}")
            # Try to load from cache as fallback
            cached = await self._load_cache()
            if cached:
                logger.info("Using cached listings as fallback")
                return [{"symbol": s, **cached.metadata.get(s, {})} for s in cached.symbols]
            return []

    def _parse_listing_csv(self, csv_data: str) -> list[dict[str, Any]]:
        """Parse CSV listing data.

        Args:
            csv_data: CSV string from API.

        Returns:
            List of listing dictionaries.
        """
        lines = csv_data.strip().split("\n")
        if len(lines) < 2:
            return []

        headers = [h.strip().lower().replace(" ", "_") for h in lines[0].split(",")]
        listings = []

        for line in lines[1:]:
            values = line.split(",")
            if len(values) >= len(headers):
                listing = dict(zip(headers, values))
                listings.append(listing)

        return listings

    def _passes_quick_filters(
        self,
        listing: dict[str, Any],
        filters: UniverseFilters,
    ) -> bool:
        """Apply quick filters that don't require API calls.

        Args:
            listing: Listing data dictionary.
            filters: Filtering criteria.

        Returns:
            True if listing passes all quick filters.
        """
        symbol = listing.get("symbol", "")
        name = listing.get("name", "").upper()
        asset_type = listing.get("assetType", listing.get("asset_type", "")).upper()
        exchange = listing.get("exchange", "").upper()

        # Exclude ADRs
        if filters.exclude_adrs:
            if any(symbol.endswith(suffix) for suffix in ADR_SUFFIXES):
                return False
            if any(pattern in exchange for pattern in ADR_EXCHANGE_PATTERNS):
                return False

        # Exclude ETFs
        if filters.exclude_etfs:
            if asset_type in ETF_ASSET_TYPES:
                return False
            if "ETF" in name:
                return False

        # Exclude SPACs
        if filters.exclude_spacs:
            if any(keyword.upper() in name for keyword in SPAC_KEYWORDS):
                return False

        # Exclude REITs (basic check - detailed check in detailed filters)
        if filters.exclude_reits:
            if any(keyword.upper() in name for keyword in REIT_KEYWORDS):
                return False

        return True

    async def _passes_detailed_filters(
        self,
        symbol: str,
        listing: dict[str, Any],
        filters: UniverseFilters,
    ) -> bool:
        """Apply detailed filters requiring API enrichment.

        Args:
            symbol: Stock ticker symbol.
            listing: Basic listing data.
            filters: Filtering criteria.

        Returns:
            True if symbol passes all detailed filters.
        """
        # Get company overview for detailed data
        try:
            overview = await self.client.get_company_overview(symbol)
        except AlphaVantageError:
            return False

        # Extract values with safe defaults
        market_cap = overview.get("market_capitalization")
        if market_cap is None:
            return False

        # Market cap filter
        if market_cap < filters.min_market_cap:
            return False

        # Price filter
        price = overview.get("fifty_two_week_high") or overview.get("fifty_two_week_low")
        if price is not None and price < filters.min_price:
            return False

        # REIT filter (detailed check using sector/industry)
        if filters.exclude_reits:
            sector = (overview.get("sector") or "").upper()
            industry = (overview.get("industry") or "").upper()
            if "REAL ESTATE" in sector or any(
                kw.upper() in industry for kw in REIT_KEYWORDS
            ):
                return False

        # Store metadata for caching
        self._metadata_cache[symbol] = SymbolMetadata(
            symbol=symbol,
            name=overview.get("name", ""),
            sector=overview.get("sector"),
            industry=overview.get("industry"),
            market_cap=market_cap,
            exchange=overview.get("exchange"),
            asset_type=listing.get("assetType", listing.get("asset_type")),
        )

        return True

    async def _load_cache(self) -> UniverseCache | None:
        """Load universe cache from storage.

        Returns:
            UniverseCache if found and valid, None otherwise.
        """
        # Try S3 first if configured
        if self.s3_bucket:
            cache = await self._load_cache_s3()
            if cache:
                return cache

        # Fall back to local file
        return await self._load_cache_local()

    async def _load_cache_local(self) -> UniverseCache | None:
        """Load cache from local file.

        Returns:
            UniverseCache if found, None otherwise.
        """
        cache_file = self.cache_dir / "universe_cache.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as f:
                data = json.load(f)
            return UniverseCache.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to load local cache: {e}")
            return None

    async def _load_cache_s3(self) -> UniverseCache | None:
        """Load cache from S3.

        Returns:
            UniverseCache if found, None otherwise.
        """
        if not self.s3_bucket:
            return None

        try:
            import boto3

            s3 = boto3.client("s3")
            response = s3.get_object(
                Bucket=self.s3_bucket,
                Key="universe/universe_cache.json",
            )
            data = json.loads(response["Body"].read().decode("utf-8"))
            return UniverseCache.from_dict(data)
        except Exception as e:
            logger.debug(f"Failed to load S3 cache: {e}")
            return None

    async def _save_cache(self, cache: UniverseCache) -> None:
        """Save universe cache to storage.

        Args:
            cache: UniverseCache to save.
        """
        # Always save locally
        await self._save_cache_local(cache)

        # Also save to S3 if configured
        if self.s3_bucket:
            await self._save_cache_s3(cache)

    async def _save_cache_local(self, cache: UniverseCache) -> None:
        """Save cache to local file.

        Args:
            cache: UniverseCache to save.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "universe_cache.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(cache.to_dict(), f, indent=2)
            logger.debug(f"Saved universe cache to {cache_file}")
        except OSError as e:
            logger.warning(f"Failed to save local cache: {e}")

    async def _save_cache_s3(self, cache: UniverseCache) -> None:
        """Save cache to S3.

        Args:
            cache: UniverseCache to save.
        """
        if not self.s3_bucket:
            return

        try:
            import boto3

            s3 = boto3.client("s3")
            s3.put_object(
                Bucket=self.s3_bucket,
                Key="universe/universe_cache.json",
                Body=json.dumps(cache.to_dict(), indent=2),
                ContentType="application/json",
            )
            logger.debug(f"Saved universe cache to S3: {self.s3_bucket}")
        except Exception as e:
            logger.warning(f"Failed to save S3 cache: {e}")
