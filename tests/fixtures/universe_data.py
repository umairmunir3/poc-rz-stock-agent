"""Sample stock data for universe filtering tests.

Contains 50 sample stocks with varied characteristics to test all filter criteria.
"""

from typing import Any

# Sample stock listings (simulating Alpha Vantage listing status response)
SAMPLE_LISTINGS: list[dict[str, Any]] = [
    # Large-cap tech stocks (should pass)
    {"symbol": "AAPL", "name": "Apple Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "GOOGL", "name": "Alphabet Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "AMZN", "name": "Amazon.com Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "META", "name": "Meta Platforms Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "TSLA", "name": "Tesla Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "V", "name": "Visa Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE", "assetType": "Stock"},

    # Healthcare stocks (should pass)
    {"symbol": "UNH", "name": "UnitedHealth Group Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "PFE", "name": "Pfizer Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "MRK", "name": "Merck & Co Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "ABBV", "name": "AbbVie Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "LLY", "name": "Eli Lilly and Company", "exchange": "NYSE", "assetType": "Stock"},

    # Financial stocks (should pass)
    {"symbol": "BAC", "name": "Bank of America Corp", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "WFC", "name": "Wells Fargo & Company", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "GS", "name": "Goldman Sachs Group Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "MS", "name": "Morgan Stanley", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "C", "name": "Citigroup Inc", "exchange": "NYSE", "assetType": "Stock"},

    # Industrial stocks (should pass)
    {"symbol": "CAT", "name": "Caterpillar Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "BA", "name": "Boeing Company", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "HON", "name": "Honeywell International Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "UPS", "name": "United Parcel Service Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "GE", "name": "General Electric Company", "exchange": "NYSE", "assetType": "Stock"},

    # Consumer stocks (should pass)
    {"symbol": "PG", "name": "Procter & Gamble Company", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "KO", "name": "Coca-Cola Company", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "PEP", "name": "PepsiCo Inc", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "WMT", "name": "Walmart Inc", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "COST", "name": "Costco Wholesale Corporation", "exchange": "NASDAQ", "assetType": "Stock"},

    # ADRs (should be excluded with exclude_adrs=True)
    {"symbol": "TSM.Y", "name": "Taiwan Semiconductor ADR", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "BABA.Y", "name": "Alibaba Group ADR", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "NIO.F", "name": "NIO Inc ADR", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "SONY", "name": "Sony Group Corporation ADR", "exchange": "OTC", "assetType": "Stock"},
    {"symbol": "TCEHY", "name": "Tencent Holdings ADR", "exchange": "PINK", "assetType": "Stock"},

    # ETFs (should be excluded with exclude_etfs=True)
    {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "exchange": "NYSE", "assetType": "ETF"},
    {"symbol": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ", "assetType": "ETF"},
    {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "exchange": "NYSE", "assetType": "ETF"},
    {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "exchange": "NYSE", "assetType": "ETF"},
    {"symbol": "GLD", "name": "SPDR Gold Shares ETF", "exchange": "NYSE", "assetType": "ETN"},

    # REITs (should be excluded with exclude_reits=True)
    {"symbol": "SPG", "name": "Simon Property Group REIT", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "O", "name": "Realty Income Corporation REIT", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "AMT", "name": "American Tower REIT Corporation", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "PLD", "name": "Prologis Real Estate Investment Trust", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "EQIX", "name": "Equinix Inc REIT", "exchange": "NASDAQ", "assetType": "Stock"},

    # SPACs (should be excluded with exclude_spacs=True)
    {"symbol": "PSTH", "name": "Pershing Square Tontine Acquisition Corp", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "CCIV", "name": "Churchill Capital Acquisition Company", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "SPCE", "name": "Virgin Galactic SPAC Holdings", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "DKNG", "name": "DraftKings Blank Check Company", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "IPOF", "name": "Social Capital Hedosophia Acquisition Corp", "exchange": "NYSE", "assetType": "Stock"},
]

# Company overview data keyed by symbol
SAMPLE_OVERVIEWS: dict[str, dict[str, Any]] = {
    # Large-cap tech (pass all filters)
    "AAPL": {
        "symbol": "AAPL",
        "name": "Apple Inc",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_capitalization": 3000000000000,  # $3T
        "exchange": "NASDAQ",
        "fifty_two_week_high": 199.62,
        "fifty_two_week_low": 124.17,
    },
    "MSFT": {
        "symbol": "MSFT",
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software",
        "market_capitalization": 2800000000000,  # $2.8T
        "exchange": "NASDAQ",
        "fifty_two_week_high": 384.30,
        "fifty_two_week_low": 275.37,
    },
    "GOOGL": {
        "symbol": "GOOGL",
        "name": "Alphabet Inc",
        "sector": "Technology",
        "industry": "Internet Content & Information",
        "market_capitalization": 1700000000000,  # $1.7T
        "exchange": "NASDAQ",
        "fifty_two_week_high": 153.78,
        "fifty_two_week_low": 102.21,
    },
    "AMZN": {
        "symbol": "AMZN",
        "name": "Amazon.com Inc",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "market_capitalization": 1600000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 189.77,
        "fifty_two_week_low": 118.35,
    },
    "META": {
        "symbol": "META",
        "name": "Meta Platforms Inc",
        "sector": "Technology",
        "industry": "Internet Content & Information",
        "market_capitalization": 900000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 384.33,
        "fifty_two_week_low": 172.89,
    },
    "NVDA": {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_capitalization": 1200000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 502.66,
        "fifty_two_week_low": 138.84,
    },
    "TSLA": {
        "symbol": "TSLA",
        "name": "Tesla Inc",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "market_capitalization": 800000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 299.29,
        "fifty_two_week_low": 152.37,
    },
    "JPM": {
        "symbol": "JPM",
        "name": "JPMorgan Chase & Co",
        "sector": "Financial Services",
        "industry": "Banks",
        "market_capitalization": 450000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 172.96,
        "fifty_two_week_low": 123.11,
    },
    "V": {
        "symbol": "V",
        "name": "Visa Inc",
        "sector": "Financial Services",
        "industry": "Credit Services",
        "market_capitalization": 500000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 290.96,
        "fifty_two_week_low": 206.70,
    },
    "JNJ": {
        "symbol": "JNJ",
        "name": "Johnson & Johnson",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "market_capitalization": 400000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 175.97,
        "fifty_two_week_low": 144.95,
    },

    # Healthcare
    "UNH": {
        "symbol": "UNH",
        "name": "UnitedHealth Group Inc",
        "sector": "Healthcare",
        "industry": "Healthcare Plans",
        "market_capitalization": 500000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 554.98,
        "fifty_two_week_low": 436.38,
    },
    "PFE": {
        "symbol": "PFE",
        "name": "Pfizer Inc",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "market_capitalization": 150000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 43.05,
        "fifty_two_week_low": 25.20,
    },
    "MRK": {
        "symbol": "MRK",
        "name": "Merck & Co Inc",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "market_capitalization": 280000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 119.65,
        "fifty_two_week_low": 99.14,
    },
    "ABBV": {
        "symbol": "ABBV",
        "name": "AbbVie Inc",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "market_capitalization": 290000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 175.91,
        "fifty_two_week_low": 130.96,
    },
    "LLY": {
        "symbol": "LLY",
        "name": "Eli Lilly and Company",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers",
        "market_capitalization": 550000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 629.97,
        "fifty_two_week_low": 309.20,
    },

    # Financial
    "BAC": {
        "symbol": "BAC",
        "name": "Bank of America Corp",
        "sector": "Financial Services",
        "industry": "Banks",
        "market_capitalization": 250000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 35.08,
        "fifty_two_week_low": 24.96,
    },
    "WFC": {
        "symbol": "WFC",
        "name": "Wells Fargo & Company",
        "sector": "Financial Services",
        "industry": "Banks",
        "market_capitalization": 180000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 51.99,
        "fifty_two_week_low": 35.25,
    },
    "GS": {
        "symbol": "GS",
        "name": "Goldman Sachs Group Inc",
        "sector": "Financial Services",
        "industry": "Capital Markets",
        "market_capitalization": 120000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 389.58,
        "fifty_two_week_low": 289.36,
    },
    "MS": {
        "symbol": "MS",
        "name": "Morgan Stanley",
        "sector": "Financial Services",
        "industry": "Capital Markets",
        "market_capitalization": 140000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 94.11,
        "fifty_two_week_low": 72.35,
    },
    "C": {
        "symbol": "C",
        "name": "Citigroup Inc",
        "sector": "Financial Services",
        "industry": "Banks",
        "market_capitalization": 95000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 53.23,
        "fifty_two_week_low": 38.17,
    },

    # Industrial
    "CAT": {
        "symbol": "CAT",
        "name": "Caterpillar Inc",
        "sector": "Industrials",
        "industry": "Farm & Heavy Construction Machinery",
        "market_capitalization": 140000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 293.88,
        "fifty_two_week_low": 197.25,
    },
    "BA": {
        "symbol": "BA",
        "name": "Boeing Company",
        "sector": "Industrials",
        "industry": "Aerospace & Defense",
        "market_capitalization": 130000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 267.54,
        "fifty_two_week_low": 159.70,
    },
    "HON": {
        "symbol": "HON",
        "name": "Honeywell International Inc",
        "sector": "Industrials",
        "industry": "Conglomerates",
        "market_capitalization": 130000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 216.77,
        "fifty_two_week_low": 172.51,
    },
    "UPS": {
        "symbol": "UPS",
        "name": "United Parcel Service Inc",
        "sector": "Industrials",
        "industry": "Integrated Freight & Logistics",
        "market_capitalization": 120000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 193.74,
        "fifty_two_week_low": 134.78,
    },
    "GE": {
        "symbol": "GE",
        "name": "General Electric Company",
        "sector": "Industrials",
        "industry": "Aerospace & Defense",
        "market_capitalization": 140000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 124.86,
        "fifty_two_week_low": 81.90,
    },

    # Consumer
    "PG": {
        "symbol": "PG",
        "name": "Procter & Gamble Company",
        "sector": "Consumer Defensive",
        "industry": "Household & Personal Products",
        "market_capitalization": 350000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 161.17,
        "fifty_two_week_low": 137.67,
    },
    "KO": {
        "symbol": "KO",
        "name": "Coca-Cola Company",
        "sector": "Consumer Defensive",
        "industry": "Beverages",
        "market_capitalization": 260000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 64.99,
        "fifty_two_week_low": 51.55,
    },
    "PEP": {
        "symbol": "PEP",
        "name": "PepsiCo Inc",
        "sector": "Consumer Defensive",
        "industry": "Beverages",
        "market_capitalization": 240000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 183.41,
        "fifty_two_week_low": 155.83,
    },
    "WMT": {
        "symbol": "WMT",
        "name": "Walmart Inc",
        "sector": "Consumer Defensive",
        "industry": "Discount Stores",
        "market_capitalization": 430000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 165.85,
        "fifty_two_week_low": 137.00,
    },
    "COST": {
        "symbol": "COST",
        "name": "Costco Wholesale Corporation",
        "sector": "Consumer Defensive",
        "industry": "Discount Stores",
        "market_capitalization": 300000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 704.89,
        "fifty_two_week_low": 466.59,
    },

    # REITs (should be excluded)
    "SPG": {
        "symbol": "SPG",
        "name": "Simon Property Group REIT",
        "sector": "Real Estate",
        "industry": "REIT - Retail",
        "market_capitalization": 45000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 151.82,
        "fifty_two_week_low": 101.47,
    },
    "O": {
        "symbol": "O",
        "name": "Realty Income Corporation",
        "sector": "Real Estate",
        "industry": "REIT - Retail",
        "market_capitalization": 40000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 66.92,
        "fifty_two_week_low": 48.37,
    },
    "AMT": {
        "symbol": "AMT",
        "name": "American Tower Corporation",
        "sector": "Real Estate",
        "industry": "REIT - Specialty",
        "market_capitalization": 85000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 227.37,
        "fifty_two_week_low": 154.58,
    },
    "PLD": {
        "symbol": "PLD",
        "name": "Prologis Inc",
        "sector": "Real Estate",
        "industry": "REIT - Industrial",
        "market_capitalization": 115000000000,
        "exchange": "NYSE",
        "fifty_two_week_high": 135.12,
        "fifty_two_week_low": 96.64,
    },
    "EQIX": {
        "symbol": "EQIX",
        "name": "Equinix Inc",
        "sector": "Real Estate",
        "industry": "REIT - Specialty",
        "market_capitalization": 75000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 848.92,
        "fifty_two_week_low": 623.45,
    },

    # Low market cap stocks (should be filtered out by min_market_cap)
    "SMALL1": {
        "symbol": "SMALL1",
        "name": "Small Cap Corp 1",
        "sector": "Technology",
        "industry": "Software",
        "market_capitalization": 500000000,  # $500M - below $10B threshold
        "exchange": "NASDAQ",
        "fifty_two_week_high": 25.00,
        "fifty_two_week_low": 15.00,
    },
    "SMALL2": {
        "symbol": "SMALL2",
        "name": "Small Cap Corp 2",
        "sector": "Healthcare",
        "industry": "Biotechnology",
        "market_capitalization": 2000000000,  # $2B - below $10B threshold
        "exchange": "NASDAQ",
        "fifty_two_week_high": 45.00,
        "fifty_two_week_low": 20.00,
    },
    "SMALL3": {
        "symbol": "SMALL3",
        "name": "Small Cap Corp 3",
        "sector": "Financial Services",
        "industry": "Banks",
        "market_capitalization": 8000000000,  # $8B - below $10B threshold
        "exchange": "NYSE",
        "fifty_two_week_high": 55.00,
        "fifty_two_week_low": 35.00,
    },

    # Penny stocks (should be filtered by min_price)
    "PENNY1": {
        "symbol": "PENNY1",
        "name": "Penny Stock Corp 1",
        "sector": "Technology",
        "industry": "Software",
        "market_capitalization": 15000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 5.00,  # Below $10 threshold
        "fifty_two_week_low": 2.00,
    },
    "PENNY2": {
        "symbol": "PENNY2",
        "name": "Penny Stock Corp 2",
        "sector": "Healthcare",
        "industry": "Biotechnology",
        "market_capitalization": 12000000000,
        "exchange": "NASDAQ",
        "fifty_two_week_high": 8.50,  # Below $10 threshold
        "fifty_two_week_low": 3.00,
    },
}

# Additional listings for small-cap and penny stocks
SAMPLE_LISTINGS.extend([
    {"symbol": "SMALL1", "name": "Small Cap Corp 1", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "SMALL2", "name": "Small Cap Corp 2", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "SMALL3", "name": "Small Cap Corp 3", "exchange": "NYSE", "assetType": "Stock"},
    {"symbol": "PENNY1", "name": "Penny Stock Corp 1", "exchange": "NASDAQ", "assetType": "Stock"},
    {"symbol": "PENNY2", "name": "Penny Stock Corp 2", "exchange": "NASDAQ", "assetType": "Stock"},
])

# Expected symbols after applying all default filters
# Should include: Large-cap stocks that are not ADRs, ETFs, REITs, or SPACs
EXPECTED_FILTERED_SYMBOLS = [
    "AAPL", "ABBV", "AMZN", "BA", "BAC", "C", "CAT", "COST", "GE", "GOOGL",
    "GS", "HON", "JNJ", "JPM", "KO", "LLY", "META", "MRK", "MS", "MSFT",
    "NVDA", "PEP", "PFE", "PG", "TSLA", "UNH", "UPS", "V", "WFC", "WMT",
]

# Symbols by sector for testing filter_by_sector
SYMBOLS_BY_SECTOR = {
    "Technology": ["AAPL", "GOOGL", "META", "MSFT", "NVDA"],
    "Healthcare": ["ABBV", "JNJ", "LLY", "MRK", "PFE", "UNH"],
    "Financial Services": ["BAC", "C", "GS", "JPM", "MS", "V", "WFC"],
    "Industrials": ["BA", "CAT", "GE", "HON", "UPS"],
    "Consumer Defensive": ["COST", "KO", "PEP", "PG", "WMT"],
    "Consumer Cyclical": ["AMZN", "TSLA"],
}
