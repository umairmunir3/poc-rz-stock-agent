"""Sample Alpha Vantage API responses for testing."""

# Daily OHLCV response for AAPL
DAILY_OHLCV_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Time Series with Splits and Dividend Events",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2024-01-15",
        "4. Output Size": "Compact",
        "4. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2024-01-15": {
            "1. open": "186.0900",
            "2. high": "188.8500",
            "3. low": "185.8300",
            "4. close": "188.6300",
            "5. adjusted close": "188.6300",
            "6. volume": "65076641",
        },
        "2024-01-12": {
            "1. open": "186.5400",
            "2. high": "187.0500",
            "3. low": "185.1900",
            "4. close": "185.5900",
            "5. adjusted close": "185.5900",
            "6. volume": "40477782",
        },
        "2024-01-11": {
            "1. open": "186.5400",
            "2. high": "187.0500",
            "3. low": "185.1900",
            "4. close": "185.1800",
            "5. adjusted close": "185.1800",
            "6. volume": "49128408",
        },
    },
}

# Daily response with missing data
DAILY_OHLCV_WITH_GAPS = {
    "Meta Data": {
        "1. Information": "Daily Time Series with Splits and Dividend Events",
        "2. Symbol": "TEST",
        "3. Last Refreshed": "2024-01-15",
        "4. Output Size": "Compact",
        "4. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2024-01-15": {
            "1. open": "100.00",
            "2. high": "105.00",
            "3. low": "99.00",
            "4. close": "104.00",
            "5. adjusted close": "104.00",
            "6. volume": "1000000",
        },
        "2024-01-14": {
            "1. open": "98.00",
            "2. high": "101.00",
            "3. low": "97.50",
            "4. close": "100.00",
            "5. adjusted close": "100.00",
            "6. volume": "800000",
        },
    },
}

# Intraday OHLCV response
INTRADAY_RESPONSE = {
    "Meta Data": {
        "1. Information": "Intraday (5min) open, high, low, close prices and volume",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2024-01-15 16:00:00",
        "4. Interval": "5min",
        "5. Output Size": "Compact",
        "6. Time Zone": "US/Eastern",
    },
    "Time Series (5min)": {
        "2024-01-15 16:00:00": {
            "1. open": "188.5000",
            "2. high": "188.6500",
            "3. low": "188.4000",
            "4. close": "188.6300",
            "5. volume": "1234567",
        },
        "2024-01-15 15:55:00": {
            "1. open": "188.3000",
            "2. high": "188.5500",
            "3. low": "188.2500",
            "4. close": "188.5000",
            "5. volume": "987654",
        },
    },
}

# Company overview response
COMPANY_OVERVIEW_RESPONSE = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "Description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Country": "USA",
    "Sector": "TECHNOLOGY",
    "Industry": "CONSUMER ELECTRONICS",
    "MarketCapitalization": "2950000000000",
    "PERatio": "29.50",
    "EPS": "6.39",
    "DividendYield": "0.0051",
    "52WeekHigh": "199.62",
    "52WeekLow": "124.17",
    "50DayMovingAverage": "185.23",
    "200DayMovingAverage": "177.45",
    "Beta": "1.25",
}

# Company overview with None values
COMPANY_OVERVIEW_WITH_NULLS = {
    "Symbol": "STARTUP",
    "Name": "Startup Corp",
    "Description": "A new startup company",
    "Exchange": "NASDAQ",
    "Currency": "USD",
    "Country": "USA",
    "Sector": "TECHNOLOGY",
    "Industry": "SOFTWARE",
    "MarketCapitalization": "None",
    "PERatio": "-",
    "EPS": "None",
    "DividendYield": "None",
    "52WeekHigh": "50.00",
    "52WeekLow": "10.00",
    "50DayMovingAverage": "None",
    "200DayMovingAverage": "None",
    "Beta": "None",
}

# Options chain response
OPTIONS_CHAIN_RESPONSE = {
    "data": [
        {
            "contractID": "AAPL240119C00180000",
            "symbol": "AAPL",
            "expiration": "2024-01-19",
            "strike": "180.00",
            "type": "call",
            "last": "9.25",
            "bid": "9.15",
            "ask": "9.35",
            "volume": "12345",
            "open_interest": "54321",
            "implied_volatility": "0.25",
            "delta": "0.65",
            "gamma": "0.03",
            "theta": "-0.15",
            "vega": "0.20",
        },
        {
            "contractID": "AAPL240119C00185000",
            "symbol": "AAPL",
            "expiration": "2024-01-19",
            "strike": "185.00",
            "type": "call",
            "last": "5.50",
            "bid": "5.40",
            "ask": "5.60",
            "volume": "8765",
            "open_interest": "32100",
            "implied_volatility": "0.23",
            "delta": "0.50",
            "gamma": "0.04",
            "theta": "-0.12",
            "vega": "0.18",
        },
        {
            "contractID": "AAPL240119P00180000",
            "symbol": "AAPL",
            "expiration": "2024-01-19",
            "strike": "180.00",
            "type": "put",
            "last": "1.25",
            "bid": "1.20",
            "ask": "1.30",
            "volume": "5432",
            "open_interest": "21000",
            "implied_volatility": "0.24",
            "delta": "-0.35",
            "gamma": "0.03",
            "theta": "-0.10",
            "vega": "0.15",
        },
    ],
}

# Empty options chain (no options available)
EMPTY_OPTIONS_RESPONSE = {"symbol": "NOOPT", "data": []}

# Invalid symbol error response
INVALID_SYMBOL_RESPONSE = {
    "Error Message": "Invalid API call. Please retry or visit the documentation (https://www.alphavantage.co/documentation/) for TIME_SERIES_DAILY_ADJUSTED."
}

# Rate limit response
RATE_LIMIT_RESPONSE = {
    "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day."
}

# Generic API error response
API_ERROR_RESPONSE = {"Error Message": "Something went wrong with the request."}

# Information message (often indicates subscription issues)
INFORMATION_RESPONSE = {
    "Information": "Thank you for using Alpha Vantage! Please visit https://www.alphavantage.co/premium/ if you need higher API call volume."
}
