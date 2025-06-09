# app/services/trading_service.py

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Helper function for robust float conversion ---
def safe_float_conversion(value):
    """
    Safely converts a value to a float, handling potential NaN/None values and different pandas types.
    """
    scalar_value = None

    if isinstance(value, pd.Series):
        if value.empty:
            return np.nan
        try:
            val_from_series = value.iloc[0]
            if isinstance(val_from_series, (np.float64, np.int64, bool)):
                scalar_value = val_from_series.item()
            else:
                scalar_value = val_from_series
        except IndexError:
            logger.warning(f"IndexError when extracting scalar from Series: {value}. Returning NaN.")
            return np.nan
        except Exception as e:
            logger.warning(f"Unexpected error extracting scalar from Series {value}: {e}. Returning NaN.")
            return np.nan
    elif isinstance(value, pd.DataFrame):
        if value.empty:
            return np.nan
        try:
            scalar_value = value.iloc[0, 0].item()
        except (IndexError, ValueError):
            logger.warning(f"Failed to extract scalar from DataFrame: {value}. Returning NaN.")
            return np.nan
    elif isinstance(value, np.ndarray):
        if value.size == 1:
            scalar_value = value.item()
        else:
            logger.warning(f"NumPy array has multiple items: {value}. Returning NaN.")
            return np.nan
    else:
        scalar_value = value

    if pd.isna(scalar_value):
        return np.nan
    try:
        return float(scalar_value)
    except (TypeError, ValueError):
        logger.warning(f"Could not convert '{scalar_value}' (type: {type(scalar_value)}) to float. Returning NaN.")
        return np.nan


def get_asset_type(symbol: str) -> str:
    """Determines the asset type based on its symbol format."""
    symbol = symbol.upper()
    if "=X" in symbol:
        return "forex"
    if "=F" in symbol:
        return "futures"
    if "-" in symbol:
        return "crypto"
    if symbol in ["^GSPC", "SP500", "^IXIC", "NASDAQ", "^DJI", "DOWJONES"]:
        return "index"
    if symbol in ["GC", "CL", "SI", "XAUUSD", "XAGUSD"]:
        return "commodity"
    return "stock"

# _download_data function has been removed from here as it's now in rag_service.py


def generate_long_term_signal_sma(symbol: str, all_symbols_data: pd.DataFrame) -> dict:
    """
    SIMPLIFIED: Now only retrieves current price. Technical analysis is removed.
    """
    normalized_symbol = symbol.upper()
    # Normalize symbols for Yahoo Finance
    if normalized_symbol == "SP500": normalized_symbol = "^GSPC"
    elif normalized_symbol == "NASDAQ": normalized_symbol = "^IXIC"
    elif normalized_symbol == "GOLD" or normalized_symbol == "XAUUSD": normalized_symbol = "GC=F"
    elif normalized_symbol == "OIL" or normalized_symbol == "CRUDE OIL": normalized_symbol = "CL=F"
    elif normalized_symbol == "SILVER" or normalized_symbol == "XAGUSD": normalized_symbol = "SI=F"

    logger.info(f"Retrieving current price for {normalized_symbol} (long-term context).")
    data = all_symbols_data # Data is passed directly from rag_service


    if data.empty:
        logger.warning(f"No data for {normalized_symbol} (long-term price retrieval). Data points: {len(data)}")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Could not retrieve current price for {symbol}.",
            "key_levels": {},
            "advice": "Current price unavailable.",
            "signal_direction": "NO_PRICE_DATA", # New signal type
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    # Ensure 'Close' column is a Series, not a DataFrame
    if isinstance(data.get('Close'), pd.DataFrame): # Use .get for safer access
        data['Close'] = data['Close'].iloc[:, 0].squeeze()
        logger.warning(f"Forced 'Close' column to Series for {normalized_symbol}.")
    elif data.get('Close') is None: # Handle if 'Close' column is missing
        logger.warning(f"Close price column missing for {normalized_symbol}.")
        return {
            "name": symbol, "category": get_asset_type(symbol), "target_zone": "N/A",
            "summary": f"Close price data missing for {symbol}.", "key_levels": {},
            "advice": "Current price unavailable.", "signal_direction": "NO_PRICE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    last_close_price = safe_float_conversion(data['Close'].iloc[-1])

    if pd.isna(last_close_price):
        logger.warning(f"Current price is NaN for {normalized_symbol}.")
        return {
            "name": symbol, "category": get_asset_type(symbol), "target_zone": "N/A",
            "summary": f"Current price is not a valid number for {symbol}.", "key_levels": {},
            "advice": "Current price unavailable.", "signal_direction": "NO_PRICE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    return {
        "name": symbol,
        "category": get_asset_type(symbol),
        "target_zone": "N/A",
        "summary": "Current price retrieved. No technical analysis or signals generated by this service.",
        "key_levels": {},
        "advice": "This service provides current prices only. No technical analysis performed.",
        "signal_direction": "CURRENT_PRICE_ONLY", # New signal type
        "last_close_price": f"{last_close_price:.4f}",
        "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
    }


def generate_short_term_signal_ema_rsi(symbol: str, all_symbols_data: pd.DataFrame) -> dict:
    """
    SIMPLIFIED: Now only retrieves current price. Technical analysis is removed.
    """
    normalized_symbol = symbol.upper()
    # Normalize symbols for Yahoo Finance
    if normalized_symbol == "SP500": normalized_symbol = "^GSPC"
    elif normalized_symbol == "NASDAQ": normalized_symbol = "^IXIC"
    elif normalized_symbol == "GOLD" or normalized_symbol == "XAUUSD": normalized_symbol = "GC=F"
    elif normalized_symbol == "OIL" or normalized_symbol == "CRUDE OIL": normalized_symbol = "CL=F"
    elif normalized_symbol == "SILVER" or normalized_symbol == "XAGUSD": normalized_symbol = "SI=F"

    logger.info(f"Retrieving current price for {normalized_symbol} (short-term context).")
    data = all_symbols_data # Data is passed directly from rag_service

    if data.empty:
        logger.warning(f"No data for {normalized_symbol} (short-term price retrieval). Data points: {len(data)}")
        return {
            "name": symbol, "category": get_asset_type(symbol), "target_zone": "N/A",
            "summary": f"Could not retrieve current price for {symbol}.", "key_levels": {},
            "advice": "Current price unavailable.", "signal_direction": "NO_PRICE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    # Ensure 'Close' column is a Series, not a DataFrame
    if isinstance(data.get('Close'), pd.DataFrame): # Use .get for safer access
        data['Close'] = data['Close'].iloc[:, 0].squeeze()
        logger.warning(f"Forced 'Close' column to Series for {normalized_symbol}.")
    elif data.get('Close') is None: # Handle if 'Close' column is missing
        logger.warning(f"Close price column missing for {normalized_symbol}.")
        return {
            "name": symbol, "category": get_asset_type(symbol), "target_zone": "N/A",
            "summary": f"Close price data missing for {symbol}.", "key_levels": {},
            "advice": "Current price unavailable.", "signal_direction": "NO_PRICE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    last_close = safe_float_conversion(data['Close'].iloc[-1])

    if pd.isna(last_close):
        logger.warning(f"Current price is NaN for {normalized_symbol}.")
        return {
            "name": symbol, "category": get_asset_type(symbol), "target_zone": "N/A",
            "summary": f"Current price is not a valid number for {symbol}.", "key_levels": {},
            "advice": "Current price unavailable.", "signal_direction": "NO_PRICE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    return {
        "name": symbol,
        "category": get_asset_type(symbol),
        "target_zone": "N/A",
        "summary": "Current price retrieved. No technical analysis or signals generated by this service.",
        "key_levels": {},
        "advice": "This service provides current prices only. No technical analysis performed.",
        "signal_direction": "CURRENT_PRICE_ONLY", # New signal type
        "last_close_price": f"{last_close:.4f}",
        "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
    }


def get_trading_signal(symbol: str, timeframe: str = "auto", all_symbols_data_map: Dict[str, pd.DataFrame] = None) -> dict:
    """
    Main function. Detects asset type and requested timeframe to call the
    appropriate price retrieval function. Returns a dictionary with price details.
    """
    if all_symbols_data_map is None:
        logger.error(f"get_trading_signal called without all_symbols_data_map for {symbol}. This should not happen.")
        return {
            "name": symbol, "category": get_asset_type(symbol), "target_zone": "N/A",
            "summary": "Internal error: Data not pre-fetched correctly.", "key_levels": {},
            "advice": "Please try again later.", "signal_direction": "ERROR",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    try:
        normalized_symbol = symbol.upper()
        if normalized_symbol == "SP500": normalized_symbol = "^GSPC"
        elif normalized_symbol == "NASDAQ": normalized_symbol = "^IXIC"
        elif normalized_symbol == "GOLD" or normalized_symbol == "XAUUSD": normalized_symbol = "GC=F"
        elif normalized_symbol == "OIL" or normalized_symbol == "CRUDE OIL": normalized_symbol = "CL=F"
        elif normalized_symbol == "US30": normalized_symbol = "^DJI" # ADDED US30
        elif normalized_symbol == "ASX200": normalized_symbol = "^AXJO" # ADDED ASX200
        elif normalized_symbol == "FTSE": normalized_symbol = "^FTSE"   # ADDED FTSE
        elif normalized_symbol == "EU50": normalized_symbol = "^STOXX50E" # ADDED EU50

        asset_type = get_asset_type(normalized_symbol)
        logger.info(f"Detected asset type: {asset_type}, Requested timeframe: {timeframe}")

        # No more routing based on strategy, just get the current price
        logger.info(f"Retrieving current price (no TA) for {symbol}.")
        # Use short-term function as it's designed for recent data retrieval
        return generate_short_term_signal_ema_rsi(symbol, all_symbols_data_map.get(normalized_symbol, pd.DataFrame()))

    except Exception as e:
        logger.exception(f"🚨 An error occurred in get_trading_signal for symbol {symbol}: {e}")

        error_message = f"An error occurred while retrieving data for {symbol}."
        if "No timezone information" in str(e) or "empty DataFrame" in str(e) or "YFPricesMissingError" in str(e):
             error_message = f"Error: Could not retrieve sufficient data for {symbol}. It might not be a valid symbol on Yahoo Finance, or historical data is unavailable."
        elif "Cannot get prices because the date is invalid" in str(e):
             error_message = f"Error: Invalid date range for {symbol}. Data might be too old or too recent."
        elif "IndexError: single positional indexer is out-of-bounds" in str(e) or "Series has multiple items or is empty when expecting single scalar" in str(e) or "Multi-column data for" in str(e):
            error_message = f"Error: Not enough data points available for {symbol} or data format issue during processing."
        else:
            error_message = f"An unexpected error occurred for {symbol}: {e}"


        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": error_message,
            "key_levels": {},
            "advice": "Data retrieval failed. Please check the symbol or try again later.",
            "signal_direction": "ERROR",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }