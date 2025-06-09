import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

# --- _download_data function HAS BEEN REMOVED FROM HERE ---


def generate_long_term_signal_sma(symbol: str, all_symbols_data: pd.DataFrame) -> dict:
    """Generates a signal based on a 20/50 Day Simple Moving Average crossover."""
    normalized_symbol = symbol.upper()
    if normalized_symbol == "SP500":
        normalized_symbol = "^GSPC"
    elif normalized_symbol == "NASDAQ":
        normalized_symbol = "^IXIC"
    elif normalized_symbol == "GOLD" or normalized_symbol == "XAUUSD":
        normalized_symbol = "GC=F"
    elif normalized_symbol == "OIL" or normalized_symbol == "CRUDE OIL":
        normalized_symbol = "CL=F"
    elif normalized_symbol == "SILVER" or normalized_symbol == "XAGUSD":
        normalized_symbol = "SI=F"


    logger.info(f"Processing long-term data for {normalized_symbol}")
    data = all_symbols_data 


    if data.empty or len(data) < 50:
        logger.warning(f"Insufficient data for {normalized_symbol} (long-term). Data points: {len(data)}")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Could not retrieve enough historical data for {symbol} to generate a long-term signal. Ensure it's a valid symbol and has sufficient history.",
            "key_levels": {},
            "advice": "No signal due to insufficient data.",
            "signal_direction": "NO_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    if isinstance(data['Close'], pd.DataFrame):
        data['Close'] = data['Close'].iloc[:, 0].squeeze()
        logger.warning(f"Forced 'Close' column to Series for {normalized_symbol} in long-term strategy.")

    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)

    if len(data) < 2:
        logger.warning(f"Not enough processed data for {normalized_symbol} after SMA calculation (long-term). Data points: {len(data)}")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Not enough processed data to generate a signal for {symbol} after SMA calculation.",
            "key_levels": {},
            "advice": "No signal due to insufficient data after processing.",
            "signal_direction": "NO_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    last_close_price = safe_float_conversion(data['Close'].iloc[-1])
    sma20_current = safe_float_conversion(data['SMA20'].iloc[-1])
    sma50_current = safe_float_conversion(data['SMA50'].iloc[-1])
    sma20_prev = safe_float_conversion(data['SMA20'].iloc[-2])
    sma50_prev = safe_float_conversion(data['SMA50'].iloc[-2])


    if any(pd.isna(x) for x in [last_close_price, sma20_current, sma50_current, sma20_prev, sma50_prev]):
        logger.warning(f"Missing SMA values for {normalized_symbol} (long-term). Data might be incomplete after calculation.")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Could not calculate SMA values for {symbol}. Data might be incomplete.",
            "key_levels": {},
            "advice": "No signal due to missing SMA values.",
            "signal_direction": "INCOMPLETE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    signal_direction = "NEUTRAL"
    summary_text = ""
    advice_text = ""
    target_zone = "N/A"
    key_levels = {}
    stop_loss_value = "N/A"
    take_profit_value = "N/A"
    risk_reward_ratio = "N/A"

    is_golden_cross = (sma20_prev < sma50_prev) and (sma20_current > sma50_current)
    is_death_cross = (sma20_prev > sma50_prev) and (sma20_current < sma50_current)

    if is_golden_cross:
        signal_direction = "BUY"
        summary_text = "A Golden Cross has occurred (20-day SMA crossed above 50-day SMA), indicating potential long-term bullish momentum."
        target_zone = f"{last_close_price * 1.05:.2f} - {last_close_price * 1.15:.2f}"
        advice_text = f"Consider long positions. Monitor for continued upward momentum. Initial support around {sma50_current:.2f}. No specific Stop Loss/Take Profit for this long-term strategy."
        key_levels = {"support": [f"{sma50_current:.2f}", f"{sma20_current:.2f}"]}

    elif is_death_cross:
        signal_direction = "SELL"
        summary_text = "A Death Cross has occurred (20-day SMA crossed below 50-day SMA), indicating potential long-term bearish momentum."
        target_zone = f"{last_close_price * 0.85:.2f} - {last_close_price * 0.95:.2f}"
        advice_text = f"Consider short positions. Monitor for continued downward pressure. Initial resistance around {sma50_current:.2f}. No specific Stop Loss/Take Profit for this long-term strategy."
        key_levels = {"resistance": [f"{sma50_current:.2f}", f"{sma20_current:.2f}"]}
    else:
        if sma20_current > sma50_current:
            signal_direction = "HOLD_UPTREND"
            summary_text = "The asset is in an uptrend (20-day SMA is above 50-day SMA)."
            advice_text = "Consider holding long positions. Monitor for any signs of reversal."
            key_levels = {"support": [f"{sma50_current:.2f}", f"{sma20_current:.2f}"]}
        elif sma20_current < sma50_current:
            signal_direction = "HOLD_DOWNTREND"
            summary_text = "The asset is in a downtrend (20-day SMA is below 50-day SMA)."
            advice_text = "Consider holding short positions or avoiding longs. Monitor for any signs of reversal."
            key_levels = {"resistance": [f"{sma50_current:.2f}", f"{sma20_current:.2f}"]}
        else:
            summary_text = "No clear long-term trend or crossover detected. Market may be consolidating."
            advice_text = "Markets are consolidating or lack clear direction. Wait for clearer signals."

    return {
        "name": symbol,
        "category": get_asset_type(symbol),
        "target_zone": target_zone,
        "summary": summary_text,
        "key_levels": key_levels,
        "advice": advice_text,
        "signal_direction": signal_direction,
        "last_close_price": f"{last_close_price:.4f}" if not pd.isna(last_close_price) else "N/A",
        "stop_loss": stop_loss_value,
        "take_profit": take_profit_value,
        "risk_reward_ratio": risk_reward_ratio
    }


def generate_short_term_signal_ema_rsi(symbol: str, all_symbols_data: pd.DataFrame) -> dict:
    """
    Generates a scalping/short-term signal based on EMA crossover, confirmed by RSI,
    with dynamic Stop Loss and Target Point calculated using ATR.
    Ensures R:R >= 1:1.
    """
    normalized_symbol = symbol.upper()
    if normalized_symbol == "SP500":
        normalized_symbol = "^GSPC"
    elif normalized_symbol == "NASDAQ":
        normalized_symbol = "^IXIC"
    elif normalized_symbol == "GOLD" or normalized_symbol == "XAUUSD":
        normalized_symbol = "GC=F"
    elif normalized_symbol == "OIL" or normalized_symbol == "CRUDE OIL":
        normalized_symbol = "CL=F"
    elif normalized_symbol == "SILVER" or normalized_symbol == "XAGUSD":
        normalized_symbol = "SI=F"

    logger.info(f"Processing short-term data for {normalized_symbol}")
    data = all_symbols_data # NEW: Data is passed directly from rag_service

    min_data_points = max(21, 14 + 1)
    if data.empty or len(data) < min_data_points:
        logger.warning(f"Insufficient hourly data for {normalized_symbol}. Data points: {len(data)}")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Could not retrieve enough hourly data for {symbol}. This symbol may not support hourly history or has insufficient data points.",
            "key_levels": {},
            "advice": "No signal due to insufficient data.",
            "signal_direction": "NO_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    if isinstance(data['Close'], pd.DataFrame):
        data['Close'] = data['Close'].iloc[:, 0].squeeze()
        logger.warning(f"Forced 'Close' column to Series for {normalized_symbol} in short-term strategy.")

    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()

    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = np.where(loss == 0, gain / 1e-9, gain / loss)
    data['RSI'] = 100 - (100 / (1 + rs))

    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    data.dropna(inplace=True)
    if len(data) < 2:
        logger.warning(f"Not enough processed hourly data for {normalized_symbol} after indicator calculation. Data points: {len(data)}")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Not enough processed hourly data to generate a short-term signal for {symbol}.",
            "key_levels": {},
            "advice": "No signal due to insufficient data after processing.",
            "signal_direction": "NO_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    is_bullish_crossover = False
    is_bearish_crossover = False

    ema9_current = safe_float_conversion(data['EMA9'].iloc[-1])
    ema21_current = safe_float_conversion(data['EMA21'].iloc[-1])
    ema9_prev = safe_float_conversion(data['EMA9'].iloc[-2])
    ema21_prev = safe_float_conversion(data['EMA21'].iloc[-2])

    if not any(pd.isna(x) for x in [ema9_current, ema21_current, ema9_prev, ema21_prev]):
        is_bullish_crossover = (ema9_prev < ema21_prev) and (ema9_current > ema21_current)
        is_bearish_crossover = (ema9_prev > ema21_prev) and (ema9_current < ema21_prev)
    else:
        logger.warning(f"EMA values are NaN for {normalized_symbol}. Crossover detection skipped.")


    last_rsi = safe_float_conversion(data['RSI'].iloc[-1])
    last_atr = safe_float_conversion(data['ATR'].iloc[-1])
    last_low = safe_float_conversion(data['Low'].iloc[-1])
    last_high = safe_float_conversion(data['High'].iloc[-1])
    last_close = safe_float_conversion(data['Close'].iloc[-1])

    if any(pd.isna(x) for x in [last_close, last_atr, last_rsi]):
        logger.warning(f"Crucial indicator values missing/NaN for {normalized_symbol} (short-term) after calculation.")
        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": f"Crucial indicator values (Close, ATR, RSI) are missing for {symbol}. Data might be incomplete.",
            "key_levels": {},
            "advice": "No signal due to missing indicator values.",
            "signal_direction": "INCOMPLETE_DATA",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }

    signal_direction = "NEUTRAL"
    summary_text = ""
    advice_text = ""
    target_zone = "N/A"
    key_levels = {}
    stop_loss_value_num = np.nan
    take_profit_value_num = np.nan
    risk_reward_ratio_num = np.nan

    min_atr_threshold = last_close * 0.0001

    if pd.isna(last_atr) or last_atr < min_atr_threshold:
        signal_direction = "NEUTRAL"
        summary_text = "ATR is too low or invalid to calculate reliable short-term SL/TP. Market might be very flat. No actionable signal."
        advice_text = "Consider waiting for more volatility or using a different strategy."
    elif is_bullish_crossover and 30 < last_rsi < 70:
        signal_direction = "BUY"
        stop_loss_value_num = last_low - (last_atr * 1.5)

        if stop_loss_value_num <= 0:
            stop_loss_value_num = last_close * 0.99
            logger.warning(f"Adjusted SL to 0.99*close for {normalized_symbol} as calculated SL was non-positive.")

        risk = last_close - stop_loss_value_num
        if risk <= 0:
            signal_direction = "NEUTRAL"
            summary_text = "Bullish crossover detected but risk calculation is invalid (SL too high or same as price). No signal."
        else:
            initial_take_profit = last_close + (last_atr * 2.0)
            reward = initial_take_profit - last_close

            if reward / risk < 1.0:
                take_profit_value_num = last_close + risk
                risk_reward_ratio_num = 1.0
                summary_text = f"A bullish EMA crossover was detected (EMA9 > EMA21) with RSI {last_rsi:.2f}. Adjusted Take Profit for R:R >= 1:1."
            else:
                take_profit_value_num = initial_take_profit
                risk_reward_ratio_num = reward / risk
                summary_text = f"A bullish EMA crossover was detected (EMA9 > EMA21) with RSI {last_rsi:.2f}."

            target_zone = f"{take_profit_value_num:.4f}"
            advice_text = f"Consider entering a long position. Suggested Stop Loss: {stop_loss_value_num:.4f}, Suggested Take Profit: {take_profit_value_num:.4f}. Risk:Reward = {risk_reward_ratio_num:.2f}:1."
            key_levels = {"support": [f"{ema21_current:.4f}", f"{ema9_current:.4f}"]}


    elif is_bearish_crossover and 30 < last_rsi < 70:
        signal_direction = "SELL"
        stop_loss_value_num = last_high + (last_atr * 1.5)

        risk = stop_loss_value_num - last_close
        if risk <= 0:
            signal_direction = "NEUTRAL"
            summary_text = "Bearish crossover detected but risk calculation is invalid (SL too low or same as price). No signal."
        else:
            initial_take_profit = last_close - (last_atr * 2.0)

            if initial_take_profit <= 0:
                initial_take_profit = last_close * 0.01

            reward = last_close - initial_take_profit

            if reward / risk < 1.0:
                take_profit_value_num = last_close - risk
                risk_reward_ratio_num = 1.0
                summary_text = f"A bearish EMA crossover was detected (EMA9 < EMA21) with RSI {last_rsi:.2f}. Adjusted Take Profit for R:R >= 1:1."
            else:
                take_profit_value_num = initial_take_profit
                risk_reward_ratio_num = reward / risk
                summary_text = f"A bearish EMA crossover was detected (EMA9 < EMA21) with RSI {last_rsi:.2f}."

            target_zone = f"{take_profit_value_num:.4f}"
            advice_text = f"Consider entering a short position. Suggested Stop Loss: {stop_loss_value_num:.4f}, Suggested Take Profit: {take_profit_value_num:.4f}. Risk:Reward = {risk_reward_ratio_num:.2f}:1."
            key_levels = {"resistance": [f"{ema21_current:.4f}", f"{ema9_current:.4f}"]}
    else:
        summary_text = "No clear short-term EMA crossover signal with RSI confirmation detected."
        advice_text = "The market lacks a strong short-term directional bias based on EMA/RSI. Consider waiting for clearer signals."

    stop_loss_str = f"{stop_loss_value_num:.4f}" if not pd.isna(stop_loss_value_num) else "N/A"
    take_profit_str = f"{take_profit_value_num:.4f}" if not pd.isna(take_profit_value_num) else "N/A"
    risk_reward_str = f"{risk_reward_ratio_num:.2f}:1" if not pd.isna(risk_reward_ratio_num) else "N/A"

    return {
        "name": symbol,
        "category": get_asset_type(symbol),
        "target_zone": target_zone,
        "summary": summary_text,
        "key_levels": key_levels,
        "advice": advice_text,
        "signal_direction": signal_direction,
        "last_close_price": f"{last_close:.4f}" if not pd.isna(last_close) else "N/A",
        "stop_loss": stop_loss_str,
        "take_profit": take_profit_str,
        "risk_reward_ratio": risk_reward_str
    }


def get_trading_signal(symbol: str, timeframe: str = "auto", all_symbols_data_map: Dict[str, pd.DataFrame] = None) -> dict:
    """
    Main function. Detects asset type and requested timeframe to call the
    appropriate strategy function. Returns a dictionary with signal details.
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
        if normalized_symbol == "SP500":
            normalized_symbol = "^GSPC"
        elif normalized_symbol == "NASDAQ":
            normalized_symbol = "^IXIC"
        elif normalized_symbol == "GOLD" or normalized_symbol == "XAUUSD":
            normalized_symbol = "GC=F"
        elif normalized_symbol == "OIL" or normalized_symbol == "CRUDE OIL":
            normalized_symbol = "CL=F"
        elif normalized_symbol == "SILVER" or normalized_symbol == "XAGUSD":
            normalized_symbol = "SI=F"

        asset_type = get_asset_type(normalized_symbol)
        logger.info(f"Detected asset type: {asset_type}, Requested timeframe: {timeframe}")

        use_short_term_strategy = False
        use_long_term_strategy = False

        if timeframe == "short":
            use_short_term_strategy = True
        elif timeframe == "long":
            use_long_term_strategy = True
        else:
            if asset_type in ["forex", "crypto", "futures", "commodity"]:
                use_short_term_strategy = True
            else:
                use_long_term_strategy = True


        if use_short_term_strategy:
            logger.info(f"Routing to Short-Term EMA/RSI strategy for {symbol}.")
            return generate_short_term_signal_ema_rsi(symbol, all_symbols_data_map.get(normalized_symbol, pd.DataFrame()))
        elif use_long_term_strategy:
            logger.info(f"Routing to Long-Term SMA strategy for {symbol}.")
            return generate_long_term_signal_sma(symbol, all_symbols_data_map.get(normalized_symbol, pd.DataFrame()))
        else:
            logger.error(f"No trading strategy selected for {symbol} with timeframe {timeframe}.")
            return {
                "name": symbol,
                "category": get_asset_type(symbol),
                "target_zone": "N/A",
                "summary": "No specific trading strategy could be determined for this asset/timeframe.",
                "key_levels": {},
                "advice": "Please specify a valid timeframe or symbol.",
                "signal_direction": "ERROR",
                "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
            }

    except Exception as e:
        logger.exception(f"🚨 An error occurred in get_trading_signal for symbol {symbol}: {e}")

        error_message = f"An error occurred while analyzing {symbol}."
        if "No timezone information" in str(e) or "empty DataFrame" in str(e) or "YFPricesMissingError" in str(e):
             error_message = f"Error: Could not retrieve sufficient data for {symbol}. It might not be a valid symbol on Yahoo Finance, or historical data is unavailable for the requested period/interval."
        elif "Cannot get prices because the date is invalid" in str(e):
             error_message = f"Error: Invalid date range for {symbol}. Data might be too old or too recent."
        elif "IndexError: single positional indexer is out-of-bounds" in str(e) or "Series has multiple items or is empty when expecting single scalar" in str(e) or "Multi-column data for" in str(e):
            error_message = f"Error: Not enough data points available for {symbol} or data format issue during processing. This might happen if Yahoo Finance provides very limited recent history or if the asset has low liquidity."
        else:
            error_message = f"An unexpected error occurred for {symbol}: {e}"


        return {
            "name": symbol,
            "category": get_asset_type(symbol),
            "target_zone": "N/A",
            "summary": error_message,
            "key_levels": {},
            "advice": "Data retrieval or analysis failed. Please check the symbol or try again later.",
            "signal_direction": "ERROR",
            "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
        }