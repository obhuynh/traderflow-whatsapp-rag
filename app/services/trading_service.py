import yfinance as yf
import pandas as pd

def get_asset_type(symbol: str) -> str:
    """Determines the asset type based on its symbol format."""
    symbol = symbol.upper()
    if "=X" in symbol: # This now correctly handles spot forex like EURUSD=X if you add them
        return "forex"
    if "=F" in symbol: # Added for futures like GC=F, CL=F
        return "futures"
    if "-" in symbol:
        return "crypto"
    return "stock"

def generate_long_term_signal_sma(symbol: str) -> str:
    """Generates a signal based on a 20/50 Day Simple Moving Average crossover."""
    data = yf.download(symbol, period="70d", interval="1d")
    if data.empty:
        return f"Could not retrieve data for the asset '{symbol}'."

    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    if len(data) < 2:
        return f"Not enough data to generate a signal for {symbol}."

    last_close_price = float(data['Close'].iloc[-1])
    
    is_golden_cross = (data['SMA20'].iloc[-2] < data['SMA50'].iloc[-2]) and (data['SMA20'].iloc[-1] > data['SMA50'].iloc[-1])
    is_death_cross = (data['SMA20'].iloc[-2] > data['SMA50'].iloc[-2]) and (data['SMA20'].iloc[-1] < data['SMA50'].iloc[-1])
    
    signal = "NEUTRAL"
    if is_golden_cross:
        signal = "BUY (Golden Cross)"
    elif is_death_cross:
        signal = "SELL (Death Cross)"

    # Add clarification for futures contracts
    display_symbol = symbol
    if symbol == "GC=F":
        display_symbol = "Gold Futures (GC=F)"
    elif symbol == "CL=F":
        display_symbol = "Crude Oil Futures (CL=F)"
    # Add more if other futures symbols are used

    return (
        f"📈 Long-Term Signal for {display_symbol}:\n\n"
        f"**Signal: {signal}**\n"
        f"Strategy: 20/50 Day SMA Crossover\n"
        f"Last Close Price: {last_close_price:.4f}\n"
    )

def generate_short_term_signal_ema_rsi(symbol: str) -> str:
    """
    Generates a scalping/short-term signal based on EMA crossover, confirmed by RSI,
    with dynamic Stop Loss and Target Point calculated using ATR.
    """
    data = yf.download(symbol, period="14d", interval="1h")
    if data.empty:
        return f"Could not retrieve hourly data for '{symbol}'. This symbol may not support hourly history."

    data['EMA9'] = data['Close'].ewm(span=9, adjust=False).mean()
    data['EMA21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()

    data.dropna(inplace=True)
    if len(data) < 2:
        return f"Not enough data to generate a short-term signal for {symbol}."

    is_bullish_crossover = (data['EMA9'].iloc[-2] < data['EMA21'].iloc[-2]) and (data['EMA9'].iloc[-1] > data['EMA21'].iloc[-1])
    is_bearish_crossover = (data['EMA9'].iloc[-2] > data['EMA21'].iloc[-2]) and (data['EMA9'].iloc[-1] < data['EMA21'].iloc[-1])
    
    last_rsi = float(data['RSI'].iloc[-1])
    last_atr = float(data['ATR'].iloc[-1])
    last_low = float(data['Low'].iloc[-1])
    last_high = float(data['High'].iloc[-1])
    last_close = float(data['Close'].iloc[-1])
    
    signal, stop_loss, target_point = "NEUTRAL", 0.0, 0.0
    
    if is_bullish_crossover and last_rsi < 70:
        signal = "BUY"
        stop_loss = last_low - (last_atr * 1.5)
        target_point = last_close + (last_atr * 2.0)
        
    elif is_bearish_crossover and last_rsi > 30:
        signal = "SELL"
        stop_loss = last_high + (last_atr * 1.5)
        target_point = last_close - (last_atr * 2.0)
        
    # Add clarification for futures contracts (also for short-term)
    display_symbol = symbol
    if symbol == "GC=F":
        display_symbol = "Gold Futures (GC=F)"
    elif symbol == "CL=F":
        display_symbol = "Crude Oil Futures (CL=F)"

    response = (
        f"📈 Scalping Signal for {display_symbol} (1-Hour Chart):\n\n" # Updated here
        f"**Signal: {signal}**\n"
        f"Strategy: 9/21 EMA Crossover with RSI/ATR\n"
        f"Last Close Price: {last_close:.4f}\n"
    )

    if signal != "NEUTRAL":
        response += (
            f"Suggested Stop Loss: {stop_loss:.4f}\n"
            f"Suggested Target Point: {target_point:.4f}"
        )
    else:
        response += "No clear buy or sell signal detected on the hourly chart."
        
    return response

def get_trading_signal(symbol: str, timeframe: str = "auto") -> str:
    """
    Main function. Detects asset type and requested timeframe to call the
    appropriate strategy function.
    """
    try:
        asset_type = get_asset_type(symbol)
        print(f"Detected asset type: {asset_type}, Requested timeframe: {timeframe}")

        use_short_term_strategy = False
        if timeframe == "short":
            use_short_term_strategy = True
        elif timeframe == "long":
            use_short_term_strategy = False
        else: # Auto-detection based on asset type
            if asset_type in ["forex", "crypto", "futures"]: # Include futures for short-term by default
                use_short_term_strategy = True
            else: # Defaults to long for stocks
                use_short_term_strategy = False
        
        if use_short_term_strategy:
            print("Routing to Short-Term EMA/RSI strategy.")
            return generate_short_term_signal_ema_rsi(symbol)
        else:
            print("Routing to Long-Term SMA strategy.")
            return generate_long_term_signal_sma(symbol)

    except Exception as e:
        print(f"An error occurred in get_trading_signal for symbol {symbol}: {e}")
        if "No timezone information" in str(e):
             return f"Error: Could not retrieve data for {symbol}. It might not be a valid symbol or historical data is unavailable."
        return f"An error occurred while analyzing {symbol}."
    
