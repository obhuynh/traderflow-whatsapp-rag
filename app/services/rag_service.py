# app/services/rag_service.py

import ollama
import chromadb
from pathlib import Path
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import Dict, Any, List
from datetime import datetime
import traceback
import logging
import pandas as pd
import re
import concurrent.futures
import yfinance as yf # For _download_data, now defined here
import time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- INITIALIZATION ---
from app.core.config import settings
from app.services.trading_service import get_trading_signal # Only get_trading_signal is imported from here

ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
rss_collection = chroma_collection = chroma_client.get_or_create_collection(name="rss_feeds")

# --- KNOWN TICKERS ---
# This dictionary maps common names/phrases to their Yahoo Finance ticker symbols.
# It is used for parsing natural language queries (e.g., "Bitcoin" -> "BTC-USD").
# The Top_10_Products list in user_profile_data should contain the actual Yahoo Finance tickers.
KNOWN_TICKERS = {
    # Crypto
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD", "SOLANA": "SOL-USD",
    "XRP": "XRP-USD", "RIPPLE": "XRP-X",
    "DOGE": "DOGE-USD", "DOGECOIN": "DOGE-USD",
    "ADA": "ADA-USD", "CARDANO": "ADA-USD",
    # Forex
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X", "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X", "NZDUSD": "NZDUSD=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
    # Commodities
    "GOLD": "GC=F", "XAUUSD": "GC=F",
    "SILVER": "SI=F", "XAGUSD": "SI=F",
    "OIL": "CL=F", "CRUDE OIL": "CL=F",
    # Indices
    "SP500": "^GSPC", "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC", "NDX": "^IXIC",
    "DOWJONES": "^DJI", "DOW": "^DJI", "US30": "^DJI", # ADDED US30
    "ASX200": "^AXJO",  # ADDED ASX200
    "DAX": "DAX",       # ADDED DAX
    "FTSE": "^FTSE",     # ADDED FTSE
    "EU50": "^STOXX50E"  # ADDED EU50
}
SORTED_TICKERS = sorted(KNOWN_TICKERS.keys(), key=len, reverse=True)

# --- Helper functions ---

# --- _download_data DEFINITION (MOVED HERE from trading_service.py) ---
def _download_data(symbol: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    """
    Handles yfinance data download with retries and ensures single-column data for a specific symbol.
    Moved here from trading_service.py to resolve ImportError.
    """
    for attempt in range(retries):
        try:
            ticker_data = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )

            if isinstance(ticker_data.columns, pd.MultiIndex):
                if symbol in ticker_data.columns.levels[1]:
                    data = ticker_data.loc[:, (slice(None), symbol)]
                    data.columns = data.columns.droplevel(1)
                elif symbol in ticker_data.columns.levels[0]:
                     data = ticker_data.loc[:, (symbol, slice(None))]
                     data.columns = data.columns.droplevel(0)
                else:
                    logger.warning(f"Multi-column data for {symbol} but symbol not found in columns. Data: {ticker_data.columns}. Returning empty.")
                    return pd.DataFrame()
            else:
                data = ticker_data

            if data.empty:
                logger.warning(f"yf.download returned empty data for {symbol} on attempt {attempt+1}.")
                if attempt < retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return pd.DataFrame()
            return data

        except TypeError as te:
            logger.warning(f"TypeError during yf.download for {symbol} (attempt {attempt+1}): {te}. Retrying...")
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return pd.DataFrame()

        except Exception as e:
            logger.warning(f"Error during yf.download for {symbol} (attempt {attempt+1}): {e}. Retrying...")
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return pd.DataFrame()
    return pd.DataFrame()
# --- END _download_data DEFINITION ---


def clean_news_context(text: str) -> str:
    """Removes unwanted phrases and URLs from the raw news context."""
    if not text:
        return ""
    unwanted_phrases = [
        r"Weekly Fundamental Analysis...", r"Daily Forex Analysis on LiteFinance Blog", r"Read more at LiteFinance",
        r"Source: \S+", r"Please remember that all trading involves risk and may result in loss\. Past performance is not indicative of future results\. Trading decisions should be based on a thorough analysis of market conditions and individual risk tolerance\. This report is for informational purposes only and should not be construed as financial advice\.",
        r"The Commodity Futures Trading Commission \(CFTC\) has released its latest figures on the speculative net positions in gold, revealing a significant rise that reflects growing confidence among investors\. As of \S+ \d{1,2}, \d{4}, gold speculative net positions have climbed to [\d.]+K, up from the previous level of [\d.]+K\. This notable increase indicates a robust resurgence in appetite for gold, often seen as a safe-haven asset in times of economic uncertainty and market volatility\. The uptick suggests that investors might be hedging against potential economic downturns or inflationary pressures that can erode their purchasing power\.The substantial rise in speculative positions showcases the market's response to current global economic trends and potential future shifts\. As gold continues to hold its allure as a protective asset, traders and analysts alike will be keenly observing the implications of this increase in the broader context of the precious metals market and the global economy\.",
        r"I welcome my fellow traders! I have made a price forecast for USCrude, XAUUSD, and EURUSD using a combination of margin zones methodology and technical analysis\. Based on the market analysis, I suggest entry signals for intraday traders\.",
        r"Main scenario for EURUSD: Consider long positions from corrections (?:below|above) the level of [\d.]+ with a target of [\d.]+ – [\d.]+\. A (?:buy|sell) signal: the correction ends and the price resumes (?:declining|rising) from the [\d.]+ level\. Stop Loss: (?:below|above) [\d.]+, Take Profit: [\d.]+ – [\d.]+\.",
        r"Main scenario for Gold \(SI=F\): Once the correction ends, consider long positions (?:above|below) the level of [\d.]+ with a target of [\d.]+ – [\d.]+\. A (?:buy|sell) signal: the price holds (?:above|below) [\d.]+\. Stop Loss: (?:below|above) [\d.]+, Take Profit: [\d.]+ – [\d.]+\.",
        r"Main scenario for \w+ \(?=F\): Once the correction ends, consider (?:long|short) positions (?:above|below) the level of [\d.]+ with a target of [\d.]+ – [\d.]+\. A (?:buy|sell) signal: the price holds (?:above|below) [\d.]+\. Stop Loss: (?:below|above) [\d.]+, Take Profit: [\d.]+ – [\d.]+\.",
        r"Major Takeaways", r"Oil Price Forecast for Today: USCrude Analysis", r"Gold continues to trade in a short-term uptrend\.",
        r"USCrude: Oil price has stalled near the the Target Zone [\d.]+ - [\d.]+\.", r"EURUSD: The euro is falling in a correction within the short-term uptrend\.",
        r"Nasdaq 100: Speculative positions in the Nasdaq 100 have recently seen a significant reduction, suggesting a shift in investor sentiment\. Investors should monitor selective investments like Nvidia while maintaining portfolio diversification across strong performing sectors\. Keeping an eye on upcoming tech sector developments will be crucial for identifying emerging growth opportunities\.",
        r"Gold prices have remained consolidated since the end of April, as trade wars appear to subside and a favorable monetary policy environment takes hold in the US\.",
        r"The Fed's ongoing support for the US GDP has buoyed gold prices, while central banks continue to buy the precious metal\.",
        r"Surging demand for solar panels has propelled silver's momentum heading into the summer\. The precious metal's rally was partly fueled by gold’s failure to hold above the psychologically important \$3,500/oz level\. Long positions targeting \$37 and \$39 remain in play\. Furthermore, ETF holdings have climbed 8% since February, underscoring silver's allure as a defensive asset during market volatility\.",
        r"This information is provided by InstaForex Company\.",
        r"\(SI=F\)", r"\(XAG/USD\)", r"\(X\)", r"\(Futures\)", r"\(Crypto\)", r"\(Index\)",
        r"\(Outlook: Neutral\)", r"\(Outlook: Upward Trend\)",
        r"Signal:\s*\*NEUTRAL\s*\|\s*Current Price:\s*[\d.]+",
        r"Signal:\s*\*UPWARD TREND\s*\|\s*Current Price:\s*\*Not provided\*",
        r"A price forecast for USCrude, XAUUSD, and EURUSD has been made using a combination of margin zones methodology and technical analysis\. The suggestion is for intraday traders\.",
        r"Oil prices have stalled near the Target Zone [\d.]+ - [\d.]+, while gold buyers are attempting to break through the upper Target Zone [\d.]+ - [\d.]+\.",
        r"The euro's short-term uptrend continues, with previous targets for euro longs already reached\.",
        r"International markets continue to fluctuate, and speculative positions are likely to play a crucial role in shaping short-term trading strategies and investment decisions related to the Australian dollar\.",
        r"The XAG/USD rally was fueled in part by gold’s failure to hold above the psychologically important \$[\d,.]+/oz level\.",
        r"Long positions targeting \$[\d.]+ and \$[\d.]+ remain in play\.",
        r"Furthermore, ETF holdings have climbed [\d.]+% since [\w]+, underscoring silver's allure as a defensive asset during market volatility\.",
        r"\$[\d,.]+/oz", r"\$[\d,.]+",
        r"\b(?:target|resistance|support|level)\s*(?:of|at)\s*[\d,.]+", r"\b(?:stop loss|take profit)\s*(?:at|of)?\s*[\d,.]+",
        r"\b(?:trade|price)\s*wars\b", r"\b(?:bullish|bearish|long|short)\s*positions?\s*(?:targeting|remain in play|consider closing)?\s*[\d,.]*(?: and [\d,.]*)?\b",
        r"ETFs?\s*holdings?\s*(?:have)?\s*(?:climbed|reduced)?\s*[\d.]+\%", r"\(SHFE Gold\)",
        r"The S&P 500 is experiencing an uptrend as it surpasses the [\d,.]+ threshold for the first time since [\w]+. This surge was fueled by a robust jobs report and renewed optimism regarding US-China trade discussions\. Despite President Trump's calls for lower interest rates, today's data has dimmed the odds of a Fed rate cut in the near and medium-term\. A significant risk to monitor is the potential collapse of the US budget bill or reinstatement of tariffs\."
    ]
    for phrase in unwanted_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text

def chroma_retriever(query: str) -> str:
    """Retrieve top 3 relevant news docs from ChromaDB and clean them."""
    try:
        results = rss_collection.query(query_texts=[query], n_results=3)
        if results and results['documents'] and results['documents'][0]:
            raw_docs = "\n\n".join(results['documents'][0])
            return clean_news_context(raw_docs)
        return "No relevant news context found."
    except Exception as e:
        logger.error(f"Error retrieving from ChromaDB: {e}")
        return "No relevant news context found."

def format_signal_output_for_llm(signals: List[Dict[str, Any]]) -> str:
    """
    Formats structured signal data into a very flat, simplified string for the LLM to parse.
    Each signal is a single line, making it extremely difficult for the LLM to invent its own structure.
    """
    if not signals:
        return "NO_SIGNALS_GENERATED_BY_TRADING_SERVICE"

    formatted_signals_list = []
    for sig in signals:
        signal_parts = []
        signal_parts.append(f"ASSET:{str(sig.get('name', 'Unknown'))} ({str(sig.get('category', 'General'))})")
        signal_parts.append(f"DIRECTION:{str(sig.get('signal_direction', 'NEUTRAL')).upper()}")
        signal_parts.append(f"PRICE:{str(sig.get('last_close_price', 'N/A'))}")
        signal_parts.append(f"SUMMARY:'{str(sig.get('summary', 'No summary.')).strip()}'")

        # Removed conditional check for BUY/SELL before adding SL/TP/RR
        # Now, these will always be present, and LLM is instructed to only use if relevant
        signal_parts.append(f"SL:{str(sig.get('stop_loss', 'N/A'))}")
        signal_parts.append(f"TP:{str(sig.get('take_profit', 'N/A'))}")
        signal_parts.append(f"RR:{str(sig.get('risk_reward_ratio', 'N/A'))}")
        
        signal_parts.append(f"GENERAL_TARGET_ZONE:'{str(sig.get('target_zone', 'N/A'))}'")
        
        key_levels = sig.get("key_levels", {})
        support_levels_str = "N/A"
        if "support" in key_levels and key_levels["support"]:
            support_levels_str = ", ".join(map(str, key_levels["support"]))
        signal_parts.append(f"SUPPORT:'{support_levels_str}'")

        resistance_levels_str = "N/A"
        if "resistance" in key_levels and key_levels["resistance"]:
            resistance_levels_str = ", ".join(map(str, key_levels["resistance"]))
        signal_parts.append(f"RESISTANCE:'{resistance_levels_str}'")
        
        signal_parts.append(f"ADVICE:'{str(sig.get('advice', 'No advice.')).strip()}'")

        formatted_signals_list.append("; ".join(signal_parts))

    return "\n".join(formatted_signals_list)


def get_system_prompt_str(is_news_query: bool) -> str:
    """
    Defines the LLM's persona and ultra-strict instructions for generating a cohesive, professional report.
    This prompt aims to prevent hallucination, control formatting, and enforce professional tone.
    """
    if is_news_query:
        return (
            "You are 'TraderFlow AI', a highly knowledgeable and professional financial market assistant. "
            "Your main task is to provide a **concise summary of relevant market news and key developments** based ONLY on the provided 'NEWS_CONTEXT'.\n\n"
            "**STRICT INSTRUCTIONS FOR NEWS REPORT GENERATION:**\n"
            "1.  **Persona & Greeting:** Begin with 'Greetings, Trader!' followed by a brief, professional introduction to the news summary. For example: 'Here's a summary of today's key market news and developments:'\n"
            "2.  **News Summary:** Read the 'NEWS_CONTEXT' carefully. Synthesize the most important, specific, and actionable news items into a coherent, flowing paragraph or two. Focus on key developments and their potential market impact. If the news context is very generic or empty, state that no specific market-moving news was found.\n"
            "3.  **No Signals:** **DO NOT provide any trading signals, prices, Stop Loss (SL), Take Profit (TP), Risk:Reward (R:R), or any specific trading parameters.** The user only asked for news.\n"
            "4.  **Conciseness:** Be concise and to the point.\n"
            "5.  **Forbidden Content:** **STRICTLY DO NOT** include any phrases that reveal your internal workings or data sources (e.g., 'based on the analysis from LiteFinance', 'main scenario', 'alternative scenario', 'read full author’s opinion', URLs, hashtags, internal API details, or any verbatim text from 'NEWS_CONTEXT' that resembles a template or news article title). DO NOT use numbered lists or generic category headings. DO NOT include price forecasts or specific trading strategies if not explicitly from TRADING_SIGNAL_DATA.\n"
            "6.  **Concluding Disclaimer:** End the *entire report* with the standard trading risk disclaimer (as markdown text: `Trading involves significant risk...`).\n"
        )
    else: # Existing prompt for trading signal reports
        # --- Adjusted prompt to reflect NO TECHNICAL ANALYSIS ---
        return (
            "You are 'TraderFlow AI', a highly knowledgeable and professional financial market assistant. "
            "Your main task is to provide a **concise market report** based ONLY on the provided data.\n\n"
            "**STRICT INSTRUCTIONS FOR REPORT GENERATION:**\n"
            "1.  **Persona & Greeting:** Begin with 'Greetings, Trader!' followed by a brief, professional market overview summarizing general market sentiment and current prices.\n"
            "2.  **Report Structure (CRITICAL):**\n"
            "    a.  **NO NUMBERING:** DO NOT use numbered lists (e.g., '1.', '2.').\n"
            "    b.  **NO CATEGORY HEADINGS:** DO NOT use generic category headings like 'Forex:', 'Futures:', 'Crypto:', 'Index:'.\n"
            "    c.  **Unified Flow:** Present a single, cohesive report. Do not break it into numbered or categorized sections. Transition smoothly between asset reports.\n"
            "    d.  **Asset Headings:** For each asset, use Markdown `## Asset_Name (Ticker) Current Price Outlook`. This is the ONLY type of heading.\n"
            "3.  **Signal Data Usage (CRITICAL FOR ACCURACY & ANTI-HALLUCINATION):**\n"
            "    a.  **ONLY USE PROVIDED DATA:** You MUST ONLY use information from 'TRADING_SIGNAL_DATA:'. This data is provided as 'KEY:VALUE' pairs separated by semicolons on a single line per asset.\n"
            "    b.  **NO TECHNICAL ANALYSIS SIGNALS:** This service **DOES NOT** provide technical analysis-based BUY/SELL/HOLD signals, nor does it generate Stop Loss (SL), Take Profit (TP), or Risk:Reward (R:R) ratios based on technical indicators. **DO NOT invent any of these parameters.**\n"
            "    c.  **REPORT CURRENT PRICE ONLY:** For each asset, state its Current Price. If the `DIRECTION` is 'NO_PRICE_DATA' or 'ERROR', explain that price data is unavailable or could not be processed, and state that no further analysis can be given.\n"
            "    d.  **NEWS IS FOR CONTEXT ONLY:** DO NOT use 'NEWS_CONTEXT' to create trading parameters or prices. It is for *contextualization only*.\n"
            "4.  **Asset-Specific Report Format:**\n"
            "    a.  **Price & Summary:** For each asset, state its Current Price: 'Current Price: **[PRICE]**'. Then, synthesize the `SUMMARY` and `ADVICE` provided from 'TRADING_SIGNAL_DATA'. Explain that the service focuses on providing current prices and related news insights.\n"
            "    b.  **News Integration:** *Integrate relevant insights from 'NEWS_CONTEXT' seamlessly* into the asset's report. Focus on how news affects this specific asset's outlook. DO NOT create separate 'Related News Insight' sections or direct quotes.\n"
            "    c.  **No Actionable Parameters:** Explicitly state that technical analysis-based actionable parameters (SL, TP, R:R) are not generated by this service.\n"
            "5.  **Comprehensive Coverage:** Report on ALL assets in 'TRADING_SIGNAL_DATA'. If 'TRADING_SIGNAL_DATA' is 'NO_SIGNALS_GENERATED_BY_TRADING_SERVICE', state this clearly as the main body of your report and generate no asset-specific sections.\n"
            "6.  **Forbidden Content:** **STRICTLY DO NOT** use `---`, `➢`, `1.`, `2.`, `Forex:`, `Futures:`, etc. for top-level formatting. DO NOT include phrases that reveal your internal workings or data sources (e.g., 'based on the analysis from LiteFinance', 'main scenario', 'alternative scenario', 'read full author’s opinion', URLs, hashtags, internal API details, or any verbatim text from 'NEWS_CONTEXT' that resembles a template or news article title). \n"
            "7.  **Markdown & Bold:** Use Markdown `##` for asset headings and `**text**` for bolding as specified. Avoid other bolding unless critical.\n"
            "8.  **Concluding Disclaimer:** End the *entire report* with the standard trading risk disclaimer (as markdown text: `Trading involves significant risk...`).\n"
        )


# --- MAIN ORCHESTRATION FUNCTION ---
def get_rag_response(user_prompt: str, user_profile_data: Dict[str, Any] = None) -> dict: # Changed return type to dict
    logger.info(f"Orchestrating response for prompt: '{user_prompt}'")

    if user_profile_data is None:
        user_profile_data = {
            "Top_10_Products": [
                "^AXJO", "DAX", "^FTSE", "^DJI", "^STOXX50E", "EURUSD=X", "GC=F", "BTC-USD" 
            ],
            "Risk_Level": "Medium",
            "User_Sentiment": "Neutral",
            "Broker": "Pepperstone",
            "Trading_Style": "Scalping"
        }

    prompt_upper = user_prompt.upper()
    trading_related_keywords = ["TRADE", "SIGNAL", "BUY", "SELL", "POSITION", "FORECAST", "ANALYZE", "MARKET", "PRICE", "OUTLOOK", "CHART", "INDICATORS", "STRATEGY"]
    
    news_keywords = ["NEWS", "SPECIFIC NEWS", "MARKET NEWS", "LATEST NEWS", "HEADLINES", "DEVELOPMENTS", "READ"]
    is_news_only_query = any(k in prompt_upper for k in news_keywords) and not any(ticker in prompt_upper for ticker in KNOWN_TICKERS) and not any(k in prompt_upper for k in ["TRADE", "SIGNAL", "BUY", "SELL", "POSITION", "FORECAST", "ANALYZE", "PRICE", "OUTLOOK", "CHART", "INDICATORS", "STRATEGY"])

    is_trading_query = any(k in prompt_upper for k in trading_related_keywords) or any(ticker in prompt_upper for ticker in KNOWN_TICKERS)


    signals_raw_data = []
    context_news = "No relevant news context found."

    try:
        # Determine symbols to query more precisely based on whether a specific ticker is found
        symbols_to_query_normalized = []
        found_ticker_in_query = None
        for ticker_phrase in SORTED_TICKERS:
            if ticker_phrase in prompt_upper:
                found_ticker_in_query = ticker_phrase
                break
        
        if is_news_only_query:
            logger.info("Detected as a news-only query. No trading signals will be generated.")
            symbols_to_query_normalized = [] 
        elif found_ticker_in_query:
            symbols_to_query_normalized = [KNOWN_TICKERS[found_ticker_in_query]]
            logger.info(f"Specific ticker '{found_ticker_in_query}' found. Querying only this symbol: {symbols_to_query_normalized}")
        elif is_trading_query:
            symbols_to_query_normalized = user_profile_data.get("Top_10_Products", [])
            logger.info(f"General trading query. Querying Top 10 products: {symbols_to_query_normalized}")
        else:
            logger.info("Not a trading signal query or news-only. No trading signals will be generated.")
            symbols_to_query_normalized = [] 

        # Determine timeframe for data download
        requested_timeframe = "auto" 
        if symbols_to_query_normalized: 
            prompt_lower = user_prompt.lower()
            short_term_keywords = ["short term", "short-term", "scalp", "hourly", "fast", "quick", "today"]
            long_term_keywords = ["long term", "long-term", "swing", "daily", "invest", "next week", "monthly"]
            if any(k in prompt_lower for k in short_term_keywords):
                requested_timeframe = "short"
            elif any(k in prompt_lower for k in long_term_keywords):
                requested_timeframe = "long"
            logger.info(f"Requested timeframe for symbols: {requested_timeframe}")
        
        # Centralized data download logic using _download_data (now defined here)
        all_downloaded_data_map = {} 
        if symbols_to_query_normalized:
            period_str = "14d" if requested_timeframe == "short" else "70d"
            interval_str = "1h" if requested_timeframe == "short" else "1d"
            
            logger.info(f"Initiating sequential download for {symbols_to_query_normalized} (Period: {period_str}, Interval: {interval_str}).")
            
            # --- _download_data DEFINITION (MOVED HERE from trading_service.py) ---
            def _download_data(symbol: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
                """
                Handles yfinance data download with retries and ensures single-column data for a specific symbol.
                Moved here from trading_service.py to resolve ImportError.
                """
                for attempt in range(retries):
                    try:
                        ticker_data = yf.download(
                            symbol,
                            period=period,
                            interval=interval,
                            auto_adjust=True,
                            progress=False,
                        )

                        if isinstance(ticker_data.columns, pd.MultiIndex):
                            if symbol in ticker_data.columns.levels[1]:
                                data = ticker_data.loc[:, (slice(None), symbol)]
                                data.columns = data.columns.droplevel(1)
                            elif symbol in ticker_data.columns.levels[0]:
                                 data = ticker_data.loc[:, (symbol, slice(None))]
                                 data.columns = data.columns.droplevel(0)
                            else:
                                logger.warning(f"Multi-column data for {symbol} but symbol not found in columns. Data: {ticker_data.columns}. Returning empty.")
                                return pd.DataFrame()
                        else:
                            data = ticker_data

                        if data.empty:
                            logger.warning(f"yf.download returned empty data for {symbol} on attempt {attempt+1}.")
                            if attempt < retries - 1:
                                time.sleep(1 * (attempt + 1))
                                continue
                            return pd.DataFrame()
                        return data

                    except TypeError as te:
                        logger.warning(f"TypeError during yf.download for {symbol} (attempt {attempt+1}): {te}. Retrying...")
                        if attempt < retries - 1:
                            time.sleep(1 * (attempt + 1))
                            continue
                        return pd.DataFrame()

                    except Exception as e:
                        logger.warning(f"Error during yf.download for {symbol} (attempt {attempt+1}): {e}. Retrying...")
                        if attempt < retries - 1:
                            time.sleep(1 * (attempt + 1))
                            continue
                        return pd.DataFrame()
                return pd.DataFrame()
            # --- END _download_data DEFINITION ---


            for sym_norm in symbols_to_query_normalized:
                data_for_sym = _download_data(sym_norm, period=period_str, interval=interval_str) # Call the local _download_data
                if not data_for_sym.empty:
                    all_downloaded_data_map[sym_norm] = data_for_sym
                else:
                    logger.warning(f"No data retrieved for {sym_norm} during initial download.")
            
            logger.info(f"Successfully downloaded data for {len(all_downloaded_data_map)} out of {len(symbols_to_query_normalized)} symbols.")


        # Generate signals for each symbol for which data was downloaded
        if all_downloaded_data_map:
            logger.info(f"Generating signals for {len(all_downloaded_data_map)} symbols.")
            max_workers = min(len(all_downloaded_data_map), 5)

            if max_workers > 0:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    symbols_to_submit_for_signal = []
                    if found_ticker_in_query:
                        if KNOWN_TICKERS.get(found_ticker_in_query, found_ticker_in_query) in all_downloaded_data_map:
                            symbols_to_submit_for_signal.append(found_ticker_in_query)
                    else:
                        for original_ticker_name in user_profile_data.get("Top_10_Products", []):
                            if KNOWN_TICKERS.get(original_ticker_name, original_ticker_name) in all_downloaded_data_map:
                                symbols_to_submit_for_signal.append(original_ticker_name)

                    future_to_symbol = {
                        executor.submit(get_trading_signal, original_ticker_name, requested_timeframe, all_downloaded_data_map): original_ticker_name
                        for original_ticker_name in symbols_to_submit_for_signal
                    }

                    for future in concurrent.futures.as_completed(future_to_symbol):
                        sym_orig_name = future_to_symbol[future]
                        try:
                            signal_data = future.result()
                            signals_raw_data.append(signal_data)
                        except Exception as exc:
                            logger.error(f'🚨 {sym_orig_name} signal generation failed concurrently: {exc}')
                            signals_raw_data.append({
                                "name": sym_orig_name, "category": "Error", "target_zone": "N/A",
                                "summary": f"Signal generation failed: {exc}", "key_levels": {},
                                "advice": "Analysis suspended due to error.", "signal_direction": "ERROR",
                                "last_close_price": "N/A", "stop_loss": "N/A", "take_profit": "N/A", "risk_reward_ratio": "N/A"
                            })
            else:
                logger.info("No symbols to process for signals after data download.")
                signals_raw_data = []
        else:
            logger.info("No data downloaded, so no signals generated.")
            signals_raw_data = []

        logger.debug(f"Raw Signals Data: {signals_raw_data}")

        # Determine how to retrieve news based on query type
        if is_news_only_query:
            logger.info("Detected as a news-only query. Retrieving specific news context.")
            context_news = chroma_retriever(user_prompt + " financial market news developments")
            if context_news == "No relevant news context found.":
                context_news = chroma_retriever("latest market moving news headlines")
        elif is_trading_query:
            context_news = chroma_retriever(user_prompt)
            if context_news == "No relevant news context found.":
                context_news = chroma_retriever("financial market news update")
        else:
            context_news = chroma_retriever(user_prompt)
            if context_news == "No relevant news context found.":
                 context_news = chroma_retriever("general market news")
        logger.debug(f"Retrieved Context:\n{context_news}")


        system_prompt = get_system_prompt_str(is_news_only_query)
        
        structured_input_for_llm = (
            "USER_QUERY: " + user_prompt + "\n\n" +
            "TRADING_SIGNAL_DATA:\n" + format_signal_output_for_llm(signals_raw_data) + "\n\n" +
            "NEWS_CONTEXT:\n" + context_news + "\n\n"
        )
        if is_news_only_query:
            structured_input_for_llm += "Generate ONLY the news report following your strict instructions. Focus on synthesizing 'NEWS_CONTEXT'."
        else:
            structured_input_for_llm += "Generate the market report following your strict instructions. "
            structured_input_for_llm += "Act as TraderFlow AI."


        ollama_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": structured_input_for_llm}
        ]

        llm_start_time = time.time()
        response = ollama_client.chat(model=settings.LLM_MODEL_ID, messages=ollama_messages)
        llm_end_time = time.time()
        llm_thinking_time = round(llm_end_time - llm_start_time, 2)

        final_response_content = response.get('message', {}).get('content', "Sorry, TraderFlow AI couldn't generate a response at this moment. Please try again.")

        logger.info(f"RAG response generated successfully in {llm_thinking_time} seconds (LLM inference).")
        return {"response": final_response_content, "thinking_time": llm_thinking_time}

    except Exception as e:
        logger.exception(f"🚨 Error in RAG response for prompt '{user_prompt}': {e}")
        return {"response": "Sorry, an internal error occurred while processing your request. TraderFlow AI is experiencing technical difficulties.", "thinking_time": "N/A"}