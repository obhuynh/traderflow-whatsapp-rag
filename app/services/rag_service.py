# app/services/rag_service.py
import ollama
import chromadb
from pathlib import Path
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import Dict, Any, List
from datetime import datetime
import traceback

from app.core.config import settings
from app.services.trading_service import get_trading_signal # Assuming this exists and works

# --- INITIALIZATION ---
ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
chroma_client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)
rss_collection = chroma_client.get_or_create_collection(name="rss_feeds")

# --- KNOWN TICKERS ---
KNOWN_TICKERS = {
    # Crypto
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD", "SOLANA": "SOL-USD",
    "XRP": "XRP-USD", "RIPPLE": "XRP-USD",
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
    "NASDAQ": "^IXIC", "NDX": "^IXIC"
}
SORTED_TICKERS = sorted(KNOWN_TICKERS.keys(), key=len, reverse=True)

# --- Helper functions ---

def chroma_retriever(query: str) -> str:
    """Retrieve top 3 relevant news docs from ChromaDB."""
    try:
        results = rss_collection.query(query_texts=[query], n_results=3)
        if results and results['documents'] and results['documents'][0]:
            # Each element in documents[0] is a chunk of text
            return "\n\n".join(results['documents'][0])
        return "No relevant news context found."
    except Exception as e:
        print(f"Error retrieving from ChromaDB: {e}")
        return "No relevant news context found."

def fill_market_prompt(template_str: str, data: dict) -> str:
    """Replace {{var}} placeholders in prompt template with actual data."""
    formatted_template = template_str.replace("{{", "{").replace("}}", "}")
    if "date" not in data:
        data["date"] = datetime.now().strftime("%Y-%m-%d")
    try:
        return formatted_template.format(**data)
    except KeyError as e:
        missing_key = e.args[0]
        # Fallback for missing keys if the template is not fully matched
        return f"Warning: Missing value for placeholder '{missing_key}' in prompt data. Remaining prompt: {formatted_template}"

def format_signal_output(signals: List[Dict[str, Any]]) -> str:
    """
    Format multiple trading signals into a neat, emoji-enhanced string.
    Each signal dict must include:
    - name, target_zone, summary, key_levels (dict with support/resistance), advice
    """
    if not signals:
        return "⚠️ Sorry, no specific trading signals are available right now for your selected pairs. Please check back later!"

    output = []
    icon_map = {
        "Oil": "🛢️",
        "Gold": "🥇",
        "EURUSD": "💶",
        "BTC": "₿",
        "Crypto": "🔗",
        "Forex": "📈",
        "Commodity": "⛏️",
        "Index": "📊"
    }

    for sig in signals:
        icon = icon_map.get(sig.get("category", ""), "📌")
        name = sig.get("name", "Unknown Asset")
        target_zone = sig.get("target_zone", "N/A")
        summary = sig.get("summary", "No detailed summary provided.")
        key_levels = sig.get("key_levels", {})
        advice = sig.get("advice", "No specific trading advice.")
        direction = sig.get("direction", "neutral").lower() # Added for smartness (e.g., 'bullish', 'bearish')

        signal_indicator = ""
        if direction == "bullish":
            signal_indicator = "⬆️ **Bullish**"
        elif direction == "bearish":
            signal_indicator = "⬇️ **Bearish**"
        else:
            signal_indicator = "↔️ **Neutral/Range-bound**"

        output.append(f"## {icon} {name} Price Forecast ({signal_indicator})\n")
        if target_zone != "N/A":
            output.append(f"🎯 **Target Zone:** `{target_zone}`\n") # Using backticks for clearer levels
        if summary:
            output.append(summary.strip() + "\n")

        if key_levels:
            levels_output = []
            if "support" in key_levels and key_levels["support"]:
                levels_output.append(f"🛡️ **Support:** {', '.join(map(str, key_levels['support']))}")
            if "resistance" in key_levels and key_levels["resistance"]:
                levels_output.append(f"🚧 **Resistance:** {', '.join(map(str, key_levels['resistance']))}")
            if levels_output:
                output.append(" | ".join(levels_output) + "\n")

        if advice:
            output.append(f"💡 **Advice:** {advice.strip()}\n")
        output.append("\n---\n")

    return "\n".join(output)

def get_system_prompt_str() -> str:
    """Defines the LLM's persona and general instructions."""
    return (
        "You are a highly knowledgeable and helpful AI assistant specializing in financial markets and general knowledge. "
        "Your primary goal is to provide accurate trading signals based on the given data and answer general questions using provided context from news feeds. "
        "When providing trading signals, be precise and use the provided data. When answering general questions, use the news context to enrich your answer if relevant. "
        "If a question is clearly a trading signal request, prioritize that. Otherwise, answer generally."
        "Always maintain a professional, clear, and concise tone. Disclaimers about trading risks are important for trading signals."
    )

def get_prompt_template_str() -> str:
    """Load raw prompt template string from file or fallback."""
    path = Path("app/prompt_template.txt")
    if not path.exists():
        # A more detailed fallback template for the LLM
        return (
            "Given the following news context and user query, provide a comprehensive response.\n\n"
            "**News Context:**\n{context}\n\n"
            "**Trading Signals:**\n{signals}\n\n"
            "**User Query:** {question}\n\n"
            "---"
            "Based on the above, please provide your answer. If the query is about trading, integrate the signals and relevant news. If it's a general question, use the news context. Always include a disclaimer for trading advice."
        )
    return path.read_text()

# --- MAIN ORCHESTRATION FUNCTION ---

def get_rag_response(user_prompt: str, user_profile_data: Dict[str, Any] = None) -> str:
    print(f"Orchestrating response for prompt: '{user_prompt}'")

    if user_profile_data is None:
        user_profile_data = {
            "Top_10_Products": [
                "EURUSD=X", "GBPJPY=X", "GC=F", "CL=F", "BTC-USD",
                "ETH-USD", "SP500", "NDX", "SI=F", "USDJPY=X"
            ],
            "Risk_Level": "Medium",
            "User_Sentiment": "Neutral",
            "Broker": "Pepperstone",
            "Trading_Style": "Scalping"
        }

    prompt_upper = user_prompt.upper()
    trading_related_keywords = ["TRADE", "SIGNAL", "BUY", "SELL", "POSITION", "FORECAST", "ANALYZE", "MARKET"]
    is_trading_query = any(k in prompt_upper for k in trading_related_keywords) or any(ticker in prompt_upper for ticker in KNOWN_TICKERS)

    signals_to_display = "" # This will hold the formatted trading signals
    context = "" # This will hold the news context

    try:
        # Step 1: Determine if it's a trading signal request
        if is_trading_query:
            print("Detected as a trading-related query.")
            found_ticker = None
            for ticker_phrase in SORTED_TICKERS:
                if ticker_phrase in prompt_upper:
                    found_ticker = ticker_phrase
                    break

            symbols_to_query = []
            if found_ticker:
                symbols_to_query = [KNOWN_TICKERS[found_ticker]]
            else:
                symbols_to_query = user_profile_data.get("Top_10_Products", [])
            print(f"Symbols to query for trading: {symbols_to_query}")

            # Detect timeframe
            prompt_lower = user_prompt.lower()
            short_term_keywords = ["short term", "short-term", "scalp", "hourly", "fast", "quick"]
            long_term_keywords = ["long term", "long-term", "swing", "daily", "invest"]
            requested_timeframe = "auto"
            if any(k in prompt_lower for k in short_term_keywords):
                requested_timeframe = "short"
            elif any(k in prompt_lower for k in long_term_keywords):
                requested_timeframe = "long"
            print(f"Requested timeframe: {requested_timeframe}")

            signals = []
            for symbol in symbols_to_query:
                # get_trading_signal should ideally return a dict for structured data
                signal_data = get_trading_signal(symbol, timeframe=requested_timeframe)
                if isinstance(signal_data, dict):
                    signals.append(signal_data)
                elif isinstance(signal_data, str):
                    # Fallback for simple string signals if trading_service returns it
                    signals.append({
                        "name": symbol,
                        "category": "General Market",
                        "summary": signal_data,
                        "target_zone": "N/A", "key_levels": {}, "advice": ""
                    })

            signals_to_display = format_signal_output(signals)
        else:
            print("Detected as a general query.")
            # For general questions, we still might want some news context
            pass # No signals generated, proceed to get news context

        # Step 2: Retrieve relevant news context regardless of query type
        context = chroma_retriever(user_prompt)
        if context == "No relevant news context found." and is_trading_query:
             # If no specific news for a trading query, try a broader query
             context = chroma_retriever("financial news market update")
             if context != "No relevant news context found.":
                 context = "General market news:\n" + context

        # Step 3: Compose the final prompt for the LLM
        system_prompt = get_system_prompt_str()
        user_input_for_llm = f"User Query: {user_prompt}\n\n"

        if signals_to_display:
            user_input_for_llm += f"**Relevant Trading Signals:**\n{signals_to_display}\n\n"

        if context and context != "No relevant news context found.":
            user_input_for_llm += f"**Relevant News Context:**\n{context}\n\n"
        else:
            user_input_for_llm += "**No specific news context found for this query.**\n\n"

        # Add a disclaimer for trading advice
        disclaimer = (
            "Please note: Trading in financial markets involves significant risk and can result in the loss of your capital. "
            "The information provided is for educational and informational purposes only and does not constitute financial advice. "
            "Always do your own research and consult with a qualified financial professional before making any investment decisions."
        )
        user_input_for_llm += disclaimer

        # Use Langchain's ChatPromptTemplate for better structured prompts
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_input_for_llm)
        ])

        final_prompt_str = chat_template.format_messages(
            context=context, # Context will be passed to the HumanMessage template
            signals=signals_to_display,
            question=user_prompt
        )

        # Ollama expects a simple string or a list of messages, not a Langchain Message object directly for `generate`
        # We need to convert Langchain messages to Ollama's expected format.
        ollama_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input_for_llm}
        ]

        # Call Ollama LLM
        # Use `chat` method for multi-turn conversations if the LLM supports it
        # Otherwise, `generate` (older) expects just a prompt string
        response = ollama_client.chat(model=settings.LLM_MODEL_ID, messages=ollama_messages)
        return response.get('message', {}).get('content', "Sorry, I couldn't generate a response. Please try again.")

    except Exception as e:
        print(f"Error in RAG response: {e}")
        traceback.print_exc()
        return "Sorry, an error occurred while processing your request. Please check the logs for details."