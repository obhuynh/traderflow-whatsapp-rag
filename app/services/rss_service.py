# app/services/rss_service.py
import feedparser
import chromadb
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import hashlib # Import for generating stable IDs

# Initialize ChromaDB client to connect to the 'chroma' service in docker-compose
client = chromadb.HttpClient(host=settings.CHROMA_HOST, port=settings.CHROMA_PORT)

# Create or get a collection to store the feed data
collection = client.get_or_create_collection(name="rss_feeds")

# Initialize a text splitter to break down large articles
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
)

def _get_entry_content(entry) -> str:
    """Helper to extract content from RSS entry, prioritizing content over summary."""
    if hasattr(entry, 'content') and entry.content:
        for content_item in entry.content:
            if hasattr(content_item, 'value') and content_item.value:
                return content_item.value
    if hasattr(entry, 'summary') and entry.summary:
        return entry.summary
    return ""

def ingest_rss_feed(feed_url: str):
    """Parses a single RSS feed and ingests/updates its content into ChromaDB."""
    try:
        print(f"Ingesting RSS feed: {feed_url}")
        feed = feedparser.parse(feed_url)

        documents_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for entry in feed.entries:
            raw_content = _get_entry_content(entry)
            soup = BeautifulSoup(raw_content, "html.parser")
            text_content = soup.get_text()

            if not text_content:
                print(f"Skipping empty entry from {entry.link if hasattr(entry, 'link') else 'N/A'} in {feed_url}")
                continue

            # Use a more stable ID, e.g., a hash of the content + link
            # This helps in idempotency if entries change slightly but link/title remain
            entry_id_base = hashlib.md5(f"{entry.link or entry.title or text_content[:50]}".encode('utf-8')).hexdigest()

            # Split the article content into smaller, manageable chunks
            chunks = text_splitter.split_text(text_content)

            if not chunks:
                print(f"No chunks generated for entry from {entry.link if hasattr(entry, 'link') else 'N/A'} in {feed_url}")
                continue

            for j, chunk in enumerate(chunks):
                doc_id = f"{entry_id_base}_{j}"
                documents_to_add.append(chunk)
                metadatas_to_add.append({"source": entry.link if hasattr(entry, 'link') else "unknown", "title": entry.title if hasattr(entry, 'title') else "No Title"})
                ids_to_add.append(doc_id)

        if documents_to_add:
            # ChromaDB's add method will update existing IDs if they match
            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                ids=ids_to_add
            )
            print(f"✅ Ingested/Updated {len(documents_to_add)} document chunks from {feed_url}")
        else:
            print(f"No new or updated documents to ingest from {feed_url}")

    except Exception as e:
        print(f"❌ Failed to ingest RSS feed {feed_url}. Error: {e}")

def update_all_feeds():
    """A helper function to update all predefined news sources."""
    print("Starting RSS feed update process...")
    # List of financial news RSS feeds
    rss_urls = [
        "https://cointelegraph.com/rss/",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://www.fxstreet.com/rss/news",
        "https://www.litefinance.org/rss-smm/blog/",
        "https://www.forexlive.com/feed",
        "https://news.instaforex.com/news"
    ]
    for url in rss_urls:
        ingest_rss_feed(url)
    print("RSS feed update process finished.")