import streamlit as st
import requests
from bs4 import BeautifulSoup
try:
    from transformers import pipeline
except ImportError:
    # Fallback import for older versions
    from transformers.pipelines import pipeline
import re
from datetime import datetime, timedelta
import time
from typing import List, Dict
import json
import feedparser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
from functools import lru_cache
import concurrent.futures
from threading import Lock

# Configure Streamlit page
st.set_page_config(
    page_title=" Global News Topic Tracker",
    page_icon="🌍",
    layout="wide"
)

# Load summarization model
@st.cache_resource
def load_summarizer():
    """Initialize and cache the summarization model with optimized settings"""
    return pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        device=-1,  # Use CPU for better compatibility
        torch_dtype="auto"
    )

# Global cache for articles to avoid re-scraping
_article_cache = {}
_cache_lock = Lock()

def get_cache_key(query: str, num_articles: int) -> str:
    """Generate cache key for articles"""
    return hashlib.md5(f"{query}_{num_articles}".encode()).hexdigest()

@lru_cache(maxsize=32)
def cached_scrape_google_news(query: str, num_articles: int) -> tuple:
    """Cached version of scrape_google_news"""
    # Convert to tuple for caching (lists aren't hashable)
    articles = scrape_google_news(query, num_articles)
    return tuple(json.dumps(article) for article in articles)

summarizer = load_summarizer()


# News scraping functions
def scrape_google_news(query: str, num_articles: int = 10) -> List[Dict]:
    """Scrape Google News for trending topics"""
    try:
        # Google News search URL
        url = f"https://news.google.com/search?q={query}&hl=en&gl=US&ceid=US%3Aen"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        articles = []
        
        # Try multiple selectors for different Google News layouts
        article_selectors = [
            'article',
            'div[jslog]',
            'div[data-n-tid]',
            'div[jscontroller]',
            '.JtKRv',
            '.xrnccd',
            '.WwrzSb'
        ]
        
        article_elements = []
        for selector in article_selectors:
            elements = soup.select(selector)
            if elements:
                article_elements = elements[:num_articles]
                break
        
        # If no articles found with selectors, try finding any div with news-like content
        if not article_elements:
            all_divs = soup.find_all('div')
            for div in all_divs:
                if div.get_text() and len(div.get_text().strip()) > 20:
                    # Check if this div contains news-like content
                    text = div.get_text().strip()
                    if any(keyword in text.lower() for keyword in ['news', 'report', 'breaking', 'update']):
                        article_elements.append(div)
                        if len(article_elements) >= num_articles:
                            break
        
        for article in article_elements:
            try:
                # Extract title - try multiple selectors
                title = "No title"
                title_selectors = ['h3', 'h4', 'h2', 'a[role="heading"]', '.JtKRv', '.xrnccd', 'a']
                
                for selector in title_selectors:
                    title_elem = article.select_one(selector)
                    if title_elem and title_elem.get_text().strip():
                        title = title_elem.get_text().strip()
                        break
                
                # If still no title, try to extract from any text content
                if title == "No title":
                    text_content = article.get_text().strip()
                    if text_content:
                        # Take first line as title if it's reasonable length
                        first_line = text_content.split('\n')[0].strip()
                        if 10 <= len(first_line) <= 200:
                            title = first_line
                
                # Extract source - try multiple selectors with improved logic
                source = "Unknown source"
                source_selectors = [
                    '.wEwyrc', '.NUnG9d', '.wEwyrc.AVN2gc', 
                    'span[class*="source"]', 'div[class*="source"]',
                    '.CEMjEf', '.NUnG9d', '.WwrzSb', '.VDXfz',
                    'a[class*="source"]', 'div[class*="publisher"]',
                    'span[class*="publisher"]', '.ssrcss-1f3bvyz-Text'
                ]
                
                for selector in source_selectors:
                    source_elem = article.select_one(selector)
                    if source_elem and source_elem.get_text().strip():
                        source_text = source_elem.get_text().strip()
                        # Filter out generic text and ensure it's a reasonable source name
                        if (len(source_text) > 2 and len(source_text) < 50 and 
                            not any(generic in source_text.lower() for generic in 
                                   ['ago', 'hours', 'minutes', 'days', 'read', 'more', 'click', 'show'])):
                            source = source_text
                            break
                
                # If still no source found, try to extract from any text that looks like a source
                if source == "Unknown source":
                    all_text = article.get_text()
                    # Look for common news source patterns
                    import re
                    source_patterns = [
                        r'([A-Z][a-z]+ [A-Z][a-z]+)',  # "Fox Business", "BBC News"
                        r'([A-Z][a-z]+ News)',         # "CNN News", "ABC News"
                        r'([A-Z]{2,})',                # "BBC", "CNN", "NPR"
                    ]
                    
                    for pattern in source_patterns:
                        matches = re.findall(pattern, all_text)
                        for match in matches:
                            if (len(match) > 2 and len(match) < 30 and 
                                not any(generic in match.lower() for generic in 
                                       ['ago', 'hours', 'minutes', 'days', 'read', 'more'])):
                                source = match
                                break
                        if source != "Unknown source":
                            break
                
                # Extract time - try multiple selectors
                time_text = "Unknown time"
                time_selectors = ['time', '.OSrXXb', '.WW6dff', 'span[class*="time"]']
                
                for selector in time_selectors:
                    time_elem = article.select_one(selector)
                    if time_elem and time_elem.get_text().strip():
                        time_text = time_elem.get_text().strip()
                        break
                
                # Extract link
                link = ""
                link_elem = article.find('a', href=True)
                if link_elem:
                    link = link_elem.get('href', '')
                if link.startswith('./'):
                    link = f"https://news.google.com{link[1:]}"
                elif link.startswith('/'):
                    link = f"https://news.google.com{link}"
                
                # Extract description/summary with improved Google News scraping
                description = ""
                
                # Try to find description in various ways
                # Method 1: Look for specific Google News description elements
                desc_selectors = [
                    '.Y3v8qd', '.GI74Re', '.s3v9rd', '.stP3nb', '.JheGif',
                    '.VDXfz', '.WwrzSb', '.JtKRv', '.xrnccd',
                    'div[jslog]', 'div[data-n-tid]', 'div[jscontroller]',
                    'p', 'span', 'div'
                ]
                
                for desc_selector in desc_selectors:
                    desc_elements = article.select(desc_selector)
                    for desc_elem in desc_elements:
                        if desc_elem and desc_elem.get_text().strip():
                            desc_text = desc_elem.get_text().strip()
                            # Check if this looks like a description
                            if (desc_text != title and 
                                len(desc_text) > 50 and len(desc_text) < 500 and
                                not desc_text.startswith(title[:30]) and
                                not any(generic in desc_text.lower() for generic in 
                                       ['ago', 'hours', 'minutes', 'days', 'read', 'more', 'click', 'show', 'view', 'source', 'news'])):
                                description = desc_text
                                break
                    if description:
                        break
                
                # Method 2: If no description found, try to extract from the full text
                if not description:
                    full_text = article.get_text()
                    if full_text and len(full_text) > len(title):
                        # Split by lines and find the best description
                        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                        for line in lines:
                            if (line != title and 
                                len(line) > 50 and len(line) < 500 and
                                not line.startswith(title[:30]) and
                                not any(generic in line.lower() for generic in 
                                       ['ago', 'hours', 'minutes', 'days', 'read', 'more', 'click', 'show', 'view', 'source', 'news', 'google'])):
                                description = line
                                break
                
                # Method 3: Try to get description from the article's parent or sibling elements
                if not description:
                    # Look in parent elements for description
                    parent = article.parent
                    if parent:
                        parent_text = parent.get_text().strip()
                        if parent_text and len(parent_text) > len(title):
                            lines = [line.strip() for line in parent_text.split('\n') if line.strip()]
                            for line in lines:
                                if (line != title and 
                                    len(line) > 50 and len(line) < 500 and
                                    not line.startswith(title[:30]) and
                                    not any(generic in line.lower() for generic in 
                                           ['ago', 'hours', 'minutes', 'days', 'read', 'more', 'click', 'show', 'view', 'source', 'news'])):
                                    description = line
                                    break
                
                # Clean up description
                if description:
                    # Remove extra whitespace and limit length
                    description = ' '.join(description.split())
                    if len(description) > 300:
                        description = description[:300] + "..."
                
                # Only add if we have at least a title
                if title != "No title" and len(title) > 5:
                    articles.append({
                        'title': title,
                        'source': source,
                        'time': time_text,
                        'link': link,
                        'description': description
                    })
                    
            except Exception as e:
                continue
                
        return articles[:num_articles]
    except Exception as e:
        st.error(f"Error scraping Google News: {str(e)}")
        # Fallback to RSS feeds
        return scrape_rss_feeds(query, num_articles)

def scrape_rss_feeds(query: str, num_articles: int = 10) -> List[Dict]:
    """Fallback method using RSS feeds when Google News scraping fails"""
    try:
        # List of RSS feeds to try - prioritize search-specific feeds
        rss_feeds = [
            f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US%3Aen",
            "https://feeds.bbci.co.uk/news/rss.xml",
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.reuters.com/reuters/topNews",
            "https://feeds.npr.org/1001/rss.xml",
            "https://feeds.foxnews.com/foxnews/latest",
            "https://rss.cbc.ca/rss/topstories.xml"
        ]
        
        articles = []
        
        for feed_url in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:num_articles * 2]:  # Get more entries to filter
                    if len(articles) >= num_articles:
                        break
                        
                    # Extract article information
                    title = entry.get('title', 'No title')
                    
                    # Simple relevance filter - check if query terms appear in title or description
                    query_terms = query.lower().split()
                    title_lower = title.lower()
                    description_lower = ""
                    
                    # Get description for filtering
                    if hasattr(entry, 'summary') and entry.summary:
                        desc_soup = BeautifulSoup(entry.summary, 'html.parser')
                        description_lower = desc_soup.get_text().strip().lower()
                    
                    # Check relevance (at least one query term should match)
                    is_relevant = any(term in title_lower or term in description_lower for term in query_terms)
                    
                    # For general feeds, be more lenient
                    if not is_relevant and not feed_url.startswith('https://news.google.com'):
                        is_relevant = True  # Accept all articles from general feeds
                    
                    if not is_relevant:
                        continue
                    
                    # Better source extraction from RSS feeds
                    source = "Unknown source"
                    if hasattr(entry, 'source') and entry.source:
                        source = entry.source.get('title', 'Unknown source')
                    elif hasattr(entry, 'author'):
                        source = entry.author
                    elif 'link' in entry:
                        # Try to extract source from URL
                        import re
                        url = entry['link']
                        if 'bbc.com' in url:
                            source = "BBC News"
                        elif 'cnn.com' in url:
                            source = "CNN"
                        elif 'reuters.com' in url:
                            source = "Reuters"
                        elif 'npr.org' in url:
                            source = "NPR"
                        elif 'foxnews.com' in url:
                            source = "Fox News"
                        elif 'nytimes.com' in url:
                            source = "New York Times"
                        elif 'washingtonpost.com' in url:
                            source = "Washington Post"
                        else:
                            # Extract domain name as source
                            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                            if domain_match:
                                domain = domain_match.group(1)
                                source = domain.replace('.com', '').replace('.org', '').title()
                    
                    link = entry.get('link', '')
                    
                    # Extract description from RSS feed with enhanced logic
                    description = ""
                    if hasattr(entry, 'summary') and entry.summary:
                        description = entry.summary
                    elif hasattr(entry, 'description') and entry.description:
                        description = entry.description
                    elif hasattr(entry, 'content') and entry.content:
                        description = entry.content
                    
                    # Clean up description (remove HTML tags and improve quality)
                    if description:
                        from bs4 import BeautifulSoup
                        desc_soup = BeautifulSoup(description, 'html.parser')
                        description = desc_soup.get_text().strip()
                        
                        # Ensure description is meaningful and different from title
                        if (description and len(description) > 40 and 
                            description != title and 
                            not description.startswith(title[:25]) and
                            not any(generic in description.lower() for generic in 
                                   ['read more', 'continue reading', 'click here', 'view full'])):
                            # Limit description length
                            if len(description) > 300:
                                description = description[:300] + "..."
                        else:
                            description = ""
                    
                    # Extract time
                    time_text = "Unknown time"
                    if hasattr(entry, 'published'):
                        try:
                            pub_date = datetime(*entry.published_parsed[:6])
                            time_text = pub_date.strftime("%Y-%m-%d %H:%M")
                        except:
                            time_text = entry.get('published', 'Unknown time')
                    
                    # Only add if we have a valid title
                    if title and title != 'No title' and len(title) > 5:
                        articles.append({
                            'title': title,
                            'source': source,
                            'time': time_text,
                            'link': link,
                            'description': description
                        })
                
                if len(articles) >= num_articles:
                    break
                    
            except Exception as e:
                continue
        
        return articles[:num_articles]
        
    except Exception as e:
        st.error(f"Error with RSS feeds: {str(e)}")
        return []

def get_trending_topics() -> List[str]:
    """Return Google News section options for the dropdown"""
    # Return the specific Google News sections as requested
    return ["Top stories", "Local news", "Picks for you", "For you", "Your topics", "Sources"]

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_real_trending_topics() -> List[str]:
    """Extract real trending topics from actual news headlines for the chart"""
    try:
        # Get actual news articles to extract trending topics from headlines
        articles = scrape_google_news("", 10)  # Reduced from 15 for faster processing
        
        if articles:
            topics = []
            for article in articles:
                title = article.get('title', '')
                if title and title != "No title" and len(title) > 10:
                    # Extract key topics from titles (first few words)
                    words = title.split()
                    if len(words) >= 3:
                        # Take the most important words (usually first 2-3)
                        topic = ' '.join(words[:3])
                        if (topic not in topics and 
                            not any(generic in topic.lower() for generic in 
                                   ['google', 'news', 'search', 'more', 'show', 'read', 'click', 'menu', 'sign', 'login', 'top stories', 'picks for you', 'breaking news'])):
                            topics.append(topic)
                            if len(topics) >= 6:  # Reduced from 8 for faster processing
                                break
            
            if len(topics) >= 3:
                return topics
        
        # Fallback: return static topics if scraping fails
        return ["Technology News", "Business Updates", "Health & Science", "World Politics", "Sports Highlights"]
        
    except Exception as e:
        print(f"Error fetching real trending topics: {e}")
        return ["Technology News", "Business Updates", "Health & Science", "World Politics", "Sports Highlights"]

def get_explore_topics() -> List[str]:
    """Return different topics for the 'Click to explore' section"""
    # These are different from trending topics - more general categories
    return [
        "Technology News",
        "Business Updates", 
        "Health & Science",
        "World Politics",
        "Sports Highlights",
        "Entertainment News",
        "Climate & Environment",
        "Economic News"
    ]

def search_news(query: str, search_type: str = "topic", num_articles: int = 10) -> List[Dict]:
    """Search news by different criteria"""
    try:
        if search_type == "topic":
            # Search by topic/keywords
            return scrape_google_news(query, num_articles)
        elif search_type == "location":
            # Search by location (e.g., "news from India", "New York news")
            location_query = f"{query} news"
            return scrape_google_news(location_query, num_articles)
        elif search_type == "source":
            # Search by source (e.g., "CNN", "BBC", "Reuters")
            source_query = f"site:{query.lower()}.com OR {query}"
            return scrape_google_news(source_query, num_articles)
        elif search_type == "category":
            # Search by category
            return scrape_google_news(query, num_articles)
        else:
            # Default to topic search
            return scrape_google_news(query, num_articles)
    except Exception as e:
        print(f"Error in search: {e}")
        return []

def get_search_categories() -> List[str]:
    """Return available search categories"""
    return [
        "Technology",
        "Business", 
        "Health",
        "Politics",
        "Sports",
        "Entertainment",
        "Science",
        "World News",
        "Economy",
        "Environment"
    ]

def get_search_locations() -> List[str]:
    """Return popular search locations"""
    return [
        "United States",
        "India", 
        "United Kingdom",
        "Canada",
        "Australia",
        "Germany",
        "France",
        "Japan",
        "China",
        "Brazil"
    ]

def get_search_sources() -> List[str]:
    """Return popular news sources"""
    return [
        "CNN",
        "BBC", 
        "Reuters",
        "Associated Press",
        "The New York Times",
        "The Guardian",
        "Washington Post",
        "NPR",
        "Al Jazeera",
        "Bloomberg"
    ]


@st.cache_data(ttl=300)  # Cache for 5 minutes
def summarize_news_articles(articles: List[Dict]) -> str:
    """Optimized news summarization with caching"""
    if not articles:
        return "No articles to summarize."
    
    # Create cache key from article titles
    cache_key = "|".join([article.get('title', '')[:50] for article in articles[:5]])
    
    # Combine article titles and descriptions for better context
    combined_text = ""
    for article in articles[:5]:  # Reduced from 8 to 5 for faster processing
        title = article.get('title', '')
        description = article.get('description', '')
        
        if title and title != "No title":
            combined_text += f"{title}. "
        if description and description != "Description not available from source":
            combined_text += f"{description} "
        combined_text += " "
    
    # Clean and prepare text for summarization
    combined_text = re.sub(r'\s+', ' ', combined_text).strip()
    if len(combined_text) > 1000:  # Reduced from 2000 for faster processing
        combined_text = combined_text[:1000] + "..."
    
    try:
        # Optimized summarization with smaller parameters for speed
        summary = summarizer(
            combined_text, 
            max_length=200,  # Reduced from 400
            min_length=80,   # Reduced from 150
            do_sample=False,
            num_beams=2,     # Reduced beam search for speed
            early_stopping=True
        )
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Main Streamlit App
st.title("🌍 Global News Topic Tracker")
st.markdown("**Search and analyze news from multiple sources with AI-powered summaries**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Number of articles
    num_articles = st.slider(
        "Number of Articles",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of articles to fetch and analyze"
    )
    
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    
    # Show descriptions option
    show_descriptions = st.checkbox("Show article descriptions", value=True)
    
    st.markdown("---")
    st.markdown("**💡 How to use:**")
    st.markdown("1. Enter a topic in the search box")
    st.markdown("2. Click 'Search News'")
    st.markdown("3. View results and AI summary below")
    
    # Manual refresh button
    if st.button("🔄 Refresh All"):
        # Clear ALL session state variables
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        
        # Also clear Streamlit's internal caches
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Set a flag to show the cleared status
        st.session_state['refresh_cleared'] = True
        
        st.success("✅ Everything cleared! All caches and data reset.")
        st.rerun()

# Main search interface - simplified
st.subheader("🔍 Search News")

# Simple search interface
col_search, col_search_btn, col_clear_btn = st.columns([3, 1, 1])

with col_search:
    search_query = st.text_input(
        "Enter your search topic:",
        placeholder="e.g., artificial intelligence, climate change, elections, technology news",
        key="main_search_input",
        help="Search for any topic, location, or news source"
    )

with col_search_btn:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

with col_clear_btn:
    clear_search_clicked = st.button("🗑️ Clear", use_container_width=True)

# Quick search suggestions
st.markdown("**💡 Quick searches:**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🤖 AI News", key="quick_ai"):
        st.session_state.search_query = "artificial intelligence"
        st.session_state.auto_fetch = True
        # Clear the search input
        if 'main_search_input' in st.session_state:
            del st.session_state.main_search_input
        st.rerun()

with col2:
    if st.button("🌍 World News", key="quick_world"):
        st.session_state.search_query = "world news"
        st.session_state.auto_fetch = True
        # Clear the search input
        if 'main_search_input' in st.session_state:
            del st.session_state.main_search_input
        st.rerun()

with col3:
    if st.button("💼 Business", key="quick_business"):
        st.session_state.search_query = "business news"
        st.session_state.auto_fetch = True
        # Clear the search input
        if 'main_search_input' in st.session_state:
            del st.session_state.main_search_input
        st.rerun()

with col4:
    if st.button("🏥 Health", key="quick_health"):
        st.session_state.search_query = "health news"
        st.session_state.auto_fetch = True
        # Clear the search input
        if 'main_search_input' in st.session_state:
            del st.session_state.main_search_input
        st.rerun()

# Handle search and clear
if search_clicked and search_query:
    st.session_state.search_query = search_query
    st.session_state.search_type = "topic"
    st.session_state.auto_fetch = True
    st.rerun()

if clear_search_clicked:
    # Clear search-related session state
    if 'search_query' in st.session_state:
        del st.session_state.search_query
    if 'search_type' in st.session_state:
        del st.session_state.search_type
    if 'auto_fetch' in st.session_state:
        del st.session_state.auto_fetch
    if 'current_summary' in st.session_state:
        del st.session_state.current_summary
    # Clear the search input
    if 'main_search_input' in st.session_state:
        del st.session_state.main_search_input
    st.success("Search cleared!")
    st.rerun()

# Show status message if everything was cleared
if 'refresh_cleared' in st.session_state:
    st.info("🔄 All data cleared! Enter a new search topic to get fresh results.")
    del st.session_state['refresh_cleared']

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Fetch and display articles when search is triggered
    if st.session_state.get('auto_fetch', False) or st.session_state.get('search_query'):
        # Get search query from session state
        search_query = st.session_state.get('search_query', '')
        search_type = st.session_state.get('search_type', 'topic')
        
        if search_query:
            with st.spinner(f"🔍 Searching for '{search_query}'..."):
                # Try multiple sources for better results
                articles = []
                
                # First try Google News
                try:
                    articles = scrape_google_news(search_query, num_articles)
                except Exception as e:
                    st.warning(f"Google News search failed: {str(e)}")
                
                # If no articles from Google News, try RSS feeds
                if not articles:
                    try:
                        articles = scrape_rss_feeds(search_query, num_articles)
                    except Exception as e:
                        st.warning(f"RSS feed search failed: {str(e)}")
                
                # If still no articles, try a broader search
                if not articles:
                    try:
                        broader_query = f"{search_query} news"
                        articles = scrape_rss_feeds(broader_query, num_articles)
                    except Exception as e:
                        st.error(f"All search methods failed: {str(e)}")
            
            if articles:
                st.success(f"✅ Found {len(articles)} articles about '{search_query}'")
                
                # Display articles
                for i, article in enumerate(articles, 1):
                    with st.expander(f"📄 {i}. {article['title'][:80]}{'...' if len(article['title']) > 80 else ''}"):
                        st.write(f"**📰 Source:** {article['source']}")
                        st.write(f"**🕒 Time:** {article['time']}")
                        st.write(f"**{article['title']}**")
                        
                        # Description
                        if article['description'] and article['description'] != "Description not available from source":
                            st.write(f"📝 **Description:** {article['description']}")
                        
                        # Link to full article
                        if article['link']:
                            st.write(f"🔗 [Read full article]({article['link']})")
                
                # Generate summary and store in session state
                with st.spinner("🧠 Generating AI summary..."):
                    summary = summarize_news_articles(articles)
                    st.session_state.current_summary = summary
                
                # Display AI Summary right after the articles
                st.markdown("---")
                st.subheader("🤖 AI-Powered Summary")
                st.info(summary)
                
            else:
                st.warning("⚠️ No articles found. Try a different search query or check your internet connection.")
                st.session_state.current_summary = None
                
                # Show helpful suggestions
                st.info("💡 **Try these suggestions:**")
                st.markdown("- Use more general terms (e.g., 'technology' instead of 'quantum computing')")
                st.markdown("- Check your internet connection")
                st.markdown("- Try different keywords")
        
        # Clear the auto_fetch flag after processing
        if 'auto_fetch' in st.session_state:
            del st.session_state.auto_fetch

with col2:
    st.subheader("📊 Analytics & Trends")
    
    # Show trending topics
    st.markdown("**🔥 Trending Topics**")
    
    # Get trending topics
    if 'sidebar_trending_topics' not in st.session_state:
        with st.spinner("Loading trending topics..."):
            st.session_state.sidebar_trending_topics = get_real_trending_topics()
    
    trending_topics_sidebar = st.session_state.sidebar_trending_topics
    
    if trending_topics_sidebar and len(trending_topics_sidebar) > 0:
        # Show trending topics as clickable buttons
        for i, topic in enumerate(trending_topics_sidebar[:5], 1):
            if st.button(f"{i}. {topic}", key=f"trending_topic_{i}", use_container_width=True):
                st.session_state.search_query = topic
                st.session_state.search_type = "topic"
                st.session_state.auto_fetch = True
                # Clear the search input
                if 'main_search_input' in st.session_state:
                    del st.session_state.main_search_input
                st.rerun()
    else:
        st.info("📊 No trending topics available. Try searching for a topic!")
    
    st.markdown("---")
    
    # Show search suggestions
    st.markdown("**💡 Popular Searches**")
    popular_searches = [
        "artificial intelligence",
        "climate change", 
        "technology news",
        "business updates",
        "health news",
        "world politics"
    ]
    
    for search_term in popular_searches:
        if st.button(f"🔍 {search_term}", key=f"popular_{search_term}", use_container_width=True):
            st.session_state.search_query = search_term
            st.session_state.search_type = "topic"
            st.session_state.auto_fetch = True
            # Clear the search input
            if 'main_search_input' in st.session_state:
                del st.session_state.main_search_input
            st.rerun()
    
    # Auto-refresh status
    if auto_refresh:
        st.info("🔄 Auto-refresh enabled (30s)")
        time.sleep(30)
        st.rerun()
    else:
        st.info("⏸️ Auto-refresh disabled")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🌍 Global News Topic Tracker | Powered by Streamlit & Transformers
    </div>
    """,
    unsafe_allow_html=True
)
