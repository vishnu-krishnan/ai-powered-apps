# 🌍 Global News Topic Tracker

## Problem Statement

In today's fast-paced digital world, staying informed about current events and trending topics is crucial for individuals, businesses, and organizations. However, the sheer volume of news content available makes it challenging to:

1. **Information Overload**: With thousands of news articles published daily across multiple sources, users struggle to identify the most relevant and important stories.

2. **Time Constraints**: Busy professionals and individuals don't have time to read through multiple lengthy articles to understand key developments.

3. **Bias and Filter Bubbles**: Traditional news consumption often leads to echo chambers where users only see content that aligns with their existing views.

4. **Lack of Context**: Individual news articles often lack broader context and connections between related events.

5. **Real-time Monitoring**: There's a need for automated systems that can continuously monitor and summarize trending topics without manual intervention.

## Solution Architecture

### Overview
The Global News Topic Tracker is a web-based application that automatically scrapes Google News, identifies trending topics, and provides AI-powered summaries using Large Language Models (LLMs).

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Google News   │───▶│  Web Scraper    │───▶│  Data Processor │
│   (Data Source) │    │  (BeautifulSoup)│    │  (Text Cleaning)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │◀───│  Summary Engine │◀───│  Article Store  │
│  (User Interface)│    │  (BART/LLM)     │    │  (In-Memory)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Features

1. **Automated News Scraping**
   - Real-time scraping of Google News
   - Support for custom search queries
   - Configurable number of articles (5-20)

2. **Trending Topic Detection**
   - Automatic identification of trending topics
   - Dynamic topic selection interface
   - Real-time topic updates

3. **AI-Powered Summarization**
   - Uses Facebook's BART-large-cnn model
   - Intelligent text chunking for long articles
   - Context-aware summarization

4. **User-Friendly Interface**
   - Clean, responsive Streamlit interface
   - Auto-refresh capabilities
   - Real-time analytics dashboard

### Technical Stack

- **Frontend**: Streamlit
- **Web Scraping**: BeautifulSoup4, Requests
- **AI/ML**: Transformers (Hugging Face), PyTorch
- **Data Processing**: Python, Regular Expressions
- **Deployment**: Local/Cloud deployment ready

### Data Flow

1. **Data Collection**
   ```
   Google News → Web Scraper → Article Extraction → Data Cleaning
   ```

2. **Processing Pipeline**
   ```
   Raw Articles → Text Chunking → LLM Summarization → Final Summary
   ```

3. **User Interaction**
   ```
   User Query → Topic Selection → Article Fetching → Summary Display
   ```

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd global-news-tracker
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Usage

1. **Access the application** at `http://localhost:8501`
2. **Select news source** (Google News or Custom Query)
3. **Choose trending topic** or enter custom search query
4. **Configure settings** (number of articles, auto-refresh)
5. **View results** including article list and AI summary

### Configuration Options

- **Number of Articles**: 5-20 articles per search
- **Auto-refresh**: 30-second intervals
- **Model Selection**: BART-large-cnn for summarization
- **Search Scope**: Global news with English language focus

### Future Enhancements

1. **Multi-language Support**: Support for multiple languages
2. **Sentiment Analysis**: Add sentiment analysis for news topics
3. **Historical Tracking**: Store and track topic trends over time
4. **Custom Sources**: Support for additional news sources
5. **API Integration**: REST API for programmatic access
6. **Mobile App**: Native mobile application
7. **Push Notifications**: Real-time alerts for breaking news

### Limitations

1. **Rate Limiting**: Google News may implement rate limiting
2. **Content Structure**: Relies on Google News HTML structure
3. **Model Limitations**: Summarization quality depends on model capabilities
4. **Network Dependency**: Requires stable internet connection
5. **Legal Compliance**: Must comply with website terms of service

### Troubleshooting

**Common Issues:**

1. **No articles found**: Check internet connection and try different search terms
2. **Scraping errors**: Google News structure may have changed
3. **Model loading issues**: Ensure sufficient system resources
4. **Memory issues**: Reduce number of articles for processing

**Solutions:**

- Verify network connectivity
- Update dependencies regularly
- Monitor system resources
- Check Google News accessibility

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Built with ❤️ using Streamlit, Transformers, and BeautifulSoup**



