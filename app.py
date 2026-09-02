import streamlit as st
import yfinance as yf
import pandas as pd
from transformers import pipeline

# page config
st.set_page_config(page_title="Financial Sentiment Dashboard", layout="wide")

# embedded custom CSS
st.markdown("""
<style>
    /* background */
    .stApp {
        background-color: #0b132b;
        color: #e0e1dd;
    }
    
    /* headers */
    h1, h2, h3, .stSubheader {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    
    /* input box */
    div[data-baseweb="input"] {
        border: 1px solid #10b981 !important;
        background-color: #1c2541 !important;
        border-radius: 8px;
    }
    
    /* metric cards */
    div[data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-weight: 800;
    }
    
    /* diver lines */
    hr {
        border-color: #1c2541;
    }
</style>
""", unsafe_allow_html=True)

# app setup
st.title("Financial News Sentiment Dashboard")
st.caption("Real-Time NLP Pipeline Powered by FinBERT & Transformers")

@st.cache_resource
def load_sentiment_pipeline():
    # downloads and caches FinBERT locally on Streamlit servers for free
    return pipeline("text-classification", model="ProsusAI/finbert")

with st.spinner("Loading FinBERT Model..."):
    nlp = load_sentiment_pipeline()

ticker = st.text_input("Stock Ticker Symbol:", value="NVDA").upper().strip()

if ticker:
    st.subheader(f"Live Market Sentiment: {ticker}")
    stock = yf.Ticker(ticker)
    raw_news = stock.news
    
    if raw_news:
        headlines = []
        for item in raw_news[:10]:
            title = item.get('title')
            publisher = item.get('publisher')

            # fallback for yfinance structure variations
            if not title and 'content' in item:
                title = item['content'].get('title')
                publisher = item['content'].get('provider', {}).get('displayName', '')
                
            if title:
                headlines.append({'title': title, 'publisher': publisher or 'N/A'})
        
        if headlines:
            results = []
            for item in headlines:
                sentiment = nlp(item['title'])[0]
                results.append({
                    "Headline": item['title'],
                    "Publisher": item['publisher'],
                    "Sentiment": sentiment['label'].title(),
                    "Confidence Score": f"{sentiment['score']:.1%}"
                })
            
            df = pd.DataFrame(results)
            
            counts = df['Sentiment'].value_counts()
            
            # display sentiment metrics in styled columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive Headlines", counts.get("Positive", 0))
            col2.metric("Neutral Headlines", counts.get("Neutral", 0))
            col3.metric("Negative Headlines", counts.get("Negative", 0))
            
            st.divider()
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No headline text found for this ticker.")
    else:
        st.error(f"Could not pull news data for '{ticker}'. Check the symbol and try again.")
