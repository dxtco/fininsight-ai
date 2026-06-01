# app.py
import streamlit as st
import joblib
import os
from ml_pipeline import clean_text
from rag_pipeline import initialize_rag_system

# Page configuration
st.set_page_config(page_title="FinInsight AI Dashboard", page_icon="📈", layout="wide")

# Cache resources so they only load once when the app starts
@st.cache_resource
def load_ml_models():
    try:
        model = joblib.load('random_forest_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Pre-trained ML models not found. Please run 'python ml_pipeline.py' first!")
        return None, None

@st.cache_resource
def load_rag_system():
    try:
        return initialize_rag_system()
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {e}")
        return None

# Load our systems
rf_model, tfidf_vectorizer = load_ml_models()
rag_chain = load_rag_system()

# --- UI Layout ---
st.title("📈 FinInsight AI – Market Intelligence Platform")
st.markdown("---")

# Setup Sidebar for Traditional ML Predictor
st.sidebar.header("🔮 Market Sentiment Predictor")
st.sidebar.markdown("Uses an ensemble **Random Forest Classifier** (90.44% Accuracy) to predict news sentiment trends.")

user_headline = st.sidebar.text_area("Enter a Financial Headline:", "Tesla shares surge after breakout quarter with massive delivery numbers.")

if st.sidebar.button("Analyze Sentiment"):
    if rf_model and tfidf_vectorizer:
        cleaned = clean_text(user_headline)
        vectorized = tfidf_vectorizer.transform([cleaned])
        prediction = rf_model.predict(vectorized)[0]
        
        if prediction == 1:
            st.sidebar.success("🚀 Result: BULLISH (Market Trend: UP)")
        else:
            st.sidebar.error("📉 Result: BEARISH (Market Trend: DOWN)")
    else:
        st.sidebar.warning("Model dependencies are not fully loaded.")

# Main Panel for LangChain RAG Chat
st.header("💬 AI Financial Document Assistant (RAG)")
st.markdown("Ask natural language questions based on the embedded corporate documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input for Chat
if user_query := st.chat_input("Ask something about the 2026 reports (e.g., 'What happened with Nvidia?'):"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("Analyzing knowledge base..."):
                try:
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Inference error: {e}")
        else:
            st.error("RAG pipeline connection is offline.")