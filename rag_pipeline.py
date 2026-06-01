# rag_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

def format_docs(docs):
    """Formats retrieved documents into a single block of text."""
    return "\n\n".join(doc.page_content for doc in docs)

    # rag_pipeline.py (partial update - top of initialize_rag_system)
def initialize_rag_system():
    # Check Streamlit Cloud secrets first, fallback to local os.getenv
    import streamlit as st
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY is missing completely from secrets and environment!")

    # 1. Connect to the existing local ChromaDB store using real Google Embeddings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", 
        google_api_key=api_key
    )
    
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
    # Pull 4 chunks for a richer context window
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # ... Rest of your code down to llm, prompt, and rag_chain stays exactly the same!

    # 2. Configure our Generative LLM via Google Gemini API
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1,
        google_api_key=api_key
    )

    # 3. Create a strict Prompt Template
    system_prompt = (
        "You are an expert financial AI assistant. Use the following pieces of retrieved context "
        "to answer the question. If you do not know the answer based on the context, say exactly "
        "'I cannot find that information in the provided reports.' Do not make up information.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)

    # 4. Assemble the RAG Chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    print("🚀 Testing FinInsight LCEL RAG Engine pipeline via Gemini...")
    try:
        chain = initialize_rag_system()
        query = "What drove the tech sector growth and what happened with Nvidia?"
        print(f"\nUser Query: {query}")
        
        response = chain.invoke(query)
        print("\n🤖 AI Response:")
        print(response)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
