# vector_store_pipeline.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

def create_vector_db():
    print("Step 1: Loading raw unstructured document...")
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()

    print("Step 2: Splitting text into semantic chunks...")
    # Increased chunk size to keep financial context intact
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} distinct chunks.")

    print("Step 3: Initializing production Google Embeddings...")
    # Using the standard general-availability text embedding string
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    print("Step 4: Creating and indexing Vector Database in ChromaDB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    print("✅ ChromaDB vector store successfully rebuilt with real Google Embeddings!")
    return vector_db

if __name__ == "__main__":
    create_vector_db()