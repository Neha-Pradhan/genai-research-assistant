# NOTE: PyPDFLoader extracts text only. 
# Limitation: Diagrams are ignored, tables may lose structure.
# Production alternatives: 
#   - PDFPlumber or Unstructured for tables
#   - PyMuPDF to extract images + vision model (LLaVA/GPT-4V) for diagrams
#   - LlamaParse for handling both tables and diagrams natively

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import os

# Paths
PAPERS_DIR = "data/papers"
CHROMA_DIR = "data/chroma_db"

def load_and_chunk_papers():
    """Load all PDFs from papers directory and split into chunks."""
    all_docs = []
    
    for filename in os.listdir(PAPERS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PAPERS_DIR, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            all_docs.extend(docs)
    
    print(f"Total pages loaded: {len(all_docs)}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vectorstore(chunks):
    """Store chunks in ChromaDB with Ollama embeddings."""
    print("Creating vector store...")
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("Vector store created successfully!")
    return vectorstore

if __name__ == "__main__":
    chunks = load_and_chunk_papers()
    create_vectorstore(chunks)