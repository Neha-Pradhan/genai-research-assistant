from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Load existing vectorstore
CHROMA_DIR = "data/chroma_db"

def load_vectorstore():
    embeddings = OllamaEmbeddings(model="llama3.2")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

@tool
def search_papers(query: str) -> str:
    """Search through the research papers to answer questions about their content."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant content found in the papers."
    
    results = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        results.append(f"Source: {source}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(results)

@tool
def compare_papers(topic: str) -> str:
    """Compare what different papers say about a specific topic."""
    docs = retriever.invoke(topic)
    if not docs:
        return "No relevant content found to compare."
    
    # Group by source paper
    papers = {}
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in papers:
            papers[source] = []
        papers[source].append(doc.page_content)
    
    # Build comparison
    comparison = f"Comparison on topic: '{topic}'\n\n"
    for source, contents in papers.items():
        paper_name = source.split("/")[-1]
        comparison += f"**{paper_name}:**\n"
        comparison += " ".join(contents[:2])
        comparison += "\n\n"
    
    return comparison