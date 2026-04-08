# GenAI Research Assistant

An agentic RAG system that answers questions about AI research papers 
and compares findings across them.

## What it does
- **Search papers** — retrieves relevant content from uploaded research papers
- **Compare papers** — structured comparison of how different papers 
approach the same topic
- **Grounded answers** — system prompt prevents hallucination by anchoring 
responses to retrieved content only

## Papers included
- Attention is All You Need (Vaswani et al., 2017)
- Retrieval Augmented Generation (Lewis et al., 2020)
- RAGAS: Automated Evaluation of RAG (Es et al., 2023)

## Tech stack
- LangGraph — agentic orchestration
- ChromaDB — local vector store
- Ollama + Llama3.2 — local LLM and embeddings
- Streamlit — chat UI

## Production considerations
- Replace ChromaDB local with hosted vector DB (Pinecone, Weaviate) 
for concurrent users
- Replace Ollama with cloud LLM for scale
- Add RAGAS eval suite for faithfulness and relevance monitoring
- Containerise with Docker for deployment

## Example questions to try

**Search:**
- "What is the multi-head attention mechanism?"
- "How does RAGAS measure faithfulness?"
- "What is the role of the retriever in RAG?"

**Compare:**
- "Compare how the attention paper and RAG paper use embeddings"
- "How do the attention paper and RAGAS paper differ in their approach 
to evaluation?"

## Run locally
```bash
pip install -r requirements.txt
python app/ingest.py
streamlit run streamlit_app.py
```

## Known limitations
- PyPDFLoader loses tables and diagrams
- Local Ollama suitable for single user only
- Hallucination observed on comparison queries where papers don't share 
  common topics — LLM fabricates connections. Mitigated partially via 
  system prompt. Full fix requires RAGAS faithfulness eval (see Project 6)