# Interview Q&A — Project 2: GenAI Research Assistant

## Chunking

**Q: Why PyPDFLoader?**
Simple and reliable for clean text-based PDFs like research papers. 
Alternatives: PDFPlumber for tables, PyMuPDF to extract images, 
Unstructured for messy layouts, LlamaParse for both tables and diagrams.

**Q: What about diagrams and tables in PDFs?**
PyPDFLoader loses both. Production fix: PDFPlumber for tables, 
PyMuPDF + vision model (LLaVA/GPT-4V) for diagrams, LlamaParse handles both natively.

**Q: Why RecursiveCharacterTextSplitter?**
Splits on natural boundaries — paragraphs first, then sentences, then words. 
Preserves meaning better than CharacterTextSplitter.
Alternatives: TokenTextSplitter for LLM context limits, 
SemanticChunker for meaning-based splits (expensive but best quality).

**Q: What is chunk overlap and why 50?**
Overlap prevents context loss at chunk boundaries — a sentence split 
across two chunks loses its meaning. 50 is 10% of chunk_size=500, 
a reasonable default. In production tune this by evaluating retrieval quality.

**Q: How do you evaluate retrieval quality?**
- Hit Rate — did correct chunk appear in top K?
- MRR — where did it rank?
- RAGAS Context Recall and Precision
- Golden dataset with known Q&A pairs
- Ablation study varying chunk size and overlap

## Retrieval

**Q: What similarity metric does ChromaDB use?**
Cosine similarity — measures angle between embedding vectors. 
Closer to 1 means more similar. Good for text because length-independent.

**Q: What retrieval strategies exist beyond top-k?**
- MMR (Maximal Marginal Relevance) — balances relevance AND diversity, 
  avoids returning chunks that all say the same thing
- Threshold based — only return chunks above a similarity score
- Hybrid search — combines semantic search with keyword search (BM25)

## LangSmith & Observability

**Q: How do you debug your agent in production?**
Use LangSmith tracing — every run shows exactly which tools were called, 
latency at each step, token usage, and full input/output. 
Pinpoint failures without guesswork.

**Q: How do you handle wrong tool selection?**
Tool selection depends on how well tool docstrings describe when to use them. 
Vague docstrings lead to wrong tool selection. Refine docstrings iteratively 
based on LangSmith traces showing misroutes.

**Q: What happens when retrieval returns empty?**
Without guardrails LLM will hallucinate rather than admit it doesn't know. 
Fix: instruct LLM in system prompt to acknowledge when context is insufficient. 
This is called graceful degradation.

## RAGAS Evaluation

**Q: What do your eval results show?**
Answer relevancy scored 0.42 — expected with small local model like Llama3.2. 
Faithfulness returned nan because retrieved contexts were not separately 
captured — contexts were approximated with final answer which doesn't give 
RAGAS enough information to measure faithfulness accurately.

**Q: What other RAGAS metrics would you add?**
- Context Recall — did we retrieve everything needed to answer?
- Context Precision — was retrieved content relevant or noisy?
- Answer Correctness — how close is answer to ground truth?
These require properly captured contexts, not just the final answer.

**Q: How would you improve the eval score?**
- Capture actual retrieved chunks separately from final answer
- Use stronger model for evaluation (GPT-4 or Claude)
- Refine tool docstrings to reduce wrong tool selection
- Fix graceful degradation so empty retrieval doesn't lead to hallucination