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