# Synopsis
Title: AUTOSAR HLD Document Analysis Assistant (local RAG)
Problem: HLD documents are long unstructured PDFs; manual review is slow and error-prone.
Solution: Local RAG (PyMuPDF, BGE-small, ChromaDB, Llama 3.2 via Ollama, Streamlit) giving page-cited, grounded answers, refusal when evidence is missing, component extraction, and human review with audit log.
Expected outcome: Faster architecture understanding with verifiable answers.
