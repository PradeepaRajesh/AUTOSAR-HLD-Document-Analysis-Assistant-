# Model, prompts and configuration
- All parameters: `Code/config.py` (chunk size, overlap, top-k, threshold, models, temperature)
- System prompt and extraction prompt: `Code/rag.py` (SYSTEM_PROMPT, EXTRACT_PROMPT)
- LLM: Llama 3.2 3B via Ollama (local). Optional: any OpenAI-compatible API via env vars LLM_BACKEND=api, API_BASE, API_MODEL, API_KEY
- Embeddings: BAAI/bge-small-en-v1.5 (with BGE query prefix)
- Vector DB: ChromaDB (cosine, persistent, local)
