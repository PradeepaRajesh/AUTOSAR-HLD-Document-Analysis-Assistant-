"""Central configuration. Every tunable value lives here so the report can cite it."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE / "Input_Data"
CHROMA_DIR = str(BASE / "chroma_db")
EVAL_DIR = BASE / "Evaluation_Results"
AUDIT_LOG = EVAL_DIR / "audit_log.csv"

# ---- Retrieval ----
COLLECTION = "hld"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_WORDS = 300          # words per chunk
CHUNK_STEP = 250           # => 50-word overlap
MIN_CHUNK_CHARS = 100
TOP_K = 5
THRESHOLD = 0.55           # cosine distance above this => "not in document" (tune with evaluate.py)
HIGH_CONF = 0.35           # cosine distance below this => high retrieval confidence

# ---- LLM ----
# LLM_BACKEND = "ollama" (local, default) or "api" (any OpenAI-compatible endpoint, e.g. Groq / Gemini)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
TEMPERATURE = 0.1
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
API_BASE = os.getenv("API_BASE", "https://api.groq.com/openai/v1")
API_MODEL = os.getenv("API_MODEL", "llama-3.1-8b-instant")
API_KEY = os.getenv("API_KEY", "")
