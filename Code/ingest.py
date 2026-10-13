"""Ingestion: PDF -> page-aware chunks -> BGE embeddings -> ChromaDB.

Usage:  python Code/ingest.py Input_Data/your.pdf [another.pdf ...]
Re-running rebuilds the collection from scratch.
"""
import sys
from pathlib import Path

import config
import rag


def chunk_pdf(pdf_path):
    """Yield (id, text, metadata) for each chunk. Chunks never cross page boundaries,
    so every chunk has exactly one page number to cite."""
    import fitz  # PyMuPDF
    name = Path(pdf_path).name
    stem = Path(pdf_path).stem
    for pno, page in enumerate(fitz.open(pdf_path), start=1):
        words = page.get_text().split()
        for i in range(0, len(words), config.CHUNK_STEP):
            text = " ".join(words[i:i + config.CHUNK_WORDS])
            if len(text) < config.MIN_CHUNK_CHARS:
                continue
            yield f"{stem}_p{pno}_w{i}", text, {"page": pno, "source": name}


def ingest_pdfs(pdf_paths):
    client = rag.get_client()
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    for p in pdf_paths:
        for cid, text, meta in chunk_pdf(p):
            ids.append(cid); texts.append(text); metas.append(meta)
    if not ids:
        raise RuntimeError("No text extracted. The PDF may be scanned (needs OCR).")

    embs = rag.get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=True).tolist()
    for s in range(0, len(ids), 500):
        col.add(ids=ids[s:s + 500], documents=texts[s:s + 500],
                embeddings=embs[s:s + 500], metadatas=metas[s:s + 500])
    return len(ids)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python Code/ingest.py <file.pdf> [more.pdf ...]")
    n = ingest_pdfs(sys.argv[1:])
    print(f"Indexed {n} chunks into '{config.COLLECTION}' at {config.CHROMA_DIR}")
