"""Load the raw corpus listed in manifest.json, chunk it, embed it, and index it in Chroma.

Usage:
    python -m src.ingest [--chunk-tokens 500] [--overlap 50] [--collection banking_baseline]
"""
import argparse
import json
import time

import chromadb
import tiktoken
from pypdf import PdfReader

from src import config

ENC = tiktoken.get_encoding("cl100k_base")
EMBED_BATCH = 50


def load_text(entry: dict) -> str:
    path = config.RAW_DIR / entry["file"]
    if path.suffix == ".pdf":
        # Page text is joined as-is, so headers/footers and mid-sentence page breaks survive (intentional).
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    tokens = ENC.encode(text)
    step = size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(ENC.decode(tokens[start:start + size]))
        if start + size >= len(tokens):
            break
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    client = config.google_client()
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
                out.extend(d.embedding for d in resp.data)
                break
            except Exception as e:  # rate limit / transient
                if attempt == 4:
                    raise
                wait = 2 ** attempt * 2
                print(f"  embed retry in {wait}s ({type(e).__name__})")
                time.sleep(wait)
    return out


def collection_exists(name: str) -> bool:
    return name in [c.name for c in chromadb.PersistentClient(path=str(config.CHROMA_DIR)).list_collections()]


def build_index(chunk_tokens: int = config.CHUNK_TOKENS, overlap: int = config.CHUNK_OVERLAP,
                collection: str = config.COLLECTION) -> None:
    """Chunk the manifest corpus, embed it, and (re)create the Chroma collection."""
    manifest = json.loads(config.MANIFEST.read_text())
    ids, docs, metas = [], [], []
    for entry in manifest:
        chunks = chunk_text(load_text(entry), chunk_tokens, overlap)
        print(f"{entry['doc_id']:28s} {len(chunks):3d} chunks")
        for i, chunk in enumerate(chunks):
            ids.append(f"{entry['doc_id']}::{i}")
            docs.append(chunk)
            metas.append({
                "doc_id": entry["doc_id"],
                "title": entry["title"],
                "version": entry["version"],
                "effective_date": entry["effective_date"] or "",
                "status": entry["status"],
                "doc_group": entry["doc_group"],
                "chunk_index": i,
            })

    print(f"embedding {len(docs)} chunks with {config.EMBEDDING_MODEL} ...")
    vectors = embed(docs)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    col = client.create_collection(collection, metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=vectors)
    print(f"indexed {col.count()} chunks into '{collection}'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-tokens", type=int, default=config.CHUNK_TOKENS)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--collection", default=config.COLLECTION)
    args = ap.parse_args()
    build_index(args.chunk_tokens, args.overlap, args.collection)


if __name__ == "__main__":
    main()
