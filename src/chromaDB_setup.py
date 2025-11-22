import os
import chromadb
from typing import Dict, List
from chromadb.config import Settings
import mmh3


def _disable_chroma_telemetry():
    """Disable Chroma/PostHog telemetry to avoid noisy errors and network calls.

    We see errors like:
      "Failed to send telemetry event ... capture() takes 1 positional argument but 3 were given"

    These originate from optional telemetry (PostHog) paths in chromadb. Some environments
    may have an incompatible `posthog` installed or blocked network, causing noisy logs.

    This helper:
      - Sets env flags recognized by Chroma to disable telemetry
      - Keeps client Settings with anonymized_telemetry=False
      - Defensively monkey‑patches posthog.capture to a no‑op (accepts *args, **kwargs)

    Telemetry can be re‑enabled by setting DISABLE_CHROMA_TELEMETRY=0.
    """
    disable = os.getenv("DISABLE_CHROMA_TELEMETRY", "1").lower() not in {"0", "false", "no"}
    if not disable:
        return

    # Env flags recognized by different chroma versions
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("CHROMA_TELEMETRY_IMPLEMENTATION", "disabled")
    os.environ.setdefault("CHROMADB_TELEMETRY_IMPLEMENTATION", "disabled")

    # Best‑effort defensive monkey‑patch in case posthog is importable
    try:
        import posthog  # type: ignore

        def _noop_capture(*args, **kwargs):  # accepts any signature
            return None

        if hasattr(posthog, "capture"):
            posthog.capture = _noop_capture  # type: ignore
        # If a client instance is created somewhere, try to no‑op that too
        if hasattr(posthog, "Posthog"):
            try:
                ph = posthog.Posthog()  # type: ignore
                setattr(ph, "capture", _noop_capture)
            except Exception:
                pass
    except Exception:
        # If posthog isn't installed or anything goes wrong, silently ignore
        pass


def setup_chroma_collections(chunked_docs: Dict, embedded_chunks: Dict, batch_size: int = 1000):
    """Create/update persistent ChromaDB collections for textual and code chunks.

    Args:
        chunked_docs: Dict with keys 'textual_chunks' and 'code_chunks'.
        embedded_chunks: Dict with keys 'textual_embeddings' and 'code_embeddings'.
        batch_size: Batch size for upserts.

    Returns:
        Dict with created collections {'textual_collection': ..., 'code_collection': ...}
    """

    # Disable telemetry first to avoid PostHog/capture noise across chroma versions
    _disable_chroma_telemetry()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", os.path.join(os.getcwd(), "data", "chroma"))
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    text_collection = client.get_or_create_collection(name="text_collection")
    code_collection = client.get_or_create_collection(name="code_collection")

    def stable_id(prefix: str, file_name: str, idx: int) -> str:
        base = f"{file_name}:{idx}"
        return f"{prefix}_{mmh3.hash128(base, signed=False)}"

    # Helper to add in batches and avoid duplicates
    def upsert_batch(collection, docs: List[Dict], embeddings: List[List[float]], prefix: str):
        for start in range(0, len(docs), batch_size):
            end = min(start + batch_size, len(docs))
            batch_docs = docs[start:end]
            batch_embs = embeddings[start:end]
            ids = [stable_id(prefix, d["file_name"], d.get("chunk_index", i + start)) for i, d in enumerate(batch_docs)]

            # Check which IDs already exist to avoid re-adding
            existing = set()
            try:
                existing_res = collection.get(ids=ids, include=["metadatas"])  # may raise if none exist
                existing = set(existing_res.get("ids", []) or [])
            except Exception:
                existing = set()

            # Filter new items
            new_items = [(i, d, e) for i, (d, e, id_) in enumerate(zip(batch_docs, batch_embs, ids)) if
                         id_ not in existing]
            if not new_items:
                continue

            new_ids = [ids[i] for i, _, _ in new_items]
            new_docs = [d["content"] for _, d, _ in new_items]
            new_metas = []
            for _, d, _ in new_items:
                meta = {"file_name": d["file_name"]}
                if prefix == "text":
                    meta.update({"content_type": "text"})
                else:
                    meta.update({"content_type": "code", "chunk_index": d.get("chunk_index")})
                new_metas.append(meta)
            new_embs = [[float(x) for x in e] for _, _, e in new_items]

            collection.add(ids=new_ids, documents=new_docs, metadatas=new_metas, embeddings=new_embs)

    # Prepare embeddings lists (ensure correct shapes)
    text_embeddings = embedded_chunks.get("textual_embeddings", [])
    code_embeddings = embedded_chunks.get("code_embeddings", [])

    upsert_batch(text_collection, chunked_docs.get("textual_chunks", []), text_embeddings, prefix="text")
    upsert_batch(code_collection, chunked_docs.get("code_chunks", []), code_embeddings, prefix="code")

    return {"textual_collection": text_collection, "code_collection": code_collection}
