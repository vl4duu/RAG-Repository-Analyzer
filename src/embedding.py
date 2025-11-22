import os
from typing import List

import numpy as np
from dotenv import load_dotenv

# Optional heavy deps
try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    import torch  # type: ignore

    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    torch = None  # type: ignore

try:
    import openai  # type: ignore
except Exception:
    openai = None  # type: ignore

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if openai is not None and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY


def embed_textual_metadata(content: str) -> List[float]:
    """
    Create an embedding for textual content. Prefers OpenAI, falls back to local hash embedding.
    Returns a 1D Python list[float] suitable for vector DBs.
    """
    # Prefer OpenAI if available and key is set
    if openai is not None and OPENAI_API_KEY:
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=content,
            )
            embedding = response.data[0].embedding
            return list(embedding)
        except Exception as e:
            # Fall through to local embedding in degraded mode
            print(f"OpenAI embedding failed, using fallback. Reason: {e}")
    # Fallback: deterministic local embedding
    return _fallback_embed(content)


def _fallback_embed(text: str, dim: int = 384) -> List[float]:
    """
    Lightweight, deterministic embedding for offline/degraded mode.
    Uses hashing over tiktoken-like bytepieces to create a fixed-size vector.
    """
    # Simple fast hashing over bytes
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec.tolist()
    # Use bytes to be language-agnostic
    b = text.encode("utf-8", errors="ignore")
    # Sliding window hashing
    h = 0
    for i, by in enumerate(b):
        h = (h * 131 + by) & 0xFFFFFFFF
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()



# # Embed a README or docstring
# textual_metadata = "This function calculates the sum of two numbers."
# metadata_embedding = embed_textual_metadata(textual_metadata)

# Load the CodeBERT tokenizer and model if available
_tokenizer = None
_model = None
if HF_AVAILABLE and os.getenv("DISABLE_HF", "0") != "1":
    try:
        _tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        _model = AutoModel.from_pretrained("microsoft/codebert-base")
    except Exception as e:
        # Degraded mode if models cannot be downloaded
        print(f"Warning: Failed to load CodeBERT models, using fallback code embeddings. Reason: {e}")
        _tokenizer = None
        _model = None


# Function to generate embeddings for a code snippet
def generate_code_embedding(code_snippet: str) -> List[float]:
    """
    Generate a code-aware embedding. Uses CodeBERT if available, otherwise a local fallback.
    Returns a 1D Python list[float].
    """
    if _tokenizer is not None and _model is not None and HF_AVAILABLE:
        try:
            inputs = _tokenizer(
                code_snippet,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():  # type: ignore
                outputs = _model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            return embedding.detach().cpu().numpy().tolist()  # type: ignore
        except Exception as e:
            print(f"CodeBERT embedding failed, using fallback. Reason: {e}")
    # Fallback: use the same local embedding, but with different dimension to reduce collision with text
    return _fallback_embed(code_snippet, dim=512)
