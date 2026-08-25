"""
tgpt_backend.embeddings
========================

A tiny embedding shim used by the generative-agents simulation.

Why?
----
The original simulation uses ``openai.Embedding.create`` (model
``text-embedding-ada-002``) for memory retrieval. The free ``tgpt`` /
``pytgpt`` stacks do not yet expose a free embedding endpoint, so by
default we provide a deterministic hash-based pseudo-embedding that
captures bag-of-words overlap well enough for the simulation to run
end-to-end and produce meaningful-looking behaviour.

If you want a real semantic embedding, drop in something like
``sentence-transformers`` in :func:`real_embedding` and have
:func:`get_embedding` dispatch to it.
"""
from __future__ import annotations

import hashlib
import math
from typing import List

# The dimensionality matches OpenAI's text-embedding-ada-002 so the
# downstream cosine-similarity code in
# ``persona/cognitive_modules/retrieve.py`` keeps working unchanged.
EMBED_DIM = 1536


def _tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters."""
    import re
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def hash_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Deterministic, bag-of-words-hashed embedding.

    Each unique token contributes (deterministically) to ``dim`` random
    buckets via a salted hash. The resulting vector is L2-normalised so
    cosine similarity is well-defined.
    """
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        # Stable, non-zero default for empty input.
        return [0.0] * dim

    for token in tokens:
        # A 16-byte digest gives us plenty of unique bit patterns.
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Use 8-byte chunks to derive two ints per digest — enough to
        # cover the whole vector without looping.
        for i in range(0, len(digest), 8):
            chunk = digest[i:i + 8]
            if len(chunk) < 8:
                chunk = chunk + b"\x00" * (8 - len(chunk))
            (a,) = int.from_bytes(chunk[:4], "big", ),
            b = int.from_bytes(chunk[4:8], "big")
            idx = a % dim
            sign = 1.0 if (b & 1) else -1.0
            vec[idx] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def real_embedding(text: str) -> List[float]:
    """Hook for a real embedding model.

    Replace the body of this function (or monkey-patch it in your own
    code) to use ``sentence-transformers``, a local FastAPI embedding
    server, etc. The vector should be 1-D and ideally L2-normalised.
    """
    raise NotImplementedError(
        "Configure tgpt_backend.embeddings.real_embedding to use a real "
        "embedding model (e.g. sentence-transformers, FastAPI, etc.)."
    )


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Dispatch helper. Tries the real embedding first, then the hash
    fallback. ``model`` is accepted for API compatibility.
    """
    try:
        return real_embedding(text)
    except NotImplementedError:
        return hash_embedding(text)
