"""Shared sentence-transformer embedder for agent memory.

Uses the lightweight `all-MiniLM-L6-v2` model (384-dim output) so that
semantically similar round states map to nearby vectors in FAISS, instead
of the previous SHA-256 hashing approach which destroyed similarity.
"""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Model name — small, fast, 384-dimensional output
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384

# Singleton instance so both agents share one loaded model
_model = None


def _get_model():
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceTransformer model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info("SentenceTransformer model loaded")
    return _model


def embed(data: dict) -> np.ndarray:
    """Convert an arbitrary dict into a 384-dim float32 vector.

    The dict is serialised to a deterministic JSON string and then
    encoded by the sentence-transformer.  This preserves semantic
    similarity: contexts with similar accuracies, strategies, and
    detection outcomes will produce nearby vectors.

    Args:
        data: Any JSON-serialisable dictionary (round context, outcome, etc.)

    Returns:
        numpy float32 array of shape (384,)
    """
    text = json.dumps(data, sort_keys=True, default=str)
    model = _get_model()
    logger.info(f"Generating semantic embedding for state vector: {text}")
    vec = model.encode(text, convert_to_numpy=True)
    return vec.astype(np.float32).flatten()


def get_dimension() -> int:
    """Return the embedding dimension (for VectorStore initialisation)."""
    return _EMBEDDING_DIM
