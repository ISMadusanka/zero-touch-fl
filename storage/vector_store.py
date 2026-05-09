"""FAISS vector store with disk persistence for agent memory."""

import json
import os
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorStore:
    """In-memory FAISS index backed by JSON metadata on disk.

    If FAISS is unavailable, falls back to brute-force numpy search.
    """

    def __init__(self, dimension: int = 64, persist_path: str = None):
        self.dimension = dimension
        self.persist_path = persist_path
        self.metadata: list[dict] = []

        if faiss:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self._vectors = []  # fallback

        if persist_path:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray, meta: dict):
        """Add a vector with associated metadata."""
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        if faiss:
            self.index.add(vec)
        else:
            self._vectors.append(vec.flatten())
        self.metadata.append(meta)

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        """Return top-k most similar metadata entries."""
        if len(self.metadata) == 0:
            return []
        k = min(k, len(self.metadata))
        q = np.array(query, dtype=np.float32).reshape(1, -1)

        if faiss:
            _, indices = self.index.search(q, k)
            return [self.metadata[i] for i in indices[0] if i < len(self.metadata)]
        else:
            # Brute-force fallback
            vecs = np.array(self._vectors)
            dists = np.linalg.norm(vecs - q, axis=1)
            top_k = np.argsort(dists)[:k]
            return [self.metadata[i] for i in top_k]

    def save(self):
        """Persist index + metadata to disk."""
        if not self.persist_path:
            return
        os.makedirs(self.persist_path, exist_ok=True)
        # Save metadata
        with open(os.path.join(self.persist_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)
        # Save vectors
        if faiss:
            faiss.write_index(self.index, os.path.join(self.persist_path, "index.faiss"))
        else:
            np.save(
                os.path.join(self.persist_path, "vectors.npy"),
                np.array(self._vectors) if self._vectors else np.empty((0, self.dimension)),
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self):
        if not self.persist_path:
            return
        meta_path = os.path.join(self.persist_path, "metadata.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            self.metadata = json.load(f)
        if faiss:
            idx_path = os.path.join(self.persist_path, "index.faiss")
            if os.path.exists(idx_path):
                self.index = faiss.read_index(idx_path)
        else:
            vec_path = os.path.join(self.persist_path, "vectors.npy")
            if os.path.exists(vec_path):
                arr = np.load(vec_path)
                self._vectors = [arr[i] for i in range(len(arr))]
