"""Dense (embedding-based) track retrieval using sentence-transformers + FAISS."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class DenseRetriever:
    """Dense retrieval using bi-encoder embeddings + FAISS ANN index.

    Uses sentence-transformers to encode tracks and queries,
    then retrieves via approximate nearest neighbor search.
    """

    def __init__(
        self,
        track_df: pd.DataFrame,
        model_name: str = "all-MiniLM-L6-v2",
        text_fields: list[str] | None = None,
        index_path: Optional[Path] = None,
    ):
        """
        Args:
            track_df: DataFrame with track metadata
            model_name: HuggingFace sentence-transformer model name
            text_fields: Columns to concatenate for document text
            index_path: Path to save/load pre-built FAISS index
        """
        self.track_df = track_df.reset_index(drop=True)
        self.model_name = model_name
        self.text_fields = text_fields or ["track_name", "artist_name", "tags"]
        self.index_path = index_path
        self.model = None
        self.faiss_index = None
        self.embeddings: np.ndarray | None = None

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading sentence-transformer: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

    def _get_doc_texts(self) -> list[str]:
        docs = []
        for _, row in self.track_df.iterrows():
            text = " ".join(str(row.get(f, "")) for f in self.text_fields)
            docs.append(text)
        return docs

    def build_index(self, batch_size: int = 512):
        """Encode all tracks and build FAISS flat index."""
        import faiss

        if self.model is None:
            self._load_model()

        docs = self._get_doc_texts()
        logger.info(f"Encoding {len(docs)} tracks...")
        self.embeddings = self.model.encode(
            docs, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # Inner product (cosine on normalized vecs)
        self.faiss_index.add(self.embeddings.astype(np.float32))
        logger.success(f"FAISS index built: {self.faiss_index.ntotal} vectors, dim={dim}")

        if self.index_path:
            self._save_index()

    def _save_index(self):
        import faiss
        path = Path(self.index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(path))
        np.save(str(path.with_suffix(".npy")), self.embeddings)
        logger.info(f"Index saved to {path}")

    def load_index(self):
        import faiss
        path = Path(self.index_path)
        self.faiss_index = faiss.read_index(str(path))
        self.embeddings = np.load(str(path.with_suffix(".npy")))
        logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve top-k tracks for a query using dense similarity."""
        if self.model is None:
            self._load_model()
        if self.faiss_index is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        q_emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.faiss_index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            row = self.track_df.iloc[idx]
            results.append({
                "track_id": row.get("track_id"),
                "score": float(score),
                "track_name": row.get("track_name", ""),
                "artist_name": row.get("artist_name", ""),
                "tags": row.get("tags", ""),
            })
        return results
