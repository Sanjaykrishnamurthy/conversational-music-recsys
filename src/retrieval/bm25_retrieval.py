"""BM25-based track retrieval from conversation context."""

from typing import Optional

import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25Retriever:
    """BM25 retrieval over music track metadata.

    Builds a BM25 index over track text fields (name + artist + tags)
    and retrieves top-k tracks given a dialogue context query.
    """

    def __init__(self, track_df: pd.DataFrame, text_fields: list[str] | None = None):
        """
        Args:
            track_df: DataFrame with track metadata
            text_fields: Columns to concatenate for BM25 document.
                         Defaults to ['track_name', 'artist_name', 'tags']
        """
        self.track_df = track_df.reset_index(drop=True)
        self.text_fields = text_fields or ["track_name", "artist_name", "tags"]
        self.index: BM25Okapi | None = None
        self._build_index()

    def _build_index(self):
        """Tokenize track text and build BM25 index."""
        logger.info(f"Building BM25 index over {len(self.track_df)} tracks...")
        corpus = []
        for _, row in tqdm(self.track_df.iterrows(), total=len(self.track_df), desc="Indexing"):
            text = " ".join(
                str(row.get(field, "")) for field in self.text_fields
            ).lower()
            corpus.append(text.split())
        self.index = BM25Okapi(corpus)
        logger.success("BM25 index built.")

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve top-k tracks for a given query string.

        Args:
            query: Natural language query (from dialogue context)
            top_k: Number of tracks to return

        Returns:
            List of dicts with track_id, score, and metadata
        """
        if self.index is None:
            raise RuntimeError("BM25 index not built. Call _build_index() first.")

        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            row = self.track_df.iloc[idx]
            results.append({
                "track_id": row.get("track_id"),
                "score": float(scores[idx]),
                "track_name": row.get("track_name", ""),
                "artist_name": row.get("artist_name", ""),
                "tags": row.get("tags", ""),
            })
        return results

    def retrieve_batch(
        self, queries: list[str], top_k: int = 20
    ) -> list[list[dict]]:
        """Retrieve top-k tracks for a batch of queries."""
        return [self.retrieve(q, top_k=top_k) for q in tqdm(queries, desc="BM25 retrieval")]
