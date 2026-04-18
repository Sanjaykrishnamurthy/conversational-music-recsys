"""Official evaluation metrics for RecSys 2026 Music-CRS challenge.

Metrics:
  - nDCG@{1, 10, 20} — macro-averaged over sessions and turns
  - Catalog Coverage  — diversity of recommended tracks
  - Lexical Diversity — response text diversity (distinct-n)
"""

import math
from collections import Counter
from typing import Optional

import numpy as np


# ── Retrieval Metrics ────────────────────────────────────────────────────────


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    relevances = relevances[:k]
    return sum(
        rel / math.log2(rank + 2)
        for rank, rel in enumerate(relevances)
    )


def ndcg_at_k(recommended: list[str], ground_truth: list[str], k: int) -> float:
    """nDCG@k for a single query.

    Args:
        recommended: Ordered list of recommended track_ids
        ground_truth: List of relevant track_ids (unordered)
        k: Cutoff rank

    Returns:
        nDCG@k score in [0, 1]
    """
    gt_set = set(ground_truth)
    relevances = [1 if tid in gt_set else 0 for tid in recommended]
    ideal = sorted(relevances, reverse=True)

    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)

    return dcg / idcg if idcg > 0 else 0.0


def compute_ndcg(
    predictions: dict[str, list[list[str]]],
    ground_truths: dict[str, list[list[str]]],
    k_values: list[int] = [1, 10, 20],
) -> dict[str, float]:
    """Compute macro-averaged nDCG@k over all sessions and turns.

    Args:
        predictions: {session_id: [turn_1_recs, turn_2_recs, ...]}
                     Each inner list is ordered recommended track_ids
        ground_truths: {session_id: [turn_1_gt, turn_2_gt, ...]}
                       Each inner list is relevant track_ids
        k_values: List of cutoffs to evaluate

    Returns:
        Dict like {'nDCG@1': 0.xx, 'nDCG@10': 0.xx, 'nDCG@20': 0.xx}
    """
    scores: dict[str, list[float]] = {f"nDCG@{k}": [] for k in k_values}

    for session_id, session_preds in predictions.items():
        session_gts = ground_truths.get(session_id, [])
        for turn_idx, turn_preds in enumerate(session_preds):
            if turn_idx >= len(session_gts):
                continue
            turn_gt = session_gts[turn_idx]
            for k in k_values:
                score = ndcg_at_k(turn_preds, turn_gt, k)
                scores[f"nDCG@{k}"].append(score)

    return {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in scores.items()}


# ── Diversity Metrics ────────────────────────────────────────────────────────


def compute_catalog_coverage(
    predictions: dict[str, list[list[str]]],
    catalog_size: int,
) -> float:
    """Fraction of unique catalog items recommended across all sessions & turns.

    Args:
        predictions: {session_id: [turn_recs, ...]}
        catalog_size: Total number of tracks in the catalog

    Returns:
        Coverage ratio in [0, 1]
    """
    all_recommended = set()
    for session_preds in predictions.values():
        for turn_preds in session_preds:
            all_recommended.update(turn_preds)

    return len(all_recommended) / catalog_size if catalog_size > 0 else 0.0


def compute_lexical_diversity(responses: list[str], n: int = 2) -> float:
    """Distinct-n lexical diversity of generated responses.

    Args:
        responses: List of generated response strings
        n: n-gram size (default 2 = distinct-2)

    Returns:
        Ratio of unique n-grams to total n-grams
    """
    all_ngrams: list[tuple] = []
    for resp in responses:
        tokens = resp.lower().split()
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ── Full Evaluation ──────────────────────────────────────────────────────────


def evaluate_all(
    predictions: dict[str, list[list[str]]],
    ground_truths: dict[str, list[list[str]]],
    responses: list[str],
    catalog_size: int,
    k_values: list[int] = [1, 10, 20],
) -> dict[str, float]:
    """Run full evaluation suite and return all metrics."""
    results = {}
    results.update(compute_ndcg(predictions, ground_truths, k_values))
    results["Catalog_Coverage"] = compute_catalog_coverage(predictions, catalog_size)
    results["Lexical_Diversity_D2"] = compute_lexical_diversity(responses, n=2)
    return results
