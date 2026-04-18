"""Unit tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    ndcg_at_k,
    compute_ndcg,
    compute_catalog_coverage,
    compute_lexical_diversity,
)


class TestNDCG:
    def test_perfect_ranking(self):
        recommended = ["t1", "t2", "t3"]
        ground_truth = ["t1", "t2", "t3"]
        assert ndcg_at_k(recommended, ground_truth, k=3) == pytest.approx(1.0)

    def test_no_relevant(self):
        recommended = ["t4", "t5"]
        ground_truth = ["t1", "t2"]
        assert ndcg_at_k(recommended, ground_truth, k=5) == 0.0

    def test_partial_relevant(self):
        recommended = ["t1", "t9", "t2"]
        ground_truth = ["t1", "t2"]
        score = ndcg_at_k(recommended, ground_truth, k=3)
        assert 0.0 < score < 1.0

    def test_empty_ground_truth(self):
        assert ndcg_at_k(["t1"], [], k=5) == 0.0


class TestCatalogCoverage:
    def test_full_coverage(self):
        predictions = {"s1": [["t1", "t2"], ["t3", "t4"]]}
        assert compute_catalog_coverage(predictions, catalog_size=4) == pytest.approx(1.0)

    def test_partial_coverage(self):
        predictions = {"s1": [["t1", "t2"]]}
        assert compute_catalog_coverage(predictions, catalog_size=10) == pytest.approx(0.2)

    def test_empty_predictions(self):
        assert compute_catalog_coverage({}, catalog_size=100) == 0.0


class TestLexicalDiversity:
    def test_identical_responses(self):
        responses = ["i love rock music", "i love rock music"]
        assert compute_lexical_diversity(responses, n=2) == pytest.approx(1.0)

    def test_diverse_responses(self):
        responses = ["i love jazz", "rock music is great", "classical piano is beautiful"]
        score = compute_lexical_diversity(responses, n=2)
        assert score > 0.0

    def test_empty_responses(self):
        assert compute_lexical_diversity([], n=2) == 0.0
