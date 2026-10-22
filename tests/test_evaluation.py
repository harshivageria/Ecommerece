import json

import pytest

from src.evaluation import (
    evaluate_retrieval,
    load_evaluation_queries,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics():
    relevant = ["a", "b"]
    retrieved = ["b", "x", "a"]

    assert recall_at_k(relevant, retrieved, k=2) == 0.5
    assert reciprocal_rank(relevant, retrieved) == 1.0
    assert ndcg_at_k(relevant, retrieved, k=2) == pytest.approx(2 / 3)


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_rank_metrics_reject_non_positive_k(metric):
    with pytest.raises(ValueError, match="at least 1"):
        metric(["a"], ["a"], k=0)


def test_evaluate_retrieval_loads_queries_and_averages_scores(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(
        json.dumps([
            {"query": "first", "relevant_product_ids": ["a"]},
            {"query": "second", "relevant_product_ids": ["b"]},
        ]),
        encoding="utf-8",
    )

    assert load_evaluation_queries(path)[0]["query"] == "first"
    result = evaluate_retrieval(
        lambda query: ["a"] if query == "first" else ["x"],
        path,
    )

    assert result["summary"]["recall@k"] == 0.5
    assert result["summary"]["mrr"] == 0.5
    assert result["summary"]["ndcg@k"] == 0.5
