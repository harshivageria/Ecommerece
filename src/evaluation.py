from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd


DEFAULT_EVALUATION_PATH = Path("data/evaluation_queries.json")


def load_evaluation_queries(path: str | Path = DEFAULT_EVALUATION_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Evaluation data must be a list of query objects.")

    return payload


def _normalize_ids(ids) -> list[str]:
    if ids is None:
        return []
    return [str(item) for item in ids]


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int = 5) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(_normalize_ids(relevant_ids))
    retrieved = _normalize_ids(retrieved_ids)[:k]
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved)) / len(relevant)


def reciprocal_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    relevant = set(_normalize_ids(relevant_ids))
    for index, product_id in enumerate(_normalize_ids(retrieved_ids), start=1):
        if product_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int = 5) -> float:
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(_normalize_ids(relevant_ids))
    retrieved = _normalize_ids(retrieved_ids)[:k]
    if not relevant:
        return 0.0

    gains = []
    for product_id in retrieved:
        gains.append(1.0 if product_id in relevant else 0.0)

    dcg = sum((2**score - 1) / (idx + 1) for idx, score in enumerate(gains))
    ideal = [1.0] * min(len(relevant), k)
    ideal_dcg = sum((2**score - 1) / (idx + 1) for idx, score in enumerate(ideal))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_retrieval(
    search_fn: Callable[[str], list[str]],
    path: str | Path = DEFAULT_EVALUATION_PATH,
    k: int = 5,
) -> dict:
    queries = load_evaluation_queries(path)
    results = []
    summary = {"recall@k": [], "mrr": [], "ndcg@k": []}

    for item in queries:
        query = str(item["query"]).strip()
        relevant_ids = _normalize_ids(item.get("relevant_product_ids", []))
        retrieved_ids = _normalize_ids(search_fn(query))
        score_recall = recall_at_k(relevant_ids, retrieved_ids, k)
        score_mrr = reciprocal_rank(relevant_ids, retrieved_ids)
        score_ndcg = ndcg_at_k(relevant_ids, retrieved_ids, k)

        summary["recall@k"].append(score_recall)
        summary["mrr"].append(score_mrr)
        summary["ndcg@k"].append(score_ndcg)

        results.append(
            {
                "query": query,
                "relevant_product_ids": relevant_ids,
                "retrieved_product_ids": retrieved_ids[:k],
                "recall@k": score_recall,
                "mrr": score_mrr,
                "ndcg@k": score_ndcg,
            }
        )

    summary_values = {
        "recall@k": sum(summary["recall@k"]) / len(summary["recall@k"]) if summary["recall@k"] else 0.0,
        "mrr": sum(summary["mrr"]) / len(summary["mrr"]) if summary["mrr"] else 0.0,
        "ndcg@k": sum(summary["ndcg@k"]) / len(summary["ndcg@k"]) if summary["ndcg@k"] else 0.0,
    }

    return {"results": results, "summary": summary_values}
