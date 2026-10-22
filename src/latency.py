from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyMetrics:
    initialization: float = 0.0
    query_embedding: float = 0.0
    keyword_retrieval: float = 0.0
    vector_retrieval: float = 0.0
    filtering: float = 0.0
    generation: float = 0.0
    total: float = 0.0
    details: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        payload = {
            "initialization": self.initialization,
            "query_embedding": self.query_embedding,
            "keyword_retrieval": self.keyword_retrieval,
            "vector_retrieval": self.vector_retrieval,
            "filtering": self.filtering,
            "generation": self.generation,
            "total": self.total,
        }
        payload.update(self.details)
        return payload


def benchmark_search_pipeline(
    query: str,
    products,
    index,
    model,
    bm25_index,
    filters: dict[str, Any] | None = None,
    top_k: int = 5,
    generation_model: Any | None = None,
):
    start_total = time.perf_counter()
    metrics = LatencyMetrics()

    init_start = time.perf_counter()
    metrics.initialization = max(time.perf_counter() - init_start, 0.0)

    embedding_start = time.perf_counter()
    from src.embeddings import embed_query
    embedding = embed_query(query, model)
    metrics.query_embedding = time.perf_counter() - embedding_start

    keyword_start = time.perf_counter()
    from src.hybrid_search import search_bm25
    keyword_results = search_bm25(query, bm25_index, top_k=max(top_k * 3, top_k))
    metrics.keyword_retrieval = time.perf_counter() - keyword_start

    vector_start = time.perf_counter()
    from src.vector_store import search_vector_index
    vector_results = search_vector_index(embedding, index, top_k=max(top_k * 3, top_k))
    metrics.vector_retrieval = time.perf_counter() - vector_start

    filter_start = time.perf_counter()
    from src.filters import apply_product_filters
    from src.search_saved_index import _product_row_to_result
    import pandas as pd
    from src.hybrid_search import reciprocal_rank_fusion

    fused_results = reciprocal_rank_fusion([vector_results, keyword_results])
    semantic_lookup = {position: float(score) for position, score in vector_results}
    keyword_lookup = {position: float(score) for position, score in keyword_results}
    candidate_rows = []
    for position, fused_score in fused_results:
        if position >= len(products):
            continue
        product = products.iloc[position]
        candidate_rows.append(
            _product_row_to_result(
                position,
                product,
                fused_score,
                semantic_lookup.get(position, 0.0),
                keyword_lookup.get(position, 0.0),
            )
        )
    filtered = apply_product_filters(pd.DataFrame(candidate_rows), filters or {}) if candidate_rows else pd.DataFrame()
    metrics.filtering = time.perf_counter() - filter_start

    generation_start = time.perf_counter()
    from src.rag_pipeline import generate_grounded_response
    answer = generate_grounded_response(
        query,
        filtered,
        filters=(filters or {}),
        generation_model=generation_model,
    )
    metrics.generation = time.perf_counter() - generation_start

    metrics.total = time.perf_counter() - start_total
    return {"results": filtered.head(top_k).to_dict(orient="records"), "answer": answer, "latency": metrics.as_dict()}


def summarize_latencies(samples: list[dict]) -> dict[str, float]:
    payload = {"p50": {}, "p95": {}}
    for key in ["initialization", "query_embedding", "keyword_retrieval", "vector_retrieval", "filtering", "generation", "total"]:
        values = [float(sample["latency"].get(key, 0.0)) for sample in samples if sample.get("latency")]
        if not values:
            continue
        values_sorted = sorted(values)
        payload["p50"][key] = statistics.median(values_sorted)
        rank_index = max(0, min(len(values_sorted) - 1, int(round(0.95 * (len(values_sorted) - 1)))))
        payload["p95"][key] = values_sorted[rank_index]
    return payload
