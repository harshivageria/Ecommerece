import numpy as np
import pandas as pd

from src.hybrid_search import build_bm25_index
from src.latency import benchmark_search_pipeline, summarize_latencies
from src.vector_store import build_vector_index


class FixedEmbeddingModel:
    def encode(self, documents, **_kwargs):
        return np.array([[1.0, 0.0] for _ in documents], dtype=np.float32)


def test_benchmark_returns_grounded_answer_without_generation_model():
    products = pd.DataFrame([{
        "product_id": "a",
        "name": "Alpha mouse",
        "category": "Electronics",
        "subcategory": "Accessories",
        "brand": "",
        "price": 25.0,
        "description": "Wireless mouse",
        "product_url": "",
        "image_url": "",
    }])
    index = build_vector_index(np.array([[1.0, 0.0]], dtype=np.float32))
    bm25 = build_bm25_index(["Alpha mouse Wireless mouse Electronics Accessories"])

    result = benchmark_search_pipeline(
        "Alpha mouse",
        products,
        index,
        FixedEmbeddingModel(),
        bm25,
    )

    assert result["results"][0]["product_id"] == "a"
    assert result["answer"]["status"] == "grounded_fallback"
    assert "Alpha mouse" in result["answer"]["answer"]
    assert result["latency"]["total"] >= 0


def test_summarize_latencies_calculates_percentiles():
    summary = summarize_latencies([
        {"latency": {"total": 1.0}},
        {"latency": {"total": 3.0}},
    ])

    assert summary["p50"]["total"] == 2.0
    assert summary["p95"]["total"] == 3.0

