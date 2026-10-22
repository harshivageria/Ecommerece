
import pytest

from src.hybrid_search import (
    tokenize,
    build_bm25_index,
    search_bm25,
    reciprocal_rank_fusion,
)


def test_tokenize_normalizes_text():
    assert tokenize("Nike AIR-MAX 270!") == [
        "nike", "air", "max", "270"
    ]


def test_tokenize_supports_unicode_scripts():
    assert tokenize("cámara inalámbrica") == ["cámara", "inalámbrica"]
    assert tokenize("संगीत खिलाड़ी") == ["संगीत", "खिलाड़ी"]


def test_bm25_retrieves_matching_document():
    documents = [
        "Product name: Wireless Mouse. Category: Electronics.",
        "Product name: Office Chair. Category: Furniture.",
        "Product name: Notebook. Category: Stationery.",
    ]

    index = build_bm25_index(documents)
    results = search_bm25("wireless mouse", index, top_k=3)

    assert results
    assert results[0][0] == 0
    assert results[0][1] > 0


def test_bm25_returns_no_results_for_unknown_terms():
    documents = [
        "wireless mouse electronics",
        "office chair furniture",
    ]

    index = build_bm25_index(documents)
    results = search_bm25("xyzqwerty", index, top_k=3)

    assert results == []


def test_rrf_rewards_consistent_high_rankings():
    semantic = [(0, 0.9), (1, 0.5)]
    keyword = [(0, 4.2), (2, 2.0)]

    results = reciprocal_rank_fusion([semantic, keyword])

    assert results[0][0] == 0


def test_rrf_does_not_count_duplicate_product_twice_per_list():
    results = reciprocal_rank_fusion([
        [(0, 0.9), (0, 0.8), (1, 0.5)]
    ])

    product_zero_score = dict(results)[0]
    expected_score = 1 / 61

    assert product_zero_score == pytest.approx(expected_score)


def test_empty_bm25_documents_are_rejected():
    with pytest.raises(ValueError):
        build_bm25_index([])
