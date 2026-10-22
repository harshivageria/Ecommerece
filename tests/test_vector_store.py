
import numpy as np
import pytest

from src.vector_store import build_vector_index, search_vector_index


def test_index_stores_all_vectors():
    vectors = np.array(
        [[1, 0], [0, 1], [-1, 0]],
        dtype=np.float32,
    )

    index = build_vector_index(vectors)

    assert index.ntotal == 3
    assert index.d == 2


def test_search_returns_best_match_first():
    vectors = np.array(
        [[1, 0], [0, 1], [-1, 0]],
        dtype=np.float32,
    )

    index = build_vector_index(vectors)
    query = np.array([0.9, 0.1], dtype=np.float32)

    results = search_vector_index(query, index, top_k=2)

    assert results[0][0] == 0
    assert results[0][1] > results[1][1]


def test_index_and_query_use_cosine_similarity():
    vectors = np.array([[100, 0], [1, 1]], dtype=np.float32)
    index = build_vector_index(vectors)

    results = search_vector_index(
        np.array([10, 10], dtype=np.float32),
        index,
        top_k=2,
    )

    assert results[0][0] == 1
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_top_k_does_not_exceed_catalogue_size():
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)
    index = build_vector_index(vectors)

    results = search_vector_index(
        np.array([1, 0], dtype=np.float32),
        index,
        top_k=10,
    )

    assert len(results) == 2


def test_empty_vectors_are_rejected():
    with pytest.raises(ValueError):
        build_vector_index(np.empty((0, 2), dtype=np.float32))


def test_non_finite_vectors_are_rejected():
    with pytest.raises(ValueError, match="invalid values"):
        build_vector_index(np.array([[np.nan, 0.0]], dtype=np.float32))


def test_zero_length_vectors_and_queries_are_rejected():
    with pytest.raises(ValueError, match="non-zero lengths"):
        build_vector_index(np.array([[0.0, 0.0]], dtype=np.float32))

    index = build_vector_index(np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-zero length"):
        search_vector_index(np.array([0.0, 0.0], dtype=np.float32), index)


def test_wrong_query_dimension_is_rejected():
    index = build_vector_index(
        np.array([[1, 0], [0, 1]], dtype=np.float32)
    )

    with pytest.raises(ValueError):
        search_vector_index(
            np.array([1, 0, 0], dtype=np.float32),
            index,
        )
