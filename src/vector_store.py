
import numpy as np
import faiss


def build_vector_index(vectors: np.ndarray):
    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Vectors must be a 2D matrix.")

    if vectors.shape[0] == 0:
        raise ValueError("Cannot index an empty vector matrix.")

    if vectors.shape[1] == 0:
        raise ValueError("Vectors must have at least one dimension.")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain invalid values.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Vectors must have finite, non-zero lengths.")

    vectors = vectors / norms
    vectors = np.ascontiguousarray(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


def search_vector_index(
    query_vector: np.ndarray,
    index,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    if index.ntotal == 0:
        return []

    query_vector = np.asarray(query_vector, dtype=np.float32)

    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError("Provide exactly one query vector.")

    if query_vector.shape[1] != index.d:
        raise ValueError("Query and index dimensions do not match.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query vector contains invalid values.")

    query_norm = np.linalg.norm(query_vector)
    if not np.isfinite(query_norm) or query_norm == 0:
        raise ValueError("Query vector must have a finite, non-zero length.")

    query_vector = query_vector / query_norm
    query_vector = np.ascontiguousarray(query_vector)

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []

    for position, score in zip(indices[0], scores[0]):
        if position >= 0:
            results.append((int(position), float(score)))

    return results
