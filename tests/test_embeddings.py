
import numpy as np
import pandas as pd
import pytest

from src.embeddings import (
    build_product_documents,
    embed_documents,
    embed_query,
)


def test_build_product_documents():
    products = pd.DataFrame([{
        "name": "Wireless Mouse",
        "category": "Electronics",
        "description": "Ergonomic mouse",
    }])

    documents = build_product_documents(products)

    assert len(documents) == 1
    assert "Wireless Mouse" in documents[0]
    assert "Electronics" in documents[0]
    assert "Ergonomic mouse" in documents[0]


class FakeEmbeddingModel:
    def encode(self, documents, **kwargs):
        return np.array([
            [3.0, 4.0] for _ in documents
        ], dtype=np.float32)


def test_embeddings_are_normalized():
    vectors = embed_documents(
        ["Product one", "Product two"],
        FakeEmbeddingModel(),
    )

    assert vectors.shape == (2, 2)
    assert vectors.dtype == np.float32

    norms = np.linalg.norm(vectors, axis=1)

    assert np.allclose(norms, 1.0)


def test_large_document_lists_are_encoded_in_bounded_chunks():
    class CountingEmbeddingModel(FakeEmbeddingModel):
        def __init__(self):
            self.batch_lengths = []

        def encode(self, documents, **kwargs):
            self.batch_lengths.append(len(documents))
            return super().encode(documents, **kwargs)

    model = CountingEmbeddingModel()
    vectors = embed_documents(["product"] * 300, model)

    assert vectors.shape == (300, 2)
    assert model.batch_lengths == [256, 44]


def test_empty_documents_are_rejected():
    with pytest.raises(ValueError, match="empty document list"):
        embed_documents([], FakeEmbeddingModel())


def test_query_embedding_is_normalized():
    vector = embed_query("  wireless mouse  ", FakeEmbeddingModel())

    assert vector.shape == (2,)
    assert vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "encoded",
    [
        np.array([[1.0, np.nan]], dtype=np.float32),
        np.array([[0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    ],
)
def test_invalid_embedding_outputs_are_rejected(encoded):
    class InvalidEmbeddingModel:
        def encode(self, _documents, **_kwargs):
            return encoded

    with pytest.raises(ValueError):
        embed_documents(["one document"], InvalidEmbeddingModel())


def test_query_embedding_rejects_empty_query():
    with pytest.raises(ValueError, match="cannot be empty"):
        embed_query("  ", FakeEmbeddingModel())
