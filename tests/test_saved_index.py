import json

import faiss
import numpy as np
import pandas as pd
import pytest

from src.embeddings import MODEL_NAME, build_product_documents
from src.search_saved_index import load_saved_index


def _write_test_index(tmp_path, monkeypatch, *, saved_document=None, vector=None):
    products_path = tmp_path / "products.csv"
    pd.DataFrame([{
        "product_id": "product-1",
        "name": "Alpha mouse",
        "category": "Electronics",
        "price": 25.0,
        "description": "Wireless mouse",
    }]).to_csv(products_path, index=False)

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    products = pd.read_csv(products_path)
    products["brand"] = ""
    products["subcategory"] = ""
    products["product_url"] = ""
    products["image_url"] = ""
    document = build_product_documents(products)[0]

    index = faiss.IndexFlatIP(2)
    index.add(np.array([vector or [1.0, 0.0]], dtype=np.float32))
    faiss.write_index(index, str(index_dir / "products.faiss"))
    (index_dir / "products.json").write_text(
        json.dumps([{
            "position": 0,
            "product_id": "product-1",
            "document": saved_document if saved_document is not None else document,
        }]),
        encoding="utf-8",
    )
    (index_dir / "index_metadata.json").write_text(
        json.dumps({
            "model_name": MODEL_NAME,
            "product_count": 1,
            "embedding_dimension": 2,
            "index_type": "IndexFlatIP",
            "normalized_embeddings": True,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.search_saved_index.DATA_PATH", products_path)
    monkeypatch.setattr("src.search_saved_index.INDEX_DIR", index_dir)
    return document


def test_load_saved_index_checks_catalogue_and_index_alignment(tmp_path, monkeypatch):
    _write_test_index(tmp_path, monkeypatch)

    products, index, records = load_saved_index()

    assert products.loc[0, "name"] == "Alpha mouse"
    assert index.ntotal == 1
    assert records[0]["position"] == 0


def test_load_saved_index_rejects_stale_product_documents(tmp_path, monkeypatch):
    _write_test_index(tmp_path, monkeypatch, saved_document="old product text")

    with pytest.raises(ValueError, match="Product content changed"):
        load_saved_index()


def test_load_saved_index_rejects_non_normalized_dense_vectors(tmp_path, monkeypatch):
    _write_test_index(tmp_path, monkeypatch, vector=[5.0, 0.0])

    with pytest.raises(ValueError, match="non-normalized vectors"):
        load_saved_index()
