
import pandas as pd
import pytest

from src.data_loader import load_products


def test_loads_valid_catalogue(tmp_path):
    path = tmp_path / "products.csv"

    pd.DataFrame({
        "product_id": [1, 2],
        "name": [" Mouse ", "Chair"],
        "category": ["Electronics", "Furniture"],
        "price": [25.99, 199.99],
        "description": ["Wireless mouse", "Office chair"],
    }).to_csv(path, index=False)

    result = load_products(path)

    assert len(result) == 2
    assert result.loc[0, "name"] == "Mouse"
    assert result["product_id"].is_unique


def test_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "products.csv"

    pd.DataFrame({
        "product_id": [1, 1],
        "name": ["Mouse", "Chair"],
        "category": ["Electronics", "Furniture"],
        "price": [25.99, 199.99],
        "description": ["Mouse", "Chair"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate product IDs"):
        load_products(path)


def test_missing_description_is_allowed(tmp_path):
    path = tmp_path / "products.csv"

    pd.DataFrame({
        "product_id": [1],
        "name": ["Mouse"],
        "category": ["Electronics"],
        "price": [25.99],
        "description": [None],
    }).to_csv(path, index=False)

    result = load_products(path)

    assert len(result) == 1
    assert result.loc[0, "description"] == ""
