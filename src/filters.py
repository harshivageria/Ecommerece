from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _coerce_float(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} cannot be null.")

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty.")
        value = text.replace(",", "")

    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc

    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be a finite number.")

    return numeric


def normalize_filter_constraints(raw_filters: dict[str, Any] | None) -> dict[str, Any]:
    if raw_filters is None:
        return {}

    cleaned: dict[str, Any] = {}

    for key, value in raw_filters.items():
        if value is None:
            continue

        normalized_key = str(key).strip()
        if not normalized_key:
            continue

        if normalized_key in {"min_price", "max_price"}:
            cleaned[normalized_key] = _coerce_float(value, normalized_key)
            if cleaned[normalized_key] < 0:
                raise ValueError(f"{normalized_key} cannot be negative.")
        elif normalized_key in {"category", "subcategory", "brand"}:
            text = str(value).strip()
            if text:
                cleaned[normalized_key] = text
        elif normalized_key in {"min_rating", "max_rating"}:
            cleaned[normalized_key] = _coerce_float(value, normalized_key)
        else:
            continue

    if "min_price" in cleaned and "max_price" in cleaned:
        if cleaned["min_price"] > cleaned["max_price"]:
            raise ValueError("min_price cannot be greater than max_price.")

    if "min_rating" in cleaned and "max_rating" in cleaned:
        if cleaned["min_rating"] > cleaned["max_rating"]:
            raise ValueError("min_rating cannot be greater than max_rating.")

    return cleaned


def _has_verified_brand_data(products: pd.DataFrame) -> bool:
    if "brand" not in products.columns:
        return False
    return bool(products["brand"].fillna("").astype(str).str.strip().ne("").any())


def apply_product_filters(
    products: pd.DataFrame,
    filters: dict[str, Any] | None,
) -> pd.DataFrame:
    if products is None or products.empty:
        return products.copy()

    filtered = products.copy()
    normalized = normalize_filter_constraints(filters)

    if not normalized:
        return filtered

    if "min_price" in normalized:
        filtered = filtered.loc[
            filtered["price"].notna() & (filtered["price"] >= normalized["min_price"])
        ].copy()

    if "max_price" in normalized:
        filtered = filtered.loc[
            filtered["price"].notna() & (filtered["price"] <= normalized["max_price"])
        ].copy()

    if "category" in normalized:
        category_value = str(normalized["category"]).strip()
        if category_value:
            filtered = filtered.loc[
                filtered["category"].fillna("").astype(str).str.strip().str.lower()
                == category_value.lower()
            ].copy()

    if "subcategory" in normalized:
        subcategory_value = str(normalized["subcategory"]).strip()
        if subcategory_value:
            filtered = filtered.loc[
                filtered["subcategory"].fillna("").astype(str).str.strip().str.lower()
                == subcategory_value.lower()
            ].copy()

    if "brand" in normalized:
        if not _has_verified_brand_data(filtered):
            raise ValueError(
                "Brand filtering is unavailable because the catalogue has no verified brand metadata."
            )

        brand_value = str(normalized["brand"]).strip()
        filtered = filtered.loc[
            filtered["brand"].fillna("").astype(str).str.strip().str.lower()
            == brand_value.lower()
        ].copy()

    return filtered
