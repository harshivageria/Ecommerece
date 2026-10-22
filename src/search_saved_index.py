import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.data_loader import load_products
from src.embeddings import MODEL_NAME, build_product_documents, embed_query, load_embedding_model
from src.filters import apply_product_filters
from src.hybrid_search import build_bm25_index, hybrid_search, reciprocal_rank_fusion, search_bm25, search_vector_index
from src.rag_pipeline import validate_user_query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/processed/products.csv"
INDEX_DIR = PROJECT_ROOT / "data/index"
TOP_K = 5


def load_saved_index():
    products = load_products(DATA_PATH)

    metadata_file = INDEX_DIR / "index_metadata.json"
    records_file = INDEX_DIR / "products.json"
    index_file = INDEX_DIR / "products.faiss"

    if not metadata_file.exists() or not records_file.exists() or not index_file.exists():
        raise FileNotFoundError(
            "Saved FAISS index not found. Rebuild it with python -m src.build_index or python src/build_index.py."
        )

    with metadata_file.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    with records_file.open("r", encoding="utf-8") as file:
        records = json.load(file)

    index = faiss.read_index(str(index_file))

    if metadata.get("model_name") != MODEL_NAME:
        raise ValueError("Embedding model does not match the saved index. Rebuild the index for the current embedding model.")

    if metadata.get("index_type") != "IndexFlatIP" or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("Saved index is not configured for inner-product cosine search. Rebuild the FAISS index.")

    if metadata.get("normalized_embeddings") is not True:
        raise ValueError("Saved index does not declare normalized embeddings. Rebuild the FAISS index.")

    if metadata.get("embedding_dimension") != index.d:
        raise ValueError(
            f"Saved index dimensions do not match metadata ({index.d} vs {metadata.get('embedding_dimension')}). Rebuild the FAISS index."
        )

    stored_vectors = index.reconstruct_n(0, index.ntotal)
    vector_norms = (stored_vectors**2).sum(axis=1) ** 0.5
    if not np.isfinite(vector_norms).all() or not np.allclose(vector_norms, 1.0, atol=1e-4):
        raise ValueError("Saved index contains non-normalized vectors. Rebuild the FAISS index.")

    if metadata.get("product_count") != len(products):
        raise ValueError(
            f"Saved index metadata product count ({metadata.get('product_count')}) does not match the catalogue length ({len(products)}). Rebuild the index."
        )

    if len(records) != len(products):
        raise ValueError("Product metadata and catalogue have different row counts. Rebuild the FAISS index.")

    if index.ntotal != len(products):
        raise ValueError("Product catalogue and FAISS index have different row counts. Rebuild the FAISS index.")

    documents = build_product_documents(products)
    for position, record in enumerate(records):
        if record.get("position") != position:
            raise ValueError("Invalid product positions in saved metadata. Rebuild the FAISS index.")
        if str(products.iloc[position]["product_id"]) != str(record.get("product_id")):
            raise ValueError("Product order changed. Rebuild the FAISS index.")
        if record.get("document") != documents[position]:
            raise ValueError("Product content changed. Rebuild the FAISS index.")

    return products, index, records


def _product_row_to_result(position: int, product: pd.Series, fused_score: float, semantic_score: float, keyword_score: float):
    return {
        "position": int(position),
        "product_id": str(product.get("product_id", "")),
        "name": str(product.get("name", "")).strip(),
        "category": str(product.get("category", "")).strip() or "Uncategorized",
        "subcategory": str(product.get("subcategory", "")).strip(),
        "brand": str(product.get("brand", "")).strip(),
        "price": product.get("price"),
        "description": str(product.get("description", "")).strip(),
        "product_url": str(product.get("product_url", "")).strip(),
        "image_url": str(product.get("image_url", "")).strip(),
        "rrf_score": float(fused_score),
        "semantic_score": float(semantic_score),
        "keyword_score": float(keyword_score),
    }


def search_products(query, products, index, model, bm25_index=None, top_k=TOP_K, filters=None):
    validated = validate_user_query(query)
    filtered_constraints = filters or {}
    if bm25_index is None:
        documents = [str(row.get("name", "")) + " " + str(row.get("description", "")) for _, row in products.iterrows()]
        bm25_index = build_bm25_index(documents)

    query_vector = embed_query(validated, model)
    semantic_results = search_vector_index(query_vector, index, top_k=max(top_k * 3, top_k))
    keyword_results = search_bm25(validated, bm25_index, top_k=max(top_k * 3, top_k))

    if not semantic_results and not keyword_results:
        return []

    fused_results = reciprocal_rank_fusion([semantic_results, keyword_results])
    semantic_lookup = {position: float(score) for position, score in semantic_results}
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

    if candidate_rows:
        candidate_df = pd.DataFrame(candidate_rows)
        filtered = apply_product_filters(candidate_df, filtered_constraints)
        final_results = filtered.to_dict(orient="records")
    else:
        final_results = []

    for rank, result in enumerate(final_results[:top_k], start=1):
        result["rank"] = rank
        result["matching_reason"] = "Retrieved by hybrid product search"

    return final_results[:top_k]


def main():
    products, index, _records = load_saved_index()
    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = load_embedding_model()
    documents = [str(row.get("name", "")) + " " + str(row.get("description", "")) for _, row in products.iterrows()]
    bm25_index = build_bm25_index(documents)

    while True:
        query = input("\nEnter a product search (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Search ended.")
            break

        if not query:
            print("Please enter a search query.")
            continue

        results = search_products(query, products, index, model, bm25_index=bm25_index, top_k=TOP_K)

        if not results:
            print("No matching products found.")
            continue

        for rank, result in enumerate(results, start=1):
            price = result.get("price")
            price_value = f"${price:.2f}" if pd.notna(price) else "Not available"
            print(f"{rank}. {result['name']}")
            print(f"   Category: {result['category']}")
            print(f"   Price: {price_value}")
            print(f"   RRF score: {result['rrf_score']:.4f}")
            if result.get("product_url"):
                print(f"   Product URL: {result['product_url']}")
            print()


if __name__ == "__main__":
    main()
