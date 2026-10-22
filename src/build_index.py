
import json
from pathlib import Path

import faiss

from src.data_loader import load_products
from src.embeddings import (
    MODEL_NAME,
    build_product_documents,
    load_embedding_model,
    embed_documents,
)
from src.vector_store import build_vector_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/processed/products.csv"
INDEX_DIR = PROJECT_ROOT / "data/index"


def main():
    products = load_products(DATA_PATH)
    documents = build_product_documents(products)

    print(f"\nLoading embedding model: {MODEL_NAME}")
    model = load_embedding_model()

    print("\nEmbedding products...")
    vectors = embed_documents(documents, model)

    print("Embedding matrix shape:", vectors.shape)

    index = build_vector_index(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(
        index,
        str(INDEX_DIR / "products.faiss"),
    )

    records = []

    for position, (_, row) in enumerate(products.iterrows()):
        records.append({
            "position": position,
            "product_id": str(row["product_id"]),
            "document": documents[position],
        })

    with (INDEX_DIR / "products.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(records, file, ensure_ascii=False)

    metadata = {
        "model_name": MODEL_NAME,
        "product_count": len(products),
        "embedding_dimension": int(vectors.shape[1]),
        "index_type": "IndexFlatIP",
        "normalized_embeddings": True,
    }

    with (INDEX_DIR / "index_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nIndex saved successfully.")
    print("Indexed products:", index.ntotal)
    print("Embedding dimension:", index.d)
    print("Index directory:", INDEX_DIR)


if __name__ == "__main__":
    main()
