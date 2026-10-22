
from pathlib import Path

from src.data_loader import load_products
from src.embeddings import (
    build_product_documents,
    load_embedding_model,
    embed_documents,
)


def main():
    project_root = Path(__file__).resolve().parents[1]
    products = load_products(project_root / "data/processed/products.csv")
    documents = build_product_documents(products)
    model = load_embedding_model()

    vectors = embed_documents(documents, model)

    print("Number of products:", len(products))
    print("Embedding matrix shape:", vectors.shape)
    print("Vector data type:", vectors.dtype)

    print("\nFirst product:")
    print(documents[0])

    print("\nFirst embedding:")
    print(vectors[0][:10])

    print("\nVector norm:")
    print((vectors[0] ** 2).sum() ** 0.5)


if __name__ == "__main__":
    main()
