from src.hybrid_search import build_bm25_index, hybrid_search


from time import perf_counter

from src.data_loader import load_products
from src.embeddings import (
    build_product_documents,
    load_embedding_model,
    embed_documents,
    embed_query,
)
from src.vector_store import build_vector_index, search_vector_index


def main():
    products = load_products("data/products.csv")

    documents = build_product_documents(products)

    model = load_embedding_model()

    product_vectors = embed_documents(documents, model)
    index = build_vector_index(product_vectors)
    bm25_index = build_bm25_index(documents)

    queries = [
        "wireless mouse",
        "comfortable chair",
        "lined notebook",
    ]

    for query in queries:
        start = perf_counter()

        results = hybrid_search(
                    query=query,
                    model=model,
                    bm25_index=bm25_index,
                    vector_index=index,
                    top_k=3,
                )

        elapsed_ms = (perf_counter() - start) * 1000

        print(f"\nQuery: {query}")
        print(f"Search latency: {elapsed_ms:.2f} ms")

        for position, score in results:
            product = products.iloc[position]

            print(
    f"- {product['name']} | "
    f"Category: {product['category']} | "
    f"Price: {product['price']} | "
    f"RRF score: {score:.4f}"
)

if __name__ == "__main__":
    main()
