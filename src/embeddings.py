
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENCODE_BATCH_SIZE = 32
DOCUMENT_CHUNK_SIZE = 256



def build_product_documents(products: pd.DataFrame) -> list[str]:
    documents = []

    for _, row in products.iterrows():
        fields = [
            f"Product name: {row['name']}",
        ]

        brand = str(row.get("brand", "")).strip()
        category = str(row.get("category", "")).strip()
        subcategory = str(row.get("subcategory", "")).strip()
        description = str(row.get("description", "")).strip()

        if brand:
            fields.append(f"Brand: {brand}")

        if category:
            fields.append(f"Category: {category}")

        if subcategory:
            fields.append(f"Subcategory: {subcategory}")

        if description:
            fields.append(f"Description: {description}")

        documents.append(". ".join(fields) + ".")

    return documents


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model."""

    return SentenceTransformer(MODEL_NAME)


def embed_documents(
    documents: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Encode and normalize product documents."""

    if not documents:
        raise ValueError("Cannot embed an empty document list.")

    encoded_chunks = []
    for start in range(0, len(documents), DOCUMENT_CHUNK_SIZE):
        encoded_chunks.append(
            model.encode(
                documents[start : start + DOCUMENT_CHUNK_SIZE],
                batch_size=ENCODE_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        )
        if len(documents) > DOCUMENT_CHUNK_SIZE:
            print(
                f"Embedded {min(start + DOCUMENT_CHUNK_SIZE, len(documents))} "
                f"of {len(documents)} product documents.",
                flush=True,
            )

    vectors = np.asarray(np.concatenate(encoded_chunks, axis=0), dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if vectors.shape[0] != len(documents) or vectors.shape[1] == 0:
        raise ValueError("Embedding dimensions do not match the document list.")

    if not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors contain invalid values.")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Embedding vectors must have finite, non-zero lengths.")

    vectors = vectors / norms

    return vectors.astype(np.float32)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    return embed_documents([query.strip()], model)[0]