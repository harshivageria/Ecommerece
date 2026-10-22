
import unicodedata

from rank_bm25 import BM25Okapi

from src.embeddings import embed_query
from src.vector_store import search_vector_index


def tokenize(text: str) -> list[str]:
    """Normalize text into searchable word tokens."""
    tokens = []
    current = []

    for character in str(text).casefold():
        if character.isalnum() or (current and unicodedata.category(character).startswith("M")):
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return tokens


def build_bm25_index(documents: list[str]):
    if not documents:
        raise ValueError("Cannot index an empty document list.")

    tokenized_documents = [tokenize(doc) for doc in documents]

    if any(not tokens for tokens in tokenized_documents):
        raise ValueError("A document has no searchable tokens.")

    return BM25Okapi(tokenized_documents)


def search_bm25(query: str, bm25_index, top_k: int = 3):
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    tokens = tokenize(query)

    if not tokens:
        return []

    scores = bm25_index.get_scores(tokens)
    ranked_positions = sorted(
        range(len(scores)),
        key=lambda i: float(scores[i]),
        reverse=True,
    )

    return [
        (int(position), float(scores[position]))
        for position in ranked_positions[:top_k]
        if scores[position] > 0
    ]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    if k < 1:
        raise ValueError("RRF k must be at least 1.")

    fused_scores = {}

    for ranked_list in ranked_lists:
        seen_in_list = set()

        for rank, (product_position, _score) in enumerate(
            ranked_list,
            start=1,
        ):
            if product_position in seen_in_list:
                continue

            seen_in_list.add(product_position)

            fused_scores[product_position] = (
                fused_scores.get(product_position, 0.0)
                + 1.0 / (k + rank)
            )

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


def _deduplicate_ranked_results(results: list[tuple[int, float]]) -> list[tuple[int, float]]:
    unique: dict[int, float] = {}
    for position, score in results:
        if position not in unique or score > unique[position]:
            unique[position] = float(score)
    return sorted(unique.items(), key=lambda item: item[1], reverse=True)


def hybrid_search(
    query: str,
    model,
    bm25_index,
    vector_index,
    top_k: int = 3,
):
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = embed_query(query, model)

    semantic_results = search_vector_index(
        query_vector,
        vector_index,
        top_k=top_k,
    )

    keyword_results = search_bm25(
        query,
        bm25_index,
        top_k=top_k,
    )

    fused_results = reciprocal_rank_fusion(
        [semantic_results, keyword_results]
    )

    return _deduplicate_ranked_results(fused_results)[:top_k]
