from __future__ import annotations

import html
import logging
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import streamlit as st

from src.data_loader import load_products
from src.hybrid_search import build_bm25_index
from src.rag_pipeline import generate_grounded_response, load_generation_model, validate_user_query

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data/processed/products.csv"
logger = logging.getLogger(__name__)


@st.cache_data
def get_filter_options() -> tuple[list[str], list[str]]:
    products = load_products(DATA_PATH)
    categories = sorted(
        {value for value in products["category"].astype(str).str.strip() if value}
    )
    subcategories = sorted(
        {value for value in products["subcategory"].astype(str).str.strip() if value}
    )
    return categories, subcategories


def _reset_filters() -> None:
    st.session_state["min_price"] = 0.0
    st.session_state["max_price"] = 0.0
    st.session_state["category"] = "All"
    st.session_state["subcategory"] = "All"


def _is_safe_external_url(url: str) -> bool:
    try:
        return urlsplit(url).scheme.lower() in {"http", "https"}
    except ValueError:
        return False


@st.cache_resource
def get_catalogue_and_index():
    from src.embeddings import load_embedding_model
    from src.search_saved_index import load_saved_index

    products, index, _records = load_saved_index()
    documents = [
        f"{row.get('name', '')} {row.get('description', '')} {row.get('category', '')} {row.get('subcategory', '')}"
        for _, row in products.iterrows()
    ]
    bm25_index = build_bm25_index(documents)
    model = load_embedding_model()
    return products, index, bm25_index, model


@st.cache_resource
def get_generation_model():
    return load_generation_model()


def _render_product_card(result: dict):
    price_value = result.get("price")
    price_text = f"${float(price_value):.2f}" if pd.notna(price_value) else "Price unavailable"
    product_name = str(result.get("name", "")).strip() or "Unnamed product"
    category = str(result.get("category", "")).strip() or "Uncategorized"
    description = str(result.get("description", "")).strip()
    product_url = str(result.get("product_url", "")).strip()
    image_url = str(result.get("image_url", "")).strip()
    safe_product_url = _is_safe_external_url(product_url)
    safe_image_url = _is_safe_external_url(image_url)

    with st.container():
        st.markdown(
            "<div style='border:1px solid #ddd; border-radius:10px; padding:12px; margin-bottom:10px;'>"
            + (
                f"<h4>{html.escape(product_name)}</h4>"
                f"<p><strong>Category:</strong> {html.escape(category)}</p>"
                f"<p><strong>Price:</strong> {price_text}</p>"
                + (f"<p>{html.escape(description[:220])}...</p>" if description else "")
                + (
                    f"<p><a href='{html.escape(product_url, quote=True)}' target='_blank' rel='noopener noreferrer'>View product</a></p>"
                    if safe_product_url
                    else ""
                )
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        if safe_image_url:
            st.image(image_url, width="stretch")


st.set_page_config(page_title="Ecommerce Product Discovery", page_icon="shopping_bag", layout="wide")

st.title("Ecommerce Product Discovery")

with st.sidebar:
    st.header("Search filters")
    min_price = st.number_input("Minimum price", min_value=0.0, value=0.0, step=1.0, key="min_price")
    max_price = st.number_input("Maximum price", min_value=0.0, value=0.0, step=1.0, key="max_price")
    if max_price and min_price > max_price:
        st.warning("Minimum price is higher than maximum price. Adjust the range before searching.")
        st.stop()

    categories, subcategories = get_filter_options()
    selected_category = st.selectbox(
        "Category",
        ["All", *categories],
        key="category",
    )
    selected_subcategory = st.selectbox(
        "Subcategory",
        ["All", *subcategories],
        key="subcategory",
    )

    st.button("Reset filters", on_click=_reset_filters)

query = st.text_input("Search products", placeholder="Find a product by name, use case, or feature")

if not query.strip():
    st.info("Enter a product name or a natural-language product description to search the catalogue.")
    st.stop()

try:
    validate_user_query(query)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

try:
    products, index, bm25_index, model = get_catalogue_and_index()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error(f"Product search is not ready: {exc}")
    st.info("Build the matching multilingual search index with: python -m src.build_index")
    st.stop()

filters = {}
if min_price > 0:
    filters["min_price"] = min_price
if max_price > 0:
    filters["max_price"] = max_price
if selected_category != "All":
    filters["category"] = selected_category
if selected_subcategory != "All":
    filters["subcategory"] = selected_subcategory

from src.search_saved_index import search_products

try:
    results = search_products(query, products, index, model, bm25_index=bm25_index, top_k=8, filters=filters)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if not results:
    st.warning("No relevant catalogue matches remain after validation and filtering. Try a broader query or remove a price constraint.")
    st.stop()

st.subheader("Results")
for result in results:
    _render_product_card(result)

st.subheader("Grounded answer")
try:
    generation_model = get_generation_model()
    answer_payload = generate_grounded_response(query, pd.DataFrame(results), filters=filters, generation_model=generation_model)
    st.write(answer_payload["answer"])
except Exception as exc:
    logger.exception("Grounded response generation failed.")
    st.write("I could not generate a grounded response from the local model, but the retrieved product records above are the verified catalogue results.")
