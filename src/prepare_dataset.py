
from pathlib import Path
import re

import pandas as pd


INPUT_PATH = Path("data/raw/amazon_products.csv")
OUTPUT_PATH = Path("data/processed/products.csv")
TARGET_ROWS = 10_000


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def clean_price(value):
    if pd.isna(value):
        return None

    cleaned = re.sub(r"[$,\s]", "", str(value))

    try:
        price = float(cleaned)
    except ValueError:
        return None

    if price < 0:
        return None

    return price


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)
    df.columns = df.columns.astype(str).str.strip()

    required = {
        "Uniq Id",
        "Product Name",
        "Category",
        "Selling Price",
        "About Product",
        "Product Specification",
        "Technical Details",
        "Product Url",
        "Image",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Required source columns missing: {sorted(missing)}"
        )

    # Standardize identifiers and names.
    df["product_id"] = df["Uniq Id"].map(clean_text)
    df["name"] = df["Product Name"].map(clean_text)

    # Remove records that cannot identify a product.
    df = df.loc[
        df["product_id"].ne("") & df["name"].ne("")
    ].copy()

    # Keep one record per product ID.
    df = df.drop_duplicates(
        subset=["product_id"],
        keep="first",
    )

    # Preserve the full hierarchy and extract broad categories.
    df["category"] = df["Category"].map(clean_text)
    df["category"] = df["category"].replace("", "Uncategorized")

    def get_subcategory(category):
        parts = [part.strip() for part in category.split("|")]
        return parts[1] if len(parts) > 1 else ""

    df["subcategory"] = df["category"].map(get_subcategory)

    # Normalize currency strings; missing prices remain missing.
    df["price"] = df["Selling Price"].map(clean_price)

    # Combine only source-provided product information.
    text_fields = [
        "About Product",
        "Product Specification",
        "Technical Details",
    ]

    def build_description(row):
        sections = []

        for field in text_fields:
            value = clean_text(row[field])

            if value:
                sections.append(value[:1500])

        return " ".join(sections)[:4000]

    df["description"] = df.apply(build_description, axis=1)

    # Retain useful source metadata.
    df["product_url"] = df["Product Url"].map(clean_text)
    df["image_url"] = df["Image"].map(
        lambda value: clean_text(value).split("|")[0]
    )

    # Brand is not a verified source column. Do not invent it.
    df["brand"] = ""

    # Select a reproducible 10,000-row catalogue.
    df = df.head(TARGET_ROWS).copy()

    output_columns = [
        "product_id",
        "name",
        "brand",
        "category",
        "subcategory",
        "price",
        "description",
        "product_url",
        "image_url",
    ]

    output = df[output_columns].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    # Record the source and processing choices.
    attribution_path = OUTPUT_PATH.parent / "SOURCE.txt"
    attribution_path.write_text(
        "Source dataset: calmgoose/amazon-product-data-2020\n"
        "Source page: "
        "https://huggingface.co/datasets/calmgoose/amazon-product-data-2020\n"
        "Raw file: data/raw/amazon_products.csv\n"
        "Processed file: data/processed/products.csv\n"
        "Processing: normalized text, parsed prices, retained "
        "source categories and URLs, combined source descriptions, "
        "deduplicated product IDs, and selected the first 10,000 "
        "valid records in source order.\n"
        "Brand: left blank because the source has no dedicated "
        "verified brand column.\n",
        encoding="utf-8",
    )

    print("Processed catalogue saved:", OUTPUT_PATH)
    print("Rows:", len(output))
    print("Unique product IDs:", output["product_id"].nunique())
    print("Missing prices:", output["price"].isna().sum())
    print("Missing descriptions:", output["description"].eq("").sum())
    print("Missing categories:", output["category"].eq("Uncategorized").sum())
    print("Missing brands:", output["brand"].eq("").sum())
    print("\nSample products:")
    print(output[["product_id", "name", "category", "price"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
