
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "product_id",
    "name",
    "category",
    "price",
    "description",
}


def load_products(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    df.columns = df.columns.astype(str).str.strip().str.lower()

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Clean text fields without turning missing values into "nan".
    for column in [
        "name",
        "brand",
        "category",
        "subcategory",
        "description",
        "product_url",
        "image_url",
    ]:
        if column not in df.columns:
            df[column] = ""

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["product_id"] = df["product_id"].astype("string").str.strip()

    invalid_ids = df["product_id"].isna() | df["product_id"].eq("")

    if invalid_ids.any():
        raise ValueError("Missing or empty product IDs found.")

    if df["product_id"].duplicated().any():
        duplicate_ids = df.loc[
            df["product_id"].duplicated(keep=False),
            "product_id",
        ].tolist()

        raise ValueError(f"Duplicate product IDs found: {duplicate_ids[:10]}")

    df = df.loc[df["name"].ne("")].copy()

    raw_prices = df["price"].copy()
    df["price"] = pd.to_numeric(raw_prices, errors="coerce")

    invalid_prices = raw_prices.notna() & df["price"].isna()

    if invalid_prices.any():
        print(
            f"Warning: {invalid_prices.sum()} invalid prices "
            "were converted to missing values."
        )

    if df["price"].lt(0).any():
        raise ValueError("Negative product prices found.")

    df = df.reset_index(drop=True)

    print(f"Products loaded: {len(df)}")
    print(f"Unique product IDs: {df['product_id'].nunique()}")
    print(f"Missing prices: {df['price'].isna().sum()}")
    print(f"Missing brands: {df['brand'].eq('').sum()}")

    return df


if __name__ == "__main__":
    products = load_products("data/processed/products.csv")
    print(products.head(3).to_string(index=False))
