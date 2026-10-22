
from pathlib import Path
from datasets import load_dataset


DATASET_NAME = "calmgoose/amazon-product-data-2020"
OUTPUT_PATH = Path("data/raw/amazon_products.csv")


def main():
    print(f"Downloading dataset: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    print("\nOriginal dataset rows:", len(dataset))
    print("Original columns:", dataset.column_names)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(str(OUTPUT_PATH))

    print("\nDataset saved to:", OUTPUT_PATH)
    print("Rows saved:", len(dataset))


if __name__ == "__main__":
    main()
