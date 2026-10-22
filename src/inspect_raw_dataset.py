
import pandas as pd

path = "data/raw/amazon_products.csv"
df = pd.read_csv(path)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 3 rows:")
print(df.head(3).to_string())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).to_string())

print("\nDuplicate rows:", df.duplicated().sum())
