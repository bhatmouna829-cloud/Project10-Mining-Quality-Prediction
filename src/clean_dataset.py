import pandas as pd

# Dataset path
file_path = "dataset/MiningProcess_Flotation_Plant_Database.csv"

# Load dataset
df = pd.read_csv(
    file_path,
    sep=",",
    quotechar='"'
)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Convert numeric columns
for column in df.columns[1:]:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Save cleaned dataset
output_path = "dataset/mining_cleaned.csv"
df.to_csv(output_path, index=False)

print("Data cleaning completed!")

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nTarget column:")
print(df["% Silica Concentrate"].head())

print("\nCleaned dataset saved to:")
print(output_path)