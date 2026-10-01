import pandas as pd

# Load cleaned dataset
file_path = "dataset/mining_cleaned.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Basic information
print("\n--- DATASET INFORMATION ---")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Date information
df["date"] = pd.to_datetime(df["date"])

print("\n--- DATE INFORMATION ---")
print("Starting date:", df["date"].min())
print("Ending date:", df["date"].max())

# Target information
target = "% Silica Concentrate"

print("\n--- SILICA CONCENTRATE ---")
print("Minimum:", df[target].min())
print("Maximum:", df[target].max())
print("Average:", df[target].mean())
print("Median:", df[target].median())

# Target statistics
print("\n--- TARGET STATISTICS ---")
print(df[target].describe())

# Unique dates
print("\n--- TIME INFORMATION ---")
print("Number of unique timestamps:", df["date"].nunique())

# Check repeated timestamps
print("\n--- FIRST 10 TIMESTAMPS ---")
print(df["date"].head(10).to_string(index=False))

# Correlation with target
print("\n--- CORRELATION WITH SILICA CONCENTRATE ---")

numeric_columns = df.select_dtypes(include="number").columns

correlation = (
    df[numeric_columns]
    .corr()["% Silica Concentrate"]
    .sort_values(ascending=False)
)

print(correlation)

print("\nAnalysis completed!")