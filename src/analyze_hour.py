import pandas as pd

# Load cleaned dataset
file_path = "dataset/mining_cleaned.csv"
df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Select the first hour
first_hour = df[df["date"] == "2017-03-10 01:00:00"]

print("First hour selected successfully!")

print("\n--- NUMBER OF MEASUREMENTS ---")
print("Measurements in first hour:", len(first_hour))

print("\n--- SILICA CONCENTRATE ---")
print(first_hour["% Silica Concentrate"].describe())

print("\n--- IRON CONCENTRATE ---")
print(first_hour["% Iron Concentrate"].describe())

print("\n--- SILICA VALUES ---")
print(first_hour["% Silica Concentrate"].value_counts())

print("\n--- FIRST 10 ROWS ---")
print(
    first_hour[
        [
            "date",
            "% Iron Feed",
            "% Silica Feed",
            "Starch Flow",
            "Amina Flow",
            "Ore Pulp Flow",
            "Ore Pulp pH",
            "% Iron Concentrate",
            "% Silica Concentrate"
        ]
    ].head(10)
)

print("\n--- UNIQUE SILICA VALUES ---")
print(
    "Unique silica values:",
    first_hour["% Silica Concentrate"].nunique()
)

print("\nAnalysis completed!")