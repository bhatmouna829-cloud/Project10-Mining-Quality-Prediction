import pandas as pd

# Load cleaned dataset
file_path = "dataset/mining_cleaned.csv"
df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

# Process measurement columns
feature_columns = [
    "% Iron Feed",
    "% Silica Feed",
    "Starch Flow",
    "Amina Flow",
    "Ore Pulp Flow",
    "Ore Pulp pH",
    "Ore Pulp Density",
    "Flotation Column 01 Air Flow",
    "Flotation Column 02 Air Flow",
    "Flotation Column 03 Air Flow",
    "Flotation Column 04 Air Flow",
    "Flotation Column 05 Air Flow",
    "Flotation Column 06 Air Flow",
    "Flotation Column 07 Air Flow",
    "Flotation Column 01 Level",
    "Flotation Column 02 Level",
    "Flotation Column 03 Level",
    "Flotation Column 04 Level",
    "Flotation Column 05 Level",
    "Flotation Column 06 Level",
    "Flotation Column 07 Level"
]

# Calculate hourly average process values
hourly_features = df.groupby("date")[feature_columns].mean()

# Get one hourly laboratory result
hourly_target = df.groupby("date")[
    ["% Iron Concentrate", "% Silica Concentrate"]
].first()

# Combine everything
hourly_df = pd.concat(
    [hourly_features, hourly_target],
    axis=1
).reset_index()

# Make sure target columns are numeric
hourly_df["% Iron Concentrate"] = pd.to_numeric(
    hourly_df["% Iron Concentrate"],
    errors="coerce"
)

hourly_df["% Silica Concentrate"] = pd.to_numeric(
    hourly_df["% Silica Concentrate"],
    errors="coerce"
)

# Save hourly dataset
output_path = "dataset/hourly_mining_data.csv"
hourly_df.to_csv(output_path, index=False)

# Display results
print("Hourly dataset created successfully!")

print("\nOriginal rows:", len(df))
print("Hourly rows:", len(hourly_df))
print("Columns:", len(hourly_df.columns))

print("\nHourly dataset shape:")
print(hourly_df.shape)

print("\nIron Concentrate:")
print(hourly_df["% Iron Concentrate"].head())

print("\nSilica Concentrate:")
print(hourly_df["% Silica Concentrate"].head())

print("\nMissing values:")
print(hourly_df.isnull().sum().sum())

print("\nSaved to:")
print(output_path)