import pandas as pd

# Load hourly dataset
file_path = "dataset/hourly_mining_data.csv"
df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort data by time
df = df.sort_values("date")

# Target variable
target = "% Silica Concentrate"

# Features
features = [
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
    "Flotation Column 07 Level",
    "% Iron Concentrate"
]

# Create input features and target
X = df[features]
y = df[target]

print("ML data prepared successfully!")

print("\nNumber of samples:", len(X))
print("Number of features:", len(X.columns))

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(target)

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 target values:")
print(y.head())