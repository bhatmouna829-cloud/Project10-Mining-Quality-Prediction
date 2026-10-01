import pandas as pd

# Load hourly dataset
file_path = "dataset/hourly_mining_data.csv"
df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort by time
df = df.sort_values("date")

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

# Target
target = "% Silica Concentrate"

X = df[features]
y = df[target]

# Use 80% for training and 20% for testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Train-test split completed!")

print("\nTotal samples:", len(df))

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining period:")
print(df["date"].iloc[0], "to", df["date"].iloc[split_index - 1])

print("\nTesting period:")
print(df["date"].iloc[split_index], "to", df["date"].iloc[-1])

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\ny_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)