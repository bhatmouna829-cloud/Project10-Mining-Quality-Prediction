import pandas as pd

file_path = "dataset/mining_cleaned.csv"

df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort by hour
df = df.sort_values("date")

# Process columns
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
    "Flotation Column 07 Level"
]

target = "% Silica Concentrate"

# Give each measurement a position within its hour
df["measurement_number"] = df.groupby("date").cumcount()

# Convert 180 measurements into approximately one reading per 20 seconds
df["seconds"] = df["measurement_number"] * 20

# Create minute number within each hour
df["minute_number"] = (df["seconds"] // 60).astype(int)

# Aggregate measurements belonging to the same minute
minute_features = df.groupby(
    ["date", "minute_number"]
)[features].mean()

# Take one laboratory target value for each hour
hourly_target = df.groupby("date")[target].first()

# Add hourly target to every minute of that hour
minute_df = minute_features.reset_index()

minute_df[target] = minute_df["date"].map(hourly_target)

# Create actual minute timestamp
minute_df["minute_timestamp"] = (
    minute_df["date"]
    + pd.to_timedelta(minute_df["minute_number"], unit="m")
)

# Rearrange columns
minute_df = minute_df[
    ["minute_timestamp"] + features + [target]
]

# Save
output_path = "dataset/minute_mining_data.csv"

minute_df.to_csv(output_path, index=False)

print("Minute-level dataset created successfully!")

print("\nRows:", len(minute_df))
print("Columns:", len(minute_df.columns))

print("\nFirst 10 rows:")
print(minute_df.head(10))

print("\nMissing values:")
print(minute_df.isnull().sum().sum())

print("\nSaved to:")
print(output_path)