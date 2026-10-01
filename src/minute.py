import pandas as pd

file_path = "dataset/mining_cleaned.csv"

df = pd.read_csv(file_path)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort data by time
df = df.sort_values("date")

print("Dataset loaded successfully!")

print("\n--- ORIGINAL DATA ---")
print("Rows:", len(df))
print("Unique timestamps:", df["date"].nunique())

# Create 1-minute time periods
df["minute"] = df["date"].dt.floor("min")

# Count measurements in each minute
minute_counts = df.groupby("minute").size()

print("\n--- 1-MINUTE ANALYSIS ---")
print("Number of unique minutes:", len(minute_counts))

print("\nMeasurements per minute:")
print(minute_counts.describe())

print("\nMost common measurements per minute:")
print(minute_counts.value_counts().head(10))

print("\nFirst 20 minute periods:")
print(minute_counts.head(20))

print("\nMinute analysis completed!")