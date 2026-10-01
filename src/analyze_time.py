import pandas as pd

# Load cleaned dataset
file_path = "dataset/mining_cleaned.csv"
df = pd.read_csv(file_path)

# Convert date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

print("Time analysis started!")

# --------------------------------------------------
# 1. Date range
# --------------------------------------------------

print("\n--- DATE RANGE ---")
print("Start:", df["date"].min())
print("End:", df["date"].max())

# --------------------------------------------------
# 2. Number of unique timestamps
# --------------------------------------------------

print("\n--- UNIQUE TIMESTAMPS ---")
print("Total unique timestamps:", df["date"].nunique())

# --------------------------------------------------
# 3. Number of rows for each timestamp
# --------------------------------------------------

rows_per_timestamp = df.groupby("date").size()

print("\n--- ROWS PER TIMESTAMP ---")
print(rows_per_timestamp.describe())

# --------------------------------------------------
# 4. Most common number of rows per timestamp
# --------------------------------------------------

print("\n--- MOST COMMON ROW COUNTS ---")
print(rows_per_timestamp.value_counts().head(10))

# --------------------------------------------------
# 5. Difference between consecutive timestamps
# --------------------------------------------------

unique_dates = df["date"].drop_duplicates().sort_values()

time_difference = unique_dates.diff().dropna()

print("\n--- TIME DIFFERENCE ---")
print(time_difference.describe())

print("\nMost common time intervals:")
print(time_difference.value_counts().head(10))

# --------------------------------------------------
# 6. Show first 20 unique timestamps
# --------------------------------------------------

print("\n--- FIRST 20 UNIQUE TIMESTAMPS ---")
print(unique_dates.head(20).to_string(index=False))

# --------------------------------------------------
# 7. Check whether hourly timestamps exist
# --------------------------------------------------

hourly_dates = unique_dates.dt.floor("h")

print("\n--- HOURLY CHECK ---")
print("Unique hourly periods:", hourly_dates.nunique())

# --------------------------------------------------
# Finished
# --------------------------------------------------

print("\nTime analysis completed!")