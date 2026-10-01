import pandas as pd

file_path = "dataset/MiningProcess_Flotation_Plant_Database.csv"

df = pd.read_csv(
    file_path,
    sep=",",
    quotechar='"'
)

print("Dataset loaded successfully!")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())