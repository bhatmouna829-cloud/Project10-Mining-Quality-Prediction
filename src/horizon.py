import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load hourly dataset
file_path = "dataset/hourly_mining_data.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

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

target = "% Silica Concentrate"

# Prediction horizons in hours
horizons = [1, 2, 3, 6, 12, 24]

print("Starting hours-ahead prediction...\n")

results = []

for hours in horizons:

    print("Testing prediction", hours, "hour(s) ahead...")

    # Shift target into the future
    future_target = df[target].shift(-hours)

    valid_data = df.copy()
    valid_data["future_target"] = future_target

    # Remove rows without future target
    valid_data = valid_data.dropna(
        subset=["future_target"]
    ).reset_index(drop=True)

    X = valid_data[features]
    y = valid_data["future_target"]

    # Time-based 80/20 split
    split_index = int(len(valid_data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # Model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Hours Ahead": hours,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
    print()

# Results table
results_df = pd.DataFrame(results)

print("\n--- FINAL HORIZON RESULTS ---")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(
    "dataset/horizon_results.csv",
    index=False
)

print("\nResults saved to:")
print("dataset/horizon_results.csv")