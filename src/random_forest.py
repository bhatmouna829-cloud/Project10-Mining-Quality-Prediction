import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Time-based train/test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
print("Training Random Forest model...")
model.fit(X_train, y_train)

print("Model training completed!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Display results
print("\n--- MODEL RESULTS ---")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

print("\n--- SAMPLE PREDICTIONS ---")

results = pd.DataFrame({
    "Actual Silica": y_test.values[:10],
    "Predicted Silica": y_pred[:10]
})

print(results)