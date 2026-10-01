# Quality Prediction in a Mining Process

## Internship Project – UCT

### 1. Project Overview

This project focuses on predicting the percentage of silica concentrate in an iron ore flotation process using machine learning.

The project uses process data collected from a real flotation plant. Machine learning models are used to study the relationship between process variables and the final silica concentration.

---

## 2. Problem Statement

Silica concentration is an important quality measure of iron ore concentrate.

The objective of this project is to investigate whether machine learning can predict `% Silica Concentrate` using available process measurements.

The project also investigates:

1. Whether silica concentrate can be predicted at minute-level intervals.
2. How many hours ahead silica concentrate can be predicted.
3. Whether silica concentrate can be predicted without using `% Iron Concentrate`.

---

## 3. Dataset

Dataset:

`MiningProcess_Flotation_Plant_Database.csv`

The dataset contains process measurements from an iron ore flotation plant.

The data contains:

- Date and time
- % Iron Feed
- % Silica Feed
- Starch Flow
- Amina Flow
- Ore Pulp Flow
- Ore Pulp pH
- Ore Pulp Density
- Flotation column air flow measurements
- Flotation column level measurements
- % Iron Concentrate
- % Silica Concentrate

Target variable:

`% Silica Concentrate`

---

## 4. Data Preparation

The original dataset contains 737,453 rows and 24 columns.

The data was cleaned by:

- Converting the date column into datetime format.
- Converting decimal values into numeric values.
- Checking missing values.
- Sorting the data according to time.

The cleaned dataset contained no missing values.

---

## 5. Hourly Data Preparation

The original process data contains approximately 180 measurements for each hourly timestamp.

Since the laboratory quality measurements are hourly, the process measurements were aggregated to create one hourly record.

The resulting hourly dataset contains:

- 4,097 hourly records
- 24 columns

The process variables were aggregated using their hourly mean values.

The target `% Silica Concentrate` was taken from the corresponding hourly laboratory measurement.

---

## 6. Machine Learning Models

Two regression models were initially tested:

### Random Forest Regressor

MAE: 0.5867

RMSE: 0.7232

R²: 0.5989

### Gradient Boosting Regressor

MAE: 0.5602

RMSE: 0.6993

R²: 0.6249

---

## 7. Prediction Without % Iron Concentrate

UCT specifically asks whether `% Silica Concentrate` can be predicted without `% Iron Concentrate`.

A separate Gradient Boosting experiment was performed after removing `% Iron Concentrate` from the input features.

Results:

MAE: 0.8713

RMSE: 1.0744

R²: 0.1147

---

## 8. Minute-Level Prediction

The original process data contains approximately 180 measurements within an hour.

These measurements were aggregated into approximately one-minute intervals to construct a minute-level process dataset.

The resulting dataset contains:

245,818 minute-level records.

A Gradient Boosting model was trained using the minute-level process features.

Results:

MAE: 0.8805

RMSE: 1.1032

R²: 0.0672

Important limitation:

The laboratory `% Silica Concentrate` measurement is hourly. Therefore, the minute-level dataset represents a constructed minute-level prediction experiment rather than independent laboratory silica measurements collected every minute.

---

## 9. Hours-Ahead Prediction

Prediction experiments were performed for different future horizons.

| Hours Ahead | MAE | RMSE | R² |
|-------------:|----:|-----:|----:|
| 1 | 0.6976 | 0.8684 | 0.4217 |
| 2 | 0.7766 | 0.9554 | 0.3008 |
| 3 | 0.8171 | 1.0087 | 0.2206 |
| 6 | 0.9613 | 1.1717 | -0.0517 |
| 12 | 1.0309 | 1.2620 | -0.2175 |
| 24 | 1.0081 | 1.2396 | -0.1722 |

Among the tested horizons, the 1-hour-ahead experiment produced the lowest MAE and RMSE and the highest R².

---

## 10. Project Results

The experiments demonstrate that machine learning can be applied to the flotation-process data to model `% Silica Concentrate`.

The results also show that prediction performance changes when:

- The prediction interval changes.
- `% Iron Concentrate` is removed.
- The data is converted from hourly process information to minute-level process information.

---

## 11. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Gradient Boosting
- Random Forest
- VS Code
- Git
- GitHub

---

## 12. Project Structure

```text
Project10_Mining_Quality_Prediction/
│
├── dataset/
│   ├── MiningProcess_Flotation_Plant_Database.csv
│   ├── mining_cleaned.csv
│   ├── hourly_mining_data.csv
│   ├── minute_mining_data.csv
│   └── horizon_results.csv
│
├── notebooks/
│
├── src/
│   ├── check_dataset.py
│   ├── clean_dataset.py
│   ├── analyze_data.py
│   ├── analyze_time.py
│   ├── analyze_hour.py
│   ├── create_hourly_data.py
│   ├── prepare_ml_data.py
│   ├── train_test_split.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   ├── without_iron.py
│   ├── minute.py
│   ├── minute_data.py
│   ├── minute_model.py
│   └── horizon.py
│
└── README.md