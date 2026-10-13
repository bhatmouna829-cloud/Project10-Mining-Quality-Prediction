# UCT Machine Learning Internship Project Report

## Project 10: Quality Prediction in a Mining Process

**Student Name:** Mouna Bhat  
**Company:** UCT  
**Domain:** Machine Learning

## 1. Introduction

This project focuses on predicting the percentage of silica concentrate in an iron ore flotation plant using machine learning techniques. The goal is to analyse industrial process data and develop models that can estimate concentrate quality.

Silica is an impurity in iron ore concentrate. Predicting its concentration can help engineers monitor the process and investigate possible changes in product quality.

## 2. Problem Statement

The objective is to predict `% Silica Concentrate` using the supplied flotation-process dataset.

The project investigates three main questions:

1. Can silica concentrate be predicted at minute-level resolution?
2. How many hours ahead can silica concentrate be predicted?
3. Can silica concentrate be predicted without using `% Iron Concentrate`?

## 3. Dataset Description

The dataset contains industrial process measurements from an iron ore flotation plant.

- Original rows: 737,453
- Original columns: 24
- Data period: March 2017 to September 2017
- Target variable: `% Silica Concentrate`
- Missing values after cleaning: 0
- Hourly records after aggregation: 4,097

The dataset includes feed-quality measurements, reagent flow, ore pulp properties, flotation-column air flow and levels, and concentrate quality measurements.

## 4. Data Preprocessing

The dataset was loaded using Pandas. The date column was converted to datetime format, and numeric values containing decimal commas were converted into standard numeric values.

The process measurements were organised by hourly timestamp. The mean of each process feature was calculated for each hour, while the corresponding hourly quality measurements were retained.

The resulting hourly dataset contained 4,097 records. The data was sorted chronologically and divided into 80% training data and 20% testing data.

## 5. Methodology and Implementation

The project was implemented in Python using Pandas and Scikit-learn.

The workflow included:

1. Loading and inspecting the dataset.
2. Cleaning numeric values and date information.
3. Analysing the target variable and feature correlations.
4. Aggregating process measurements into hourly records.
5. Splitting data chronologically into training and testing sets.
6. Training regression models.
7. Evaluating the predictions using MAE, RMSE, and R².
8. Testing prediction without `% Iron Concentrate`.
9. Evaluating minute-level and future-hour prediction experiments.

The main Gradient Boosting model used 100 estimators, a learning rate of 0.05, and a maximum tree depth of 3.

## 6. Main Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.5867 | 0.7232 | 0.5989 |
| Gradient Boosting | 0.5602 | 0.6993 | 0.6249 |

Gradient Boosting achieved better results than Random Forest in the main experiment. Its R² score of 0.6249 indicates that it explained approximately 62.5% of the variation in the target values in the test set.

## 7. Prediction Without Iron Concentrate

A separate Gradient Boosting experiment was performed after removing `% Iron Concentrate` from the input features.

| Metric | Result |
|---|---:|
| MAE | 0.8713 |
| RMSE | 1.0744 |
| R² | 0.1147 |

The model performed less effectively without `% Iron Concentrate`. This suggests that this variable provides useful predictive information for estimating silica concentrate in the supplied dataset.

## 8. Minute-Level Prediction

A minute-resolution dataset was constructed by grouping the 20-second process readings into minute intervals within each hour. The hourly laboratory silica value was associated with the constructed minute records.

The resulting dataset contained 245,818 rows.

| Metric | Result |
|---|---:|
| MAE | 0.8805 |
| RMSE | 1.1032 |
| R² | 0.0672 |

**Limitation:** The supplied laboratory silica measurements are hourly, not independent measurements every minute. Therefore, this experiment uses a constructed minute-resolution representation with the hourly target repeated across minute records. It does not establish that reliable independent laboratory silica predictions are available every minute.

## 9. Hours-Ahead Prediction

The target was shifted forward to evaluate predictions at different future horizons.

| Forecast Horizon | MAE | RMSE | R² |
|---|---:|---:|---:|
| 1 hour | 0.6976 | 0.8684 | 0.4217 |
| 2 hours | 0.7766 | 0.9554 | 0.3008 |
| 3 hours | 0.8171 | 1.0087 | 0.2206 |
| 6 hours | 0.9613 | 1.1717 | -0.0517 |
| 12 hours | 1.0309 | 1.2620 | -0.2175 |
| 24 hours | 1.0081 | 1.2396 | -0.1722 |

Among the tested horizons, the one-hour-ahead model performed best. The longer horizons generally produced larger errors and lower R² scores. These experiments do not establish the absolute maximum possible prediction horizon.

## 10. Key Findings

- The dataset was successfully cleaned and analysed.
- Hourly aggregation reduced the dataset to 4,097 records.
- Gradient Boosting performed better than Random Forest in the main experiment.
- Removing `% Iron Concentrate` reduced predictive performance.
- The constructed minute-level experiment had limited predictive performance and requires cautious interpretation.
- Among the tested forecasting horizons, one-hour-ahead prediction performed best.

## 11. Learning Outcomes

Through this project, I learned how to:

- Work with a large real-world industrial dataset.
- Clean and preprocess data using Pandas.
- Convert date and numeric columns
