# E-commerce Demand Prediction & Inventory Alert System

A machine learning based application that predicts one-week-ahead SKU-level product demand using historical e-commerce sales data and provides inventory restocking recommendations based on current stock and a configurable safety-stock buffer.

## Live Demo

Streamlit deployment link will be added here after deployment.

## Project Overview

Managing inventory across multiple e-commerce SKUs can be difficult when product demand changes over time.

This project uses historical marketplace order data to predict future weekly demand for individual SKUs. The predicted demand is then compared with current inventory to identify products that may require restocking.

The application provides:

- One-week-ahead SKU demand forecasting
- Current inventory input
- Adjustable safety-stock buffer
- Reorder quantity calculation
- Inventory status classification
- Historical demand analytics
- Product-level demand visualization
- Model performance metrics

## Dataset

The project was developed using historical Flipkart order data from a real small-scale e-commerce business.

The raw dataset contained approximately:

- 1,967 order-item records
- Historical data from December 2025 to September 2026
- Multiple product SKUs
- Order status, SKU, product title, quantity and order date information

For privacy reasons, the original marketplace order report is not included in this repository.

Only aggregated SKU-level weekly demand data is used in the public repository.

## Data Preprocessing

The preprocessing pipeline performs the following steps:

1. Loads the Flipkart Orders sheet.
2. Selects relevant columns.
3. Converts order dates into datetime format.
4. Standardizes text fields and order statuses.
5. Excludes cancelled and rejected orders.
6. Aggregates individual orders into weekly SKU-level demand.
7. Creates zero-demand records for missing weeks after each SKU first appears.
8. Removes the final incomplete week.
9. Saves the processed weekly demand dataset.

## Feature Engineering

The forecasting models use historical demand features:

- Demand 1 week ago (`lag_1`)
- Demand 2 weeks ago (`lag_2`)
- Demand 3 weeks ago (`lag_3`)
- Demand 4 weeks ago (`lag_4`)
- Month
- Week of year

These features allow the model to learn from recent sales patterns while preventing future information from leaking into training.

## Models

Two regression models were evaluated:

### Linear Regression

Used as a simple baseline machine learning model.

### Random Forest Regressor

Used to capture nonlinear relationships between historical demand and future product demand.

The models were evaluated using a chronological train-test split, where the latest four complete weeks were kept as unseen test data.

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.679 | 1.384 | 0.356 |
| Random Forest | 0.672 | 1.345 | 0.391 |

Random Forest Regressor was selected as the final forecasting model because it achieved lower MAE and RMSE and a higher R² score on the test period.

## Inventory Alert Logic

The application compares predicted weekly demand with the current stock entered by the user.

A configurable safety-stock percentage is added to predicted demand.

Inventory is classified into three categories:

- **GOOD** — sufficient inventory including safety stock
- **LOW STOCK** — enough stock for predicted demand but insufficient safety stock
- **REORDER** — current stock is below predicted demand

The application also calculates the recommended reorder quantity.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Linear Regression
- Streamlit
- Joblib

## Project Structure

```text
Ecommerce-demand-prediction/
│
├── data/
│   ├── weekly_demand.csv
│   ├── next_week_forecast.csv
│   └── validation_predictions.csv
│
├── models/
│   ├── demand_model.pkl
│   └── model_metrics.json
│
├── app.py
├── data_preprocessing.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore