# E-commerce Demand Prediction & Inventory Alert System

A machine learning based application that predicts one-week-ahead SKU-level product demand using historical e-commerce sales data and provides inventory restocking recommendations based on current stock and a configurable safety-stock buffer.

## Live Demo

[Open the deployed Streamlit application](https://github.com/imrozakram/ecommerce-demand-prediction/blob/main/README.md)

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

## Train-Test Strategy

Because this is a time-dependent forecasting problem, the dataset was **not randomly shuffled**.

The latest four complete weeks were kept as unseen test data, while all earlier weeks were used for training.

This chronological split better represents a real forecasting scenario where past data is used to predict future demand.

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

## Forecast Horizon

The deployed application performs a:

**1 Week Ahead Forecast**

Forecasts are generated one week ahead from the latest complete week available in the historical dataset.

The application is not connected to a live marketplace API, so it uses the processed historical dataset included in the project.

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

How to Run Locally

```bash
git clone https://github.com/imrozakram/ecommerce-demand-prediction
cd ecommerce-demand-prediction
pip install -r requirements.txt
streamlit run app.py

Project Workflow

Raw E-commerce Order Data
        ↓
Data Cleaning & Preprocessing
        ↓
Weekly SKU-Level Demand
        ↓
Zero-Demand Week Handling
        ↓
Feature Engineering
        ↓
Chronological Train-Test Split
        ↓
Linear Regression
        ↓
Random Forest Regressor
        ↓
Model Evaluation
        ↓
Random Forest Selected
        ↓
One-Week-Ahead Demand Forecast
        ↓
Current Inventory Input
        ↓
Safety Stock Calculation
        ↓
Inventory Alert & Reorder Recommendation
        ↓
Streamlit Dashboard

Limitations
- The dataset contains relatively sparse SKU-level demand.
- Many SKUs have weeks with zero demand.
- The dataset cannot always distinguish between genuine zero demand and temporary stock unavailability.
- Pricing changes are not currently included as model features.
- Advertising and promotional activity are not included.
- Marketplace events and external seasonal factors are not explicitly modeled.
- The project currently uses a fixed historical dataset rather than a live marketplace API.
- Forecasting performance depends on the amount and quality of historical data available.

Future Improvements
Possible improvements include:
- Integration with live marketplace APIs
- Automatic inventory synchronization
- Inclusion of product pricing and discount information
- Advertising and promotional features
- Longer historical sales data
- Product-category-level forecasting
- Seasonal and festival-related features
- Automated periodic model retraining

Author
Md Imroz Akram
B.Tech — Artificial Intelligence & Machine Learning