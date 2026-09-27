import streamlit as st
import pandas as pd
import json
import joblib
import numpy as np

st.set_page_config(
    page_title="E-commerce Demand Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("E-commerce Demand Prediction & Inventory Alert System")
st.write("Predict next week SKU demand and identify products that may require restocking.")

forecast_df = pd.read_csv("data/next_week_forecast.csv")
weekly_demand = pd.read_csv("data/weekly_demand.csv",
                parse_dates=["week"]
)
validation_df = pd.read_csv("data/validation_predictions.csv",
                parse_dates=["week"]
)

with open(
    "models/model_metrics.json",
    "r"
) as file:
    metrics = json.load(file)

model = joblib.load(
    "models/demand_model.pkl"
)

total_skus = len(forecast_df)
total_predicted_demand = int(
    forecast_df["predicted_units"].sum()
)
forecast_horizon = "1 Week Ahead"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "Total SKUs",
        total_skus
    )
with col2:
    st.metric(
        "Predicted Weekly Demand",
        total_predicted_demand
    )
with col3:
    st.metric(
        "Forecast Horizon",
        forecast_horizon
    )
with col4:
    st.metric(
        "Selected Model",
        "Random Forest"
    )
st.caption(
    "Forecasts are generated one week ahead from the latest complete week available in the historical dataset."
)

st.subheader("Next-Week Demand Forecast")
forecast_display = forecast_df[
    [
        "sku",
        "product_title",
        "predicted_units"
    ]
].copy()

forecast_display.columns = [
    "SKU",
    "Product",
    "Predicted Demand"
]
forecast_display = forecast_display.sort_values(
    "Predicted Demand",
    ascending=False
)

st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True
)

st.divider()
st.subheader("Inventory Alert & Reorder Recommendations")
st.write(
    "Enter the current stock for each SKU. "
    "The system compares inventory with predicted next-week demand."
)
safety_buffer = st.slider(
    "Safety Stock Buffer (%)",
    min_value=0,
    max_value=50,
    value=20,
    step=5
)

inventory_input = forecast_df[
    [
        "sku",
        "product_title",
        "predicted_units"
    ]
].copy()
inventory_input["current_stock"] = 0

edited_inventory = st.data_editor(
    inventory_input,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "sku",
        "product_title",
        "predicted_units"
    ],
    column_config={
        "sku": "SKU",
        "product_title": "Product",
        "predicted_units": "Predicted Demand",
        "current_stock": st.column_config.NumberColumn(
            "Current Stock",
            min_value=0,
            step=1
        )
    }
)

buffer_decimal = safety_buffer / 100
edited_inventory["safety_stock"] = np.ceil(
    edited_inventory["predicted_units"] * buffer_decimal
).astype(int)

edited_inventory["required_stock"] = (
    edited_inventory["predicted_units"] + edited_inventory["safety_stock"]
)

edited_inventory["reorder_quantity"] = (
    edited_inventory["required_stock"] - edited_inventory["current_stock"]
).clip(lower=0)

def stock_status(row):
    if row["current_stock"] < row["predicted_units"]:
        return "REORDER"
    elif row["current_stock"] < row["required_stock"]:
        return "LOW STOCK"
    else:
        return "GOOD"

edited_inventory["status"] = edited_inventory.apply(
    stock_status,
    axis=1
)

reorder_products = (
    edited_inventory["status"] == "REORDER"
).sum()
low_stock_products = (
    edited_inventory["status"] == "LOW STOCK"
).sum()
good_stock_products = (
    edited_inventory["status"] == "GOOD"
).sum()

inv_col1, inv_col2, inv_col3 = st.columns(3)
with inv_col1:
    st.metric(
        "Reorder Required",
        int(reorder_products)
    )
with inv_col2:
    st.metric(
        "Low Stock",
        int(low_stock_products)
    )
with inv_col3:
    st.metric(
        "Stock Sufficient",
        int(good_stock_products)
    )
    
status_filter = st.selectbox(
    "Filter Inventory Status",
    [
        "ALL",
        "REORDER",
        "LOW STOCK",
        "GOOD"
    ]
)

if status_filter == "ALL":
    filtered_inventory = edited_inventory.copy()

else:
    filtered_inventory = edited_inventory[
        edited_inventory["status"] == status_filter
    ]

st.dataframe(
    filtered_inventory[
        [
            "sku",
            "product_title",
            "predicted_units",
            "current_stock",
            "safety_stock",
            "required_stock",
            "reorder_quantity",
            "status"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.divider()
st.subheader("Sales & Demand Analytics")

weekly_total = (
    weekly_demand
    .groupby("week", as_index=False)["units_sold"]
    .sum()
)
st.write("### Historical Weekly Demand")
st.line_chart(
    weekly_total,
    x="week",
    y="units_sold"
)

top_products = (
    forecast_df[
        [
            "sku",
            "predicted_units"
        ]
    ]
    .sort_values(
        "predicted_units",
        ascending=False
    )
    .head(10)
)

st.write("### Top 10 SKUs by Predicted Demand")
st.bar_chart(
    top_products,
    x="sku",
    y="predicted_units"
)

st.divider()
st.subheader("Model Performance")

rf_metrics = metrics["random_forest"]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "MAE",
        f'{rf_metrics["mae"]:.3f}'
    )
with col2:
    st.metric(
        "RMSE",
        f'{rf_metrics["rmse"]:.3f}'
    )
with col3:
    st.metric(
        "R²",
        f'{rf_metrics["r2"]:.3f}'
    )