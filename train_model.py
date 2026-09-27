import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import json


df = pd.read_csv("data/weekly_demand.csv")
df["week"] = pd.to_datetime(df["week"])
df= df.sort_values(["sku", "week"]).reset_index(drop=True)
print(df.head())
print("\nDataset shape:")
print(df.shape)

for lag in [1, 2, 3, 4, 5]:
    df[f"lag_{lag}"] = (
        df.groupby("sku")["units_sold"]
        .shift(lag)
    )


df["month"] = df["week"].dt.month
df["week_of_year"] = (
    df["week"].dt.isocalendar()
    .week
    .astype(int)
)

model_data = df.dropna(
    subset=[
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4"
    ]
).copy()

print("\nModel-ready data:")
print(
    model_data[
        [
            "week",
            "sku",
            "units_sold",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "month",
            "week_of_year"
        ]
    ].head(20).to_string(index=False)
)
print("\nModel datset shape:")
print(model_data.shape)

unique_weeks = sorted(model_data["week"].unique())
test_weeks = unique_weeks[-4:]
train_data = model_data[
    ~model_data["week"].isin(test_weeks)
].copy()
test_data = model_data[
    model_data["week"].isin(test_weeks)
].copy()

print("\nTraining period:")
print(train_data["week"].min(), "to", train_data["week"].max())
print("\nTesting period:")
print(test_data["week"].min(), "to", test_data["week"].max())
print("\nTraining rows:")
print(len(train_data))
print("\nTesting rows:")
print(len(test_data))

features = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "month",
    "week_of_year"
]
target = "units_sold"

x_train = train_data[features]
y_train = train_data[target]
x_test = test_data[features]
y_test = test_data[target]

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
linear_model_predictions = linear_model.predict(x_test)
linear_predictions = np.maximum(linear_model_predictions, 0)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("\nLinear Regression Performance:")
print(f"MAE: {linear_mae:.3f}")
print(f"RMSE: {linear_rmse:.3f}")
print(f"R2 Score: {linear_r2:.3f}")

random_forest_model = RandomForestRegressor(n_estimators=200, random_state=42)
random_forest_model.fit(x_train, y_train)
rf_predictions = random_forest_model.predict(x_test)
rf_predictions = np.maximum(rf_predictions, 0)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Performance:")
print(f"MAE: {rf_mae:.3f}")
print(f"RMSE: {rf_rmse:.3f}")
print(f"R2 Score: {rf_r2:.3f}")


non_zero_mask = y_test > 0
rf_non_zero_mae = mean_absolute_error(y_test[non_zero_mask],rf_predictions[non_zero_mask])
print("\nRandom Forest MAE on non-zero demand weeks:")
print(f"MAE: {rf_non_zero_mae:.3f}")
print("\nNon-zero test rows:")
print(non_zero_mask.sum())
print("\nTotal test rows:")
print(len(y_test))

metrics = {
    "linear_regression": {
        "mae": float(linear_mae),
        "rmse": float(linear_rmse),
        "r2": float(linear_r2)
    },
    "random_forest": {
        "mae": float(rf_mae),
        "rmse": float(rf_rmse),
        "r2": float(rf_r2)
    },
    "random_forest_non_zero_mae": float(rf_non_zero_mae)
}

with open("models/model_metrics.json","w") as file:
    json.dump(metrics,file,indent=4)
print("\nModel metrics saved to models/model_metrics.json")

X_all = model_data[features]
y_all = model_data[target]
final_model = RandomForestRegressor(
    n_estimators=200,random_state=42
)
final_model.fit(X_all, y_all)

joblib.dump(final_model, "models/demand_model.pkl")
print("\nFinal model saved to models/demand_model.pkl")
print("\nFinal Random Forest trained on all available historical data.")


latest_week = df["week"].max()
next_week = latest_week + pd.Timedelta(weeks=1)
print("\nLatest complete week:")
print(latest_week)
print("\nForecast week:")
print(next_week)

forecast_rows = []
for sku in df["sku"].unique():
    sku_data = (
        df[df["sku"] == sku]
        .sort_values("week")
    )

    if len(sku_data) < 4:
        continue
    recent_4 = sku_data.tail(4)
    forecast_rows.append({
        "sku": sku,

        "fsn": sku_data["fsn"].iloc[-1],

        "product_title":
            sku_data["product_title"].iloc[-1],

        "lag_1":
            recent_4["units_sold"].iloc[-1],

        "lag_2":
            recent_4["units_sold"].iloc[-2],

        "lag_3":
            recent_4["units_sold"].iloc[-3],

        "lag_4":
            recent_4["units_sold"].iloc[-4],

        "month":
            next_week.month,

        "week_of_year":
            int(next_week.isocalendar().week)
    })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df["predicted_demand"] = final_model.predict(
    forecast_df[features]
)

forecast_df["predicted_demand"] = (
    forecast_df["predicted_demand"]
    .clip(lower=0)
)

forecast_df["predicted_units"] = (
    forecast_df["predicted_demand"]
    .round()
    .astype(int)
)

forecast_df["forecast_week"] = next_week

print("\nNext-week demand forecast:")
print(
    forecast_df[
        [
            "forecast_week",
            "sku",
            "product_title",
            "predicted_demand",
            "predicted_units"
        ]
    ]
    .sort_values(
        "predicted_demand",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)

forecast_df.to_csv("data/next_week_forecast.csv",index=False)

print("\nForecast saved to data/next_week_forecast.csv")

validation_results = test_data[
    [
        "week",
        "sku",
        "product_title",
        "units_sold"
    ]
].copy()

validation_results["linear_prediction"] = linear_predictions
validation_results["rf_prediction"] = rf_predictions
validation_results.rename(
    columns={
        "units_sold": "actual_demand"
    },
    inplace=True
)

validation_results.to_csv(
    "data/validation_predictions.csv",
    index=False
)
print("\nValidation predictions saved to data/validation_predictions.csv")