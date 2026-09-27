import pandas as pd
file_path = "data/flipkart_orders.xlsx"
df = pd.read_excel(file_path, sheet_name="Orders")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumns:")
print(df.columns.tolist())

df = df[
    [
        "order_date",
        "order_item_status",
        "sku",
        "fsn",
        "product_title",
        "quantity"
    ]
]

print("\nSelected Columns:")
print(df.head())
print("\nNew shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

df["order_date"] = pd.to_datetime(df["order_date"])

text_columns = ["order_item_status", "sku", "fsn", "product_title"]

for column in text_columns:
    df[column] = (
        df[column]
        .str.replace('"', '', regex=False)
        .str.strip()
    )

df["order_item_status"] = df["order_item_status"].str.upper()

print("\nCleaned data:")
print(df.head())

print("\nData types after cleaning:")
print(df.dtypes)

print("\nOrder statuses:")
print(df["order_item_status"].value_counts())

print("\nDate range:")
print("First order:", df["order_date"].min())
print("Last order:", df["order_date"].max())

excluded_statuses = ["CANCELLED", "REJECTED"]
demand_df = df[
    ~df["order_item_status"].isin(excluded_statuses)
].copy()

print("\nRows before filtering:")
print(len(df))
print("\nRows after filtering:")
print(len(demand_df))
print("\nDemand order statuses:")
print(demand_df["order_item_status"].value_counts())
print("\nTotal demand units:")
print(demand_df["quantity"].sum())

demand_df["week"] = (
    demand_df["order_date"]
    .dt.to_period("W")
    .apply(lambda x: x.start_time)
)
print("\nOrder dates with week:")
print(demand_df[["order_date", "week"]].head(10))

weekly_sales = (
    demand_df
    .groupby(
        ["week", "sku", "fsn", "product_title"],
        as_index=False
    )["quantity"]
    .sum()
)
weekly_sales.rename(
    columns={"quantity": "units_sold"},
    inplace=True
)
print("\nWeekly SKU demand:")
print(weekly_sales.head(20))
print("\nWeekly dataset shape:")
print(weekly_sales.shape)

last_order_date = demand_df["order_date"].max()
last_week_start = last_order_date.to_period("W").start_time

if last_order_date.weekday() != 6:
    last_complete_week = last_week_start - pd.Timedelta(weeks=1)
else:
    last_complete_week = last_week_start
weekly_sales = weekly_sales[
    weekly_sales["week"] <= last_complete_week
].copy()

print("\nLast complete week:")
print(last_complete_week)

completed_weekly_data = []
for sku in weekly_sales["sku"].unique():
    sku_data = weekly_sales[
        weekly_sales["sku"] == sku
    ].copy()

    sku_data = sku_data.sort_values("week")
    first_week = sku_data["week"].min()
    all_weeks = pd.date_range(
        start=first_week,
        end=last_complete_week,
        freq="W-MON"
    )
    sku_data = (
        sku_data
        .set_index("week")
        .reindex(all_weeks)
    )
    sku_data.index.name = "week"
    sku_data["sku"] = sku
    sku_data["units_sold"] = sku_data["units_sold"].fillna(0)

    sku_data["fsn"] = sku_data["fsn"].ffill().bfill()
    sku_data["product_title"] = sku_data["product_title"].ffill().bfill()
    sku_data = sku_data.reset_index()
    completed_weekly_data.append(sku_data)

    weekly_sales_complete = pd.concat(
    completed_weekly_data,
    ignore_index=True
)

weekly_sales_complete = weekly_sales_complete.sort_values(
    ["sku", "week"]
).reset_index(drop=True)

print("\nCompleted weekly dataset:")
print(weekly_sales_complete.head(20))
print("\nCompleted dataset shape:")
print(weekly_sales_complete.shape)
print("\nZero-demand weeks:")
print((weekly_sales_complete["units_sold"] == 0).sum())

weekly_sales_complete.to_csv(
    "data/weekly_demand.csv",
    index=False
)
print("\nSaved cleaned weekly data to data/weekly_demand.csv")
