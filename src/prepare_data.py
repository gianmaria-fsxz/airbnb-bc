from typing import Optional
import pandas as pd
from utils.io_utils import read_and_rename
import yaml
import os

filename = os.path.abspath(__file__)
CONFIG_PATH = os.path.join(os.path.dirname(filename), os.pardir, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

ABS_DATA_DIR = os.path.join(
    os.path.dirname(filename), os.pardir, CONFIG["INPUT_DATA_DIR"]
)


# TODO config_yaml
# INPUT_DATA_DIR =
df = read_and_rename(os.path.join(ABS_DATA_DIR, "BrightonPerformanceData.csv"))

geo_columns = ["latitude", "longitude", "zipcode", "city"]
constants = ["scraped_during_month", "country_code", "currency_native"]
not_useful = ["property_type", "airbnb_host_id", "last_seen"]
cols_to_drop = geo_columns + constants + not_useful
df = df.drop(cols_to_drop, axis=1)


df.sort_values(by=["airbnb_property_id", "reporting_month"], inplace=True)

# Shift the reporting_month column by one row for each airbnb_property_id
df["next_reporting_month"] = df.groupby("airbnb_property_id")["reporting_month"].shift(
    -1
)

# Create a new column that is True if the next month's row exists for that airbnb_property_id
df["target"] = ~df["next_reporting_month"].isnull()

df = df[df["reporting_month"] != "2023-10-01"].drop(["next_reporting_month"], axis=1)
df["event_timestamp"] = pd.to_datetime(df["reporting_month"])

data_df1 = df[["airbnb_property_id", "event_timestamp"] + ["bedrooms", "bathrooms"]]
data_df2 = df[
    ["airbnb_property_id", "event_timestamp"]
    + ["blocked_days", "available_days", "occupancy_rate", "reservation_days"]
]

target_df = df[["airbnb_property_id", "target", "event_timestamp"]]


data_df1.loc[data_df1.bedrooms == "Studio", "bedrooms"] = "1"
data_df1.loc[:, "bedrooms"] = data_df1["bedrooms"].astype("int32")

data_df1.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], "data_df1.parquet"))
data_df2.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], "data_df2.parquet"))
target_df.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], "target_df.parquet"))
