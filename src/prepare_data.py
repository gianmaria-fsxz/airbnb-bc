import os
import yaml
import logging
import pandas as pd

from utils.io_utils import read_and_rename
from utils.geo_processing_utils import get_dist_from_bc, get_num_neighbours

log = logging.getLogger("INSTALLATION")
filename = os.path.abspath(__file__)
CONFIG_PATH = os.path.join(os.path.dirname(filename), os.pardir, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

ABS_DATA_DIR = os.path.join(
    os.path.dirname(filename), os.pardir, CONFIG["INPUT_DATA_DIR"]
)


# TODO config_yaml
# INPUT_DATA_DIR =
df = read_and_rename(os.path.join(ABS_DATA_DIR, CONFIG["INPUT_FILE"]))

geo_columns = ["latitude", "longitude", "zipcode", "city"]
constants = ["scraped_during_month", "country_code", "currency_native"]
not_useful = ["property_type", "airbnb_host_id", "last_seen"]
cols_to_drop = constants + not_useful
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
data_df3 = df[["airbnb_property_id", "event_timestamp"] + geo_columns]
data_df3 = get_num_neighbours(data_df3, GEO_ID="airbnb_property_id")
data_df3 = get_dist_from_bc(data_df3, GEO_ID="airbnb_property_id")

target_df = df[["airbnb_property_id", "target", "event_timestamp"]]


data_df1.loc[data_df1.bedrooms == "Studio", "bedrooms"] = "1"
data_df1.loc[:, "bedrooms"] = data_df1["bedrooms"].astype("int32")

data_df1.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], CONFIG["DATA1"]))
data_df2.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], CONFIG["DATA2"]))
data_df3.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], CONFIG["DATA3"]))


# SPLIT TRAIN AND TEST_DATA
train_df, test_df = (
    target_df[target_df[CONFIG["EVENT_TIMESTAMP"]] < "2023-09-01"],
    target_df[target_df[CONFIG["EVENT_TIMESTAMP"]] >= "2023-09-01"],
)
train_df.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], "train_df.parquet"))
test_df.to_parquet(path=os.path.join(CONFIG["OUT_DATA_DIR"], "test_df.parquet"))
