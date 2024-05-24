from datetime import timedelta
import os
import yaml

from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float32, Float64, Int32, Int64, String

filename = os.path.abspath(__file__)
CONFIG_PATH = os.path.join(
    os.path.dirname(filename), os.pardir, os.pardir, os.pardir, "config.yaml"
)
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = os.path.join(os.path.dirname(filename), "data")

# Declaring an entity for the dataset
property_entity = Entity(
    name="airbnb_property_id", description="The ID of the property"
)

# Declaring the source of the first set of features
f_source1 = FileSource(
    name="source1",
    path=os.path.join(DATA_DIR, CONFIG["DATA1"]),
    timestamp_field=CONFIG["EVENT_TIMESTAMP"],
)

# Defining the first set of features
df1_fv = FeatureView(
    name="df1_feature_view",
    ttl=timedelta(days=1),
    entities=[property_entity],
    schema=[
        Field(name="listing_type", dtype=String),
        Field(name="bedrooms", dtype=Int32),
        Field(name="bathrooms", dtype=Int32),
        # Field(name="cleaning_fee", dtype=Float32)
    ],
    source=f_source1,
)

# Declaring the source of the second set of features
f_source2 = FileSource(
    name="source2",
    path=os.path.join(DATA_DIR, CONFIG["DATA2"]),
    timestamp_field=CONFIG["EVENT_TIMESTAMP"],
)


# Defining the second set of features
df2_fv = FeatureView(
    name="df2_feature_view",
    ttl=timedelta(days=1),
    entities=[property_entity],
    schema=[
        Field(name="blocked_days", dtype=Int32),
        Field(name="available_days", dtype=Int32),
        Field(name="occupancy_rate", dtype=Float32),
        Field(name="cleaning_fee", dtype=Float32),
        Field(name="number_of_reservation", dtype=Float32),
        Field(name="reservation_days", dtype=Int32),
        Field(name="adr_usd", dtype=Float32),
        Field(name="revenue_usd", dtype=Int64),
    ],
    source=f_source2,
)

# Declaring the source of the third set of features
f_source3 = FileSource(
    name="source3",
    path=os.path.join(DATA_DIR, CONFIG["DATA3"]),
    timestamp_field=CONFIG["EVENT_TIMESTAMP"],
)

# Defining the third set of features
df3_fv = FeatureView(
    name="df3_feature_view",
    ttl=timedelta(days=1),
    entities=[property_entity],
    schema=[
        Field(name="num_neighbours", dtype=Int64),
        Field(name="dist_from_bc", dtype=Float64),
        # Field(name="adr_usd", dtype=Float32)
    ],
    source=f_source3,
)


# source1_push_source = PushSource(
#     name="source1_push_source",
#     batch_source=f_source1,
# )

# source2_push_source = PushSource(
#     name="source2_push_source",
#     batch_source=f_source2,
# )

# Declaring the source of the targets
# target_source = FileSource(
#     name="target_source",
#     path=os.path.join(DATA_DIR, CONFIG["TARGETDF"]),
#     timestamp_field=CONFIG["EVENT_TIMESTAMP"],
# )

# # Defining the targets
# target_fv = FeatureView(
#     name="target_feature_view",
#     entities=[property_entity],
#     ttl=timedelta(days=1),
#     schema=[Field(name="target", dtype=Bool)],
#     source=target_source,
# )
