# Importing dependencies
from google.protobuf.duration_pb2 import Duration
from feast import Entity, Feature, FeatureView, FileSource, ValueType

# Declaring an entity for the dataset
patient = Entity(
    name="airbnb_property_id", 
    value_type=ValueType.INT64, 
    description="The ID of the property")

# Declaring the source of the first set of features
f_source1 = FileSource(
    path=r"/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/data_df1.parquet",
    event_timestamp_column="event_timestamp"
)

# Defining the first set of features
df1_fv = FeatureView(
    name="df1_feature_view",
    ttl=Duration(seconds=86400 * 3),
    entities=["airbnb_property_id"],
    features=[
        Feature(name="listing_type", dtype=ValueType.STRING),
        Feature(name="bedrooms", dtype=ValueType.INT32),
        Feature(name="bathrooms", dtype=ValueType.INT32),
        Feature(name="airbnb_property_id", dtype=ValueType.STRING),
        Feature(name="cleaning_fee", dtype=ValueType.FLOAT)
        ],    
    source=f_source1
)

# Declaring the source of the second set of features
f_source2 = FileSource(
    path=r"/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/data_df2.parquet",
    event_timestamp_column="event_timestamp"
)

# Defining the second set of features
df2_fv = FeatureView(
    name="df2_feature_view",
    ttl=Duration(seconds=86400 * 3),
    entities=["airbnb_property_id"],
    features=[
        Feature(name="blocked_days", dtype=ValueType.INT32),
        Feature(name="available_days", dtype=ValueType.INT32),
        Feature(name="occupancy_rate", dtype=ValueType.FLOAT),
        Feature(name="reservation_days", dtype=ValueType.INT32),
        Feature(name="adr_usd", dtype=ValueType.FLOAT)
        ],    
    source=f_source2
)

# Declaring the source of the targets
target_source = FileSource(
    path=r"/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/target_df.parquet", 
    created_timestamp_column="event_timestamp"
)

# Defining the targets
target_fv = FeatureView(
    name="target_feature_view",
    entities=["airbnb_property_id"],
    ttl=Duration(seconds=86400 * 3),
    features=[
        Feature(name="target", dtype=ValueType.INT32)        
        ],    
    source=target_source
)