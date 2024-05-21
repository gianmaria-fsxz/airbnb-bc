# Importing dependencies
from feast import Entity, FeatureView, FileSource, Field, PushSource
from feast.types import Float32, Float64, Int64, Int32, String, Bool
from datetime import timedelta


# Declaring an entity for the dataset
property = Entity(
    name="airbnb_property_id", 
    description="The ID of the property")

# Declaring the source of the first set of features
f_source1 = FileSource(
    name='source1',
    path=r"/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/data_df1.parquet",
    timestamp_field="event_timestamp"
)

# Defining the first set of features
df1_fv = FeatureView(
    name="df1_feature_view",
    ttl=timedelta(days=1),
    entities=[property],
    schema=[
        Field(name="listing_type", dtype=String),
        Field(name="bedrooms", dtype=Int32),
        Field(name="bathrooms", dtype=Int32),
        # Field(name="cleaning_fee", dtype=Float32)
        ],    
    source=f_source1
)

# Declaring the source of the second set of features
f_source2 = FileSource(
    name='source2',
    path=r"/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/data_df2.parquet",
    timestamp_field="event_timestamp"
)

# Defining the second set of features
df2_fv = FeatureView(
    name="df2_feature_view",
    ttl=timedelta(days=1),
    entities=[property],
    schema=[
        Field(name="blocked_days", dtype=Int32),
        Field(name="available_days", dtype=Int32),
        Field(name="occupancy_rate", dtype=Float32),
        Field(name="reservation_days", dtype=Int32),
        # Field(name="adr_usd", dtype=Float32)
        ],    
    source=f_source2
)

source1_push_source = PushSource(
    name="source1_push_source",
    batch_source=f_source1,
)

source2_push_source = PushSource(
    name="source2_push_source",
    batch_source=f_source2,
)



# Declaring the source of the targets
target_source = FileSource(
    path=r"/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/target_df.parquet", 
    timestamp_field="event_timestamp"
)

# Defining the targets
target_fv = FeatureView(
    name="target_feature_view",
    entities=[property],
    ttl=timedelta(days=1),
    schema=[
        Field(name="target", dtype=Bool)        
        ],    
    source=target_source
)