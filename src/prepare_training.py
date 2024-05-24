import os
import pandas as pd
import feast
from feast.types import Float64, Int64
from feast import Field
from feast.on_demand_feature_view import on_demand_feature_view
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage


fs = feast.FeatureStore(
    repo_path=os.path.join(os.path.dirname(__file__), "feature_store/feature_repo")
)


@on_demand_feature_view(
    sources=[fs.get_feature_view(name="df2_feature_view")],
    schema=[
        Field(name="rate_blocked_days", dtype=Float64),
        Field(name="rate_available_days", dtype=Float64),
    ],
)
def on_demand_rates(input_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["rate_blocked_days"] = input_df["blocked_days"] / input_df["available_days"]
    out["rate_available_days"] = input_df["available_days"] / input_df["blocked_days"]

    return out


fs.apply([on_demand_rates])

df = pd.read_parquet(
    os.path.join(
        os.path.dirname(__file__), "feature_store/feature_repo/data", "train_df.parquet"
    )
)


training_df = fs.get_historical_features(
    entity_df=df,
    features=[
        "df1_feature_view:bathrooms",
        "df2_feature_view:available_days",
        "df2_feature_view:blocked_days",
        "on_demand_rates:rate_blocked_days",
        "on_demand_rates:rate_available_days",
        "df3_feature_view:num_neighbours",
        "df3_feature_view:dist_from_bc",
    ],
)


dataset = fs.create_saved_dataset(
    from_=training_df,
    name="training_dataset",
    allow_overwrite=True,
    storage=SavedDatasetFileStorage(
        path=os.path.join(
            os.path.dirname(__file__),
            "feature_store/feature_repo/data",
            "training_dataset.parquet",
        )
    ),
    tags={"author": "fsxz"},
)
