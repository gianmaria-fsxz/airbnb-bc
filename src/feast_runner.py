import feast
import pandas as pd
import numpy as np
import feast

from feast.dqm.profilers.ge_profiler import ge_profiler

from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import PandasDataset


fs = feast.FeatureStore(repo_path="/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/feature_repo")

from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int32
from feast import Field

@on_demand_feature_view(
    sources=[
      fs.get_feature_view(name='df2_feature_view')
    ],
    schema=[
        Field(name="rate_blocked_days", dtype=Float64),
        Field(name="rate_available_days", dtype=Float64),
   
    ]
)
def on_demand_stuff(input_df:pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["rate_blocked_days"] = input_df["blocked_days"] / input_df["available_days"]
    out["rate_available_days"] = input_df["available_days"] / input_df["blocked_days"]
    
    return out

fs.apply([on_demand_stuff])



from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
df = pd.read_parquet("/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/data/target_df.parquet")
# Retrieve training data from BigQuery

training_df = fs.get_historical_features(
    entity_df= df,
    features=['df1_feature_view:bathrooms',
              'df2_feature_view:available_days',
              'df2_feature_view:blocked_days',
              'on_demand_stuff:rate_blocked_days',
              'on_demand_stuff:rate_available_days'
              ], 
)

import numpy as np
import feast

from feast.dqm.profilers.ge_profiler import ge_profiler

from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import PandasDataset



dataset = fs.create_saved_dataset(
    from_=training_df,
    name='my_training_ds',
    allow_overwrite=True,
    storage=SavedDatasetFileStorage(path='/home/gianmaria/repos/airbnb-bc/src/feast/airbnb/feature_repo/my_training_ds.parquet'),
    tags={'author': 'fsxz'}
)


@ge_profiler
def stats_profiler(ds: PandasDataset) -> ExpectationSuite:
    # simple checks on data consistency
    ds.expect_column_values_to_be_between(
        "available_days",
        min_value=1,
        max_value=31,
        mostly=0.99  # allow some outliers
    )

    ds.expect_column_values_to_be_between(
        "bathrooms",
        min_value=1,
        max_value=1,
        mostly=0.99  # allow some outliers
    )

    return ds.get_expectation_suite(discard_failed_expectations=False)

# ds = fs.get_saved_dataset('my_training_ds')
# ds.get_profile(profiler=stats_profiler)

# validation_reference = ds.as_reference(name="validation_reference_dataset", profiler=stats_profiler)

job = fs.get_historical_features(
    entity_df= df,
    features=['df1_feature_view:bathrooms',
              'df2_feature_view:available_days',
              'df2_feature_view:blocked_days',
              'on_demand_stuff:rate_blocked_days',
              'on_demand_stuff:rate_available_days'
              ], 
)

from feast.saved_dataset import ValidationReference
vr = ValidationReference.from_saved_dataset('check', dataset=fs.get_saved_dataset('my_training_ds'), profiler=stats_profiler)

from feast.dqm.errors import ValidationFailed
try:
    validated = job.to_df(validation_reference=vr)
except ValidationFailed as exc:
    print("validation failed")