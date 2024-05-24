import os
import sys
import yaml
import logging
import subprocess
import pandas as pd
import feast
from feast import FeatureStore
import mlflow
from mlflow.entities import ViewType
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage
from feast.dqm.profilers.ge_profiler import ge_profiler
from feast.dqm.errors import ValidationFailed

from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.dataset import PandasDataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("INFERENCE")


@ge_profiler
def stats_profiler(ds: PandasDataset) -> ExpectationSuite:
    # simple checks on data consistency
    ds.expect_column_values_to_be_between(
        "available_days", min_value=1, max_value=31, mostly=0.99  # allow some outliers
    )

    ds.expect_column_values_to_be_between(
        "bathrooms", min_value=1, max_value=1, mostly=0.99  # allow some outliers
    )

    ds.expect_column_values_to_be_between(
        "num_neighbours",
        min_value=0,
        max_value=100.0,
        mostly=0.99,  # allow some outliers
    )

    return ds.get_expectation_suite(discard_failed_expectations=False)


def get_best_model_id(experiment_name: str, metric: str = "mean_test_f1") -> str:
    """Retrieves the run id of the best trained model

    Args:
        experiment_name (str): name of the experiment
        metric (str, optional): criterion to choose the best model. Defaults to "mean_test_f1".

    Returns:
        str: run id of the best trained model
    """
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    df = mlflow.search_runs(
        experiment_names=[experiment_name], run_view_type=ViewType.ALL
    )
    df = df.sort_values(by=[f"metrics.{metric}"], ascending=False)
    return df.iloc[0].run_id


def get_config():
    filename = os.path.abspath(__file__)
    CONFIG_PATH = os.path.join(os.path.dirname(filename), os.pardir, "config.yaml")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = yaml.safe_load(f)
    return CONFIG


def get_feast_fs() -> FeatureStore:
    fs = feast.FeatureStore(
        repo_path=os.path.join(os.path.dirname(__file__), "feature_store/feature_repo")
    )
    return fs


def get_customer_to_predict() -> pd.DataFrame:
    CONFIG = get_config()
    df = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__),
            "feature_store/feature_repo/data/",
            CONFIG["TESTDF"],
        )
    )
    return df


# get_best_model("airbnb-bc")

if __name__ == "__main__":

    CONFIG = get_config()

    log.info("loading best model from mlflow")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    loaded_model_id = get_best_model_id("test1")
    loaded_model = mlflow.pyfunc.load_model(f"runs:/{loaded_model_id}/sk_models")

    fs = get_feast_fs()

    test = fs.get_historical_features(
        entity_df=get_customer_to_predict(),
        features=[
            "df1_feature_view:bedrooms",
            "df1_feature_view:bathrooms",
            "df2_feature_view:cleaning_fee",
            "df2_feature_view:available_days",
            "df2_feature_view:blocked_days",
            "df2_feature_view:occupancy_rate",
            "df2_feature_view:reservation_days",
            "df2_feature_view:adr_usd",
            "df2_feature_view:number_of_reservation",
            "df3_feature_view:num_neighbours",
            "df3_feature_view:dist_from_bc",
        ],
    )

    dataset = fs.create_saved_dataset(
        from_=test,
        name="my_inference_ds",
        allow_overwrite=True,
        storage=SavedDatasetFileStorage(
            path=os.path.join(
                os.path.dirname(__file__),
                "feature_store/feature_repo/data/my_inference_ds.parquet",
            )
        ),
        tags={"author": "fsxz"},
    )

    ds = fs.get_saved_dataset("my_inference_ds")
    vr = ds.as_reference(name="validation_reference_dataset", profiler=stats_profiler)

    try:
        validated_test = test.to_df(validation_reference=vr)
    except ValidationFailed as exc:
        print("VALIDATION FAILED! THERE'S SOME PROBLEM IN THE DATA!")
        bad_data = test.to_df()

    log.info("Sutting down mlflow ui")
    subprocess.call("kill $(lsof -t -i:5000)", shell=True)
