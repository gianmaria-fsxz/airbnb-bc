import pandas as pd
import mlflow
from mlflow.entities import ViewType


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


# get_best_model("airbnb-bc")


import mlflow
import os


mlflow.set_tracking_uri("http://127.0.0.1:5000")
loaded_model_id = get_best_model_id("test1")
loaded_model = mlflow.pyfunc.load_model(f"runs:/{loaded_model_id}/sk_models")
