import os
import sys
import warnings
import tempfile
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import pandas as pd
import mlflow

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("TRAINING")


def mlflow_trainer(
    clf: GridSearchCV,
    training_data: pd.DataFrame,
    target_col: str = "target",
    experiment_name: str = "airbnb-bc",
) -> None:
    """_summary_

    Args:
        clf (GridSearchCV[Pipeline]): model created from create_model function. Is should be ready to fit
    """
    existing_exp = mlflow.get_experiment_by_name(experiment_name)

    if not existing_exp:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = existing_exp.experiment_id

    timestamp = datetime.now().isoformat().split(".")[0].replace(":", ".")
    with mlflow.start_run(experiment_id=experiment_id, run_name=timestamp) as run:
        clf.fit(training_data, training_data[target_col])
        cv_results = clf.cv_results_
        best_index = clf.best_index_
        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][best_index])
            mlflow.log_metric(
                score_name.replace("mean", "std"),
                cv_results[score_name.replace("mean", "std")][best_index],
            )

        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        filename = "%s-%s-cv_results.csv" % ("RandomForest", timestamp)
        csv = os.path.join(tempdir, filename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(cv_results).to_csv(csv, index=False)

        mlflow.log_artifact(csv, "cv_results")
        mlflow.sklearn.log_model(clf.best_estimator_, "sk_models")


def create_model() -> GridSearchCV:
    """Creates a basic ml pipeline with column selection and cross validation

    Returns:
        GridSearchCV: object ready to fit
    """

    class PreprocessDF:
        def __init__(self):

            # ensure the order and needed columns
            self.needed_columns = [
                "bedrooms",
                "bathrooms",
                "cleaning_fee",
                "available_days",
                "blocked_days",
                "occupancy_rate",
                "reservation_days",
                "adr_usd",
                "number_of_reservation",
                "num_neighbours",
                "dist_from_bc",
            ]

        def fit(self, df, y=None):
            """This function is required for sklearn Pipeline, but in our case, the fit methos isn't doing anything"""
            return self

        def transform(self, input_df):
            df = input_df.copy()  # creating a copy to avoid changes to original dataset
            return df[self.needed_columns].astype("float32")

    # it guarantees that model and preprocessing needed are always togheter
    model = Pipeline(
        steps=[("preprocess", PreprocessDF()), ("classifier", RandomForestClassifier())]
    )

    search_params = {
        "classifier__criterion": ["gini"],
        "classifier__max_depth": [20, 30],
        "classifier__n_estimators": [10, 80],
    }
    # best model with f1, other metrics are only monitored
    cv_clf = GridSearchCV(
        model,
        search_params,
        scoring=[
            "f1",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "roc_auc",
        ],
        refit="f1",
        cv=3,
    )
    return cv_clf


if __name__ == "__main__":

    log.info("reading training data")
    training_df = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__),
            "feature_store/feature_repo/data/training_dataset.parquet",
        )
    )

    log.info("training the model")
    import subprocess

    subprocess.call("mlflow ui --host 0.0.0.0 --port 5000 --worker 1 &", shell=True)

    cv_model = create_model()

    # default url for local mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8000")

    mlflow_trainer(
        cv_model,
        training_data=training_df,
        target_col="target",
        experiment_name="test1",
    )
    log.info("training completed!")

    # subprocess.call("kill $(lsof -t -i:5000)", shell=True)
