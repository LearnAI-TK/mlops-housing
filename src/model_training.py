# Model training and MLflow tracking
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("California Housing Regression")

def train_and_log_model(model_name, model_class, params, X_train, y_train, X_test, y_test):
    try:
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(params)

            model = model_class(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)

            mlflow.log_metrics({"rmse": rmse, "r2_score": r2})

            # ---- add signature & input example to silence warnings ----
            signature = infer_signature(X_test, predictions)
            input_example = X_test.iloc[:2]

            # NOTE: sklearn flavor still uses artifact_path; warning is harmless.
            # If you're on newest MLflow, you can also use mlflow.log_model instead.
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="CaliforniaHousingRegressor",
                signature=signature,
                input_example=input_example,
            )

            mlflow.set_tag("run_status", "success")
            logger.info(f"Logged {model_name} with MLflow. Run ID: {run.info.run_id}")
            logger.info(f"View at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
            return run.info.run_id, rmse
    except Exception as e:
        mlflow.set_tag("run_status", "failed")
        logger.error(f"Failed to train/log model {model_name}: {e}", exc_info=True)
        raise e


def _get_model_version_for_run(client: MlflowClient, model_name: str, run_id: str):
    """Return the model version (string) that was created for this run_id."""
    mvs = client.search_model_versions(f"name = '{model_name}'")
    for mv in mvs:
        if mv.run_id == run_id:
            return str(mv.version)
    return None


def _promote_with_alias(client: MlflowClient, model_name: str, version: str, alias: str = "staging"):
    """
    Prefer aliases over stages (stages are deprecated).
    If your MLflow version doesn't support aliases, fall back to a tag.
    """
    try:
        # MLflow >= 2.9
        client.set_registered_model_alias(model_name, alias, version)
        logger.info(f"Set alias '{alias}' -> version {version} for model '{model_name}'.")
    except Exception as e:
        logger.warning(f"Alias API not available or failed ({e}). Falling back to tag on the model version.")
        client.set_model_version_tag(model_name, version, "alias", alias)
        logger.info(f"Tagged version {version} with alias='{alias}' for model '{model_name}'.")


def main():
    # Load preprocessed data
    X_train = pd.read_csv("data/processed/train_features.csv")
    y_train = pd.read_csv("data/processed/train_target.csv").squeeze()
    X_test = pd.read_csv("data/processed/test_features.csv")
    y_test = pd.read_csv("data/processed/test_target.csv").squeeze()

    # Train models
    lr_run_id, lr_rmse = train_and_log_model(
        "LinearRegression", LinearRegression, {}, X_train, y_train, X_test, y_test
    )

    dt_params = {"max_depth": 10, "random_state": 42}
    dt_run_id, dt_rmse = train_and_log_model(
        "DecisionTreeRegressor", DecisionTreeRegressor, dt_params, X_train, y_train, X_test, y_test
    )

    # Select best model
    best_run_id, best_model_name, best_rmse = ((lr_run_id, "LinearRegression", lr_rmse)
    if lr_rmse <= dt_rmse
       else (dt_run_id, "DecisionTreeRegressor", dt_rmse)
    )

    best_rmse = min(lr_rmse, dt_rmse)
    logger.info(f"Best model: {best_model_name} (Run ID: {best_run_id}) with RMSE: {best_rmse:.4f}")

    # Promote best model
    client = MlflowClient()
    model_name = "CaliforniaHousingRegressor"

    # Retry mechanism: give time for model version registration to sync
    version = None
    for attempt in range(5):
        version = _get_model_version_for_run(client, model_name, best_run_id)
        if version:
            break
        logger.warning("Model version not found yet. Retrying...")
        time.sleep(2)
#To track the mlflow tracking UIR
    print("Tracking URI:", mlflow.get_tracking_uri())

    if version:
        _promote_with_alias(client, model_name, version, alias="staging")
        _promote_with_alias(client, model_name, version, alias="production")  # Optional
    else:
        logger.warning("No model version found to promote.")

if __name__ == "__main__":
    main()
