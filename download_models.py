import os
import mlflow
import argparse
from mlflow.tracking import MlflowClient


parser = argparse.ArgumentParser(description='Pass MLflow run id to download models')
parser.add_argument('--run_id', required=True)
parser.add_argument('--tracking_uri', required=True)
parser.add_argument('--tracking_username', required=True)
parser.add_argument('--tracking_password', required=True)

args = parser.parse_args()

MLFLOW_TRACKING_URI = args.tracking_uri
MLFLOW_TRACKING_USERNAME = args.tracking_username
MLFLOW_TRACKING_PASSWORD = args.tracking_password
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_local_path = mlflow_client.download_artifacts(args.run_id,
                                             'model',
                                                    os.getcwd())
scaler_local_path = mlflow_client.download_artifacts(args.run_id,
                                              'scaler.pkl',
                                                     os.getcwd())
print(model_local_path, scaler_local_path)
