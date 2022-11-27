import pytest
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from main_old import app

mlflow_client = TestClient(app)


parser = argparse.ArgumentParser(description='Pass MLflow run id to download models')
parser.add_argument('--tracking_uri', required=True)
parser.add_argument('--tracking_username', required=True)
parser.add_argument('--tracking_password', required=True)
parser.add_argument('--registry_model_name', required=True)

args = parser.parse_args()

MLFLOW_TRACKING_URI = args.tracking_uri
MLFLOW_TRACKING_USERNAME = args.tracking_username
MLFLOW_TRACKING_PASSWORD = args.tracking_password
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD


def read_model_from_registry():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    registered_model = mlflow_client.get_registered_model(args.registry_model_name)
    version = registered_model.latest_versions[-1]
    return mlflow.pyfunc.load_model(version.source)


def test_model_instance():
    model = read_model_from_registry()
    assert isinstance(model, mlflow.pyfunc.PyFuncModel)
