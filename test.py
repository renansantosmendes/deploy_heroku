import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from main_old import app

mlflow_client = TestClient(app)


def test_read_main():
    response = mlflow_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


# def test_api_route():
#     response = mlflow_client.get("/api/predict")
#     assert response.status_code == 200
#     assert response.json() == {"key": "value"}