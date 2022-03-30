from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello World!"


def test_predict_1():
    request_body = {
        "age": 49,
        "workclass": "Private",
        "fnlgt": 187454,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Sales",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 99999,
        "capital-loss": 0,
        "hours-per-week": 65,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}


def test_predict_2():
    request_body = {
        "age": 60,
        "workclass": "Local-gov",
        "fnlgt": 98350,
        "education": "Some-college",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Other-service",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "Philippines"
    }
    response = client.post("/predict", json=request_body)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}