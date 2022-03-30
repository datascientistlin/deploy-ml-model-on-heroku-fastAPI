import json
import requests

URL = "https://udacity-nd0821-c3.herokuapp.com/predict"

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

response = requests.post(URL, data=json.dumps(request_body))

dictionary = {
    "REQUEST BODY": json.dumps(request_body),
    "STATUS CODE": response.status_code,
    "PREDICTION": response.json()
}

print(json.dumps(dictionary, indent=4))