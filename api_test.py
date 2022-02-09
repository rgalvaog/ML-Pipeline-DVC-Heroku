'''
api_test.py

Rafael Guerra
Feb 2022

This script conducts three unit tests on the API set up in main.py

'''

# Import libraries
import os
import json
from main import app
from fastapi.testclient import TestClient

# load TestClient
client = TestClient(app)

# Test GET method
def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to this MLOps project!"

# Case where salary <=50K
prediction_below_50k = {
            "individual_id": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

def test_prediction_below_50K():
    r = client.post("/inference/", json=prediction_below_50k)
    assert r.status_code == 200
    assert r.json() == "<=50K"

# Case where salary >50K
prediction_above_50k = {
            "individual_id": {
                "age": 43,
                "workclass": "Private",
                "fnlgt": 237993,
                "education": "Some-college",
                "education_num": 10,
                "marital_status": "Married-civ-spouse",
                "occupation": "Tech-support",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

def test_prediction_above_50K():
    r = client.post("/inference/", json=prediction_above_50k)
    assert r.status_code == 200
    assert r.json() == ">50K"