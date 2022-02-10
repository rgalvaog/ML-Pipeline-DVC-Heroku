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
    assert r.json() == {"greeting":"Welcome to this MLOps project!"}

# Post test below 50K
# prediction to return 0
def test_below_50K():

    prediction_below_50K= {
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": "<=50K"
    }

    r = client.post('/inference/',data=json.dumps(prediction_below_50K))

    assert r.status_code == 200
    assert r.json()['inference'] == 0
    # class 0 represent salary <=50K, according to LabelBinarizer output

# Post test above 50K
# prediction to return 1
def test_above_50K():

    # salary (not provided)
    prediction_above_50K = {
        "age": 40,
        "workclass": "Private",
        "fnlgt": 121956,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-spouse-absent",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital-gain": 13550,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Cambodia",
        "salary": ">50K"
    }
    r = client.post('/inference/',data=json.dumps(prediction_above_50K))

    assert r.status_code == 200
    assert r.json()['inference'] == 1