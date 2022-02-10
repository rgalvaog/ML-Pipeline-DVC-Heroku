'''
api_sample_request.py

Rafael Guerra
Feb 2022

This script generates a sample request for the API hosted by Heroku.

'''

# Import libraries
import requests
import json

# Test prediction
# Copied from api_test.py
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

example = json.dumps(prediction_below_50K)

# Load app URL
heroku_url = 'https://ml-census.herokuapp.com/inference/'

# Generate response
response = requests.post(heroku_url,example)

print(response)
