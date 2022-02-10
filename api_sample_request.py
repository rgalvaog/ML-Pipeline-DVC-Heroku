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

example = json.dumps(prediction_below_50k)

# Load app URL
heroku_url = 'https://ml-census.herokuapp.com/inference/'

# Generate response
response = requests.post(heroku_url,example)
print("Response: ",response)
