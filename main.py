'''
main.py

Rafael Guerra
Feb 2022

Main.py creates the endpoints for FastAPI.

'''

# Heroku Code for DVC Integration


# Import libraries
import os
import pandas as pd
import pickle
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from data import process_data
from model import inference as infr

# Heroku Code for DVC Integration
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


#Instantiate FastAPI
app = FastAPI()

# Load model, encoder, and lb from models folder
model = pickle.load(open(os.getcwd() + '/model/' + 'model.pkl','rb'))
encoder = pickle.load(open(os.getcwd() + '/model/' + 'encoder.pkl','rb'))
lb = pickle.load(open(os.getcwd() + '/model/' + 'lb.pkl','rb'))

# Load categorical variables
cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country"
    ]


# Create class with type hints from Pydantic
class Input(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: Union[int,float]
    capital_loss: Union[int,float]
    hours_per_week: Union[int,float]
    native_country: str

    # Example
    # This is the first entry of clean_census.csv
    # Output should be <=50 or 0 in binary classification
    class Config:
        example = {
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

    
# GET message that gives a welcome message
@app.get("/")
async def greeting():
    return {"greeting":"Welcome to this MLOps project!"}

# POST method that does model inference
# Type Hinting must be used
@app.post("/inference/")
async def inference(data: Input):
    data = pd.DataFrame([{
                "age": data.age,
                "workclass": data.workclass,
                "fnlgt": data.fnlgt,
                "education": data.education,
                "education_num": data.education_num,
                "marital_status": data.marital_status,
                "occupation": data.occupation,
                "relationship": data.relationship,
                "race": data.race,
                "sex": data.sex,
                "capital_gain": data.capital_gain,
                "capital_loss": data.capital_loss,
                "hours_per_week": data.hours_per_week,
                "native_country": data.native_country        
    }])

    X_test,_,_,_ = process_data(data,cat_features,False,encoder,lb)
    inference = infr(model,X_test)
    return inference