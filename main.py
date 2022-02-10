'''
main.py

Rafael Guerra
Feb 2022

Main.py creates the endpoints for FastAPI.

'''

# Import libraries
import os
import pandas as pd
import pickle
from typing import Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
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

# Create Input Base Model
# Example: First row of clean_census.csv
# Expected output: <=50K
class Input(BaseModel):
    age: Union[float,int] = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: Union[float,int] = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: Union[float,int] = Field(..., alias="education-num", example=13)
    marital_status: str = Field(...,alias="marital-status",example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: Union[float,int] = Field(..., alias="capital-gain", example=2174)
    capital_loss: Union[float,int] = Field(..., alias="capital-loss", example=0)
    hours_per_week: Union[float,int] = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(...,alias="native-country",example="United-States")
    salary: Optional[str]

# Output is either 0 or 1, and therefore int or float
class Output(BaseModel):
    predict: Union[int,float]

# GET message that gives a welcome message
@app.get("/")
async def greeting():
    return {"greeting":"Welcome to this MLOps project!"}

# POST method that does model inference
# Type Hinting must be used
@app.post("/prediction/", response_model=Output, status_code=200)
async def predict(input: Input):

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # load predict_data
    request_dict = input.dict(by_alias=True)
    request_data = pd.DataFrame(request_dict, index=[0])

    # We only need X_test
    X_test, y_train, enc, load_balancing = process_data(
        request_data,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb)

    prediction = model.predict(X_test)
    return {"predict": prediction[0]}