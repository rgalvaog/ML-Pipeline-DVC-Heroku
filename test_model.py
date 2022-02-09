'''
test_model.py

Rafael Guerra
Feb 2022

This script conducts unit tests for three functions of model.py

'''

# Import packages
import os
import pandas as pd
import numpy as np
from model import train_model, compute_model_metrics, inference
from data import process_data
import sklearn.linear_model
from sklearn.model_selection import train_test_split

# Load test dataset
X = pd.read_csv(os.getcwd() + "/data/unit_test_sample_data.csv",index_col=False)
y = np.array([1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1])

# Test Train Model
def test_train_model():
    try:
        assert type(train_model(X,y))==sklearn.linear_model._logistic.LogisticRegression
    except TypeError as err:
        raise err

# Test Inference
def test_inference():
    try:
        model = train_model(X,y)
        inference(model, X)
        assert type(inference(model, X))==np.ndarray
    except AssertionError as err:
        raise err    

# Test Compute Model Metrics
def test_compute_model_metrics():
    try:
        predictions = np.array([1,0,1,0,1,0,1,1,1,1,1,0,1,1,0,1,1,0,0,1,0,1,0,0,1,0,1,0,1])
        precision,recall,fbeta = compute_model_metrics(y, predictions)
        assert [precision,recall,fbeta] == [1,1,1]
    except ValueError as err:
        raise err

# Run tests
if __name__ == "__main__":
    test_train_model()
    test_inference()
    test_compute_model_metrics()