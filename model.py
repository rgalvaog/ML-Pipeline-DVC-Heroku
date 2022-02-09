'''
model.py

Rafael Guerra
Feb 2022

This script defines the random forest classifier from scikit-learn.

'''

# Import Packages
#from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, precision_score, recall_score
from data import process_data
import pandas as pd
import pickle
import os

# Define categorical features (from train_model.py)
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

# Train Random Forest
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rf_model = LogisticRegression()
    rf_model.fit(X_train,y_train)
    try:
        return rf_model
    except:
        print("One of more parameters were not correctly set.")

# Compute precision and recall
def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    print("Precision:",precision)
    print("Recall: ",recall)
    print("F Beta: ",fbeta)
    return precision, recall, fbeta

# Predict X from model
def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    try:
        return predictions
    except:
        print("Double check that you have fed the model and the X data slice into the function.")

# Make sliced inferences on categorical features
def sliced_inference(cat_features,data,y,predictions):
    with open(os.getcwd() + "/data/slice_output.txt", "w") as f:
        for i in cat_features:
            for j in sorted(data[i].unique()):
                pred_df = data[data[i]==j].index
                sliced_pred = predictions[pred_df]
                y_sliced = y[pred_df]
                precision,recall,fbeta = compute_model_metrics(y_sliced,sliced_pred)
                f.write(f"Feature: {i}")
                f.write('\n')
                f.write(f"Feature Value: {j}")
                f.write('\n')
                f.write(f"Precision: {precision}")
                f.write('\n')
                f.write(f"Recall: {recall}")
                f.write('\n')
                f.write(f"FBeta: {fbeta}")

