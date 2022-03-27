import pandas as pd
import numpy as np
import pytest
import os

from data import process_data
from model import train_model, compute_model_metrics, inference

from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier

@pytest.fixture
def data():
    """ Funciontion to load cleand dataset"""
    ROOT_DIR = os.path.abspath(os.curdir)
    df = pd.read_csv(os.path.join(ROOT_DIR, 'data/census_clean.csv'))
    return df

@pytest.fixture
def cat_features():
    """ Function to return categorical feature columns"""
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
    return cat_features

def test_preprocess(data, cat_features):
    """ Test the preprocess data function to ensure proper data are returned """
    X, y, encoder, lb = process_data(
        data, categorical_features=cat_features, label="salary", training=True
    )
    assert data.shape[0] == X.shape[0]
    assert data.shape[0] == y.shape[0]

def test_train_model(data, cat_features):
    """ Test whether an trained model is returned"""
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    rf = train_model(X_train, y_train)
    assert is_classifier(rf)

def test_inference_model(data, cat_features):
    """ Test whether an inference result is returned"""
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    rf = train_model(X_train, y_train)
    pred = inference(rf, X_test)
    assert isinstance(pred, np.ndarray)