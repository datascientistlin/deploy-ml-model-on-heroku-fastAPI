# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
import pickle
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Loading pickles
ROOT_DIR = os.path.abspath(os.curdir)
model_path = os.path.join(ROOT_DIR, "model/model.pkl")
encoder_path = os.path.join(ROOT_DIR, "model/encoder.pkl")
with open (model_path, 'rb') as f:
    model = pickle.load(f)
with open (encoder_path, 'rb') as f:
    encoder = pickle.load(f)

app = FastAPI()

@app.get("/")
async def say_hello():
    return "Hello World!"

class CensusData(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(
        ..., alias="marital-status", example="Never-married"
    )
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(
        ..., alias="native-country", example="United-States"
    )

@app.post("/predict")
async def predict_salary(request_data: CensusData):
    data = pd.DataFrame.from_dict([request_data.dict(by_alias=True)])

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

    X_categorical = data[cat_features].values
    X_categorical = encoder.transform(X_categorical)
    X_continuous = data.drop(*[cat_features], axis=1)
    X = np.concatenate([X_continuous, X_categorical], axis=1)

    prediction = model.predict(X)
    res = "<=50K" if prediction[0] == 0 else ">50K"
    return {"prediction": res}