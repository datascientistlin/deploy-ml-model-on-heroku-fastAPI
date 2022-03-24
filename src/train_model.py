# Script to train machine learning model.
import pandas as pd
import yaml
from yaml import CLoader as Loader
import pickle
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, compute_model_metrics, inference

# Add code to load in the data.
data = pd.read_csv('../data/census_clean.csv')
with open ('./params.yaml', 'rb') as f:
    params = yaml.load(f, Loader=Loader)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
rf = train_model(X_train, y_train, params['n_estimators'])
pred = inference(rf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, pred)

print(precision)
print(recall)
print(fbeta)

model_path = '../model/model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)

