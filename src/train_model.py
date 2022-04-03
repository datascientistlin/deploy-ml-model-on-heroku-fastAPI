# Script to train machine learning model.
import pandas as pd
import yaml
from yaml import CLoader as Loader
import pickle
import os
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from data import process_data
from model import train_model, compute_model_metrics, inference

# Add code to load in the data.
ROOT_DIR = os.path.abspath(os.curdir)
data = pd.read_csv(os.path.join(ROOT_DIR, 'data/census_clean.csv'))
with open (os.path.join(ROOT_DIR, 'src/params.yaml'), 'rb') as f:
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

model_path = os.path.join(ROOT_DIR, 'model/model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(rf, f)
encoder_path = os.path.join(ROOT_DIR, 'model/encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)

def sliced_inference(sliced_feature):
    with open(os.path.join(ROOT_DIR, 'output/slice_output.txt'), 'w') as f:
        f.write(f'Feature: {sliced_feature}\n')
        for slice in sorted(test[sliced_feature].unique()):
            sliced_test = test[test[sliced_feature] == slice]
            f.write(f'Slice: {slice}\n')
            X_test_sliced, y_test_sliced, _, _ = process_data(
                sliced_test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )
            pred = inference(rf, X_test_sliced)
            precision, recall, fbeta = compute_model_metrics(y_test_sliced, pred)
            f.write(f'Precision: {precision:.2f}, recall: {recall:.2f}, fbeta: {fbeta:.2f}\n')

sliced_inference('education')
