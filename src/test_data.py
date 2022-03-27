import pandas as pd
import pytest

@pytest.fixture
def data():
    """ Funciont to load cleand dataset"""
    df = pd.read_csv('../data/census_clean.csv')
    return df

def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."

def test_target_var(data):
    """ Test if the target variable only consists of legitimate values"""
    assert data['salary'].isin(['<=50K', '>50K']).all()

def test_cat_feature_col(data):
    """" Test if all categorical feature columns are present"""
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
    found_col = [col for col in data.columns if col in cat_features]
    assert len(cat_features) == len(found_col)

def test_num_feature_col(data):
    """" Test if all numerical feature columns are of valid range"""
    num_features = [
        "age",
        "fnlgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]
    assert data[num_features].ge(0).all().all()
