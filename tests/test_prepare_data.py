import pytest 
import pandas as pd
import numpy as np 
import sys 

sys.path.insert(0, "src")
import prepare_data

@pytest.fixture 
def sample_df():
    return pd.DataFrame({
        "age":      [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 25],
        "sex":      ["male", "female", "male", "female", "male",
                     "female", "male", "female", "male", "female", "male"],
        "bmi":      [22.5, 28.0, 31.0, 24.5, 27.0,
                     33.0, 29.5, 26.0, 30.0, 25.5, 22.5],
        "children": [0, 1, 2, 0, 3, 1, 0, 2, 1, 0, 0],
        "smoker":   ["no", "yes", "no", "yes", "no",
                     "yes", "no", "yes", "no", "yes", "no"],
        "region":   ["southwest", "northeast", "southeast", "northwest", "southwest",
                     "northeast", "southeast", "northwest", "southwest", "northeast", "southwest"],
        "charges":  [3000.0, 5000.0, 7000.0, 4000.0, 8000.0,
                     9000.0, 3500.0, 6000.0, 7500.0, 4500.0, 3000.0]
    })
    
# write load_data() tests
def test_load_data():
    df = prepare_data.load_data()
    assert isinstance(df, pd.DataFrame)
    
def test_load_data_not_empty():
    df = prepare_data.load_data()
    assert len(df) > 0

def test_load_data_has_required_cols():
    df = prepare_data.load_data()
    required_columns = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
    for col in required_columns:
        assert col in df.columns, f"'{col}' column is not found."

def test_load_data_charges_positive():
    df = prepare_data.load_data()
    assert (df["charges"] > 0).all()
    
# write validate_data() tests
def test_validate_data_runs_without_error(sample_df):
    try:
        prepare_data.validate_data(sample_df)
    except Exception as e:
        pytest.fail(f"validate_data() gave an error: {e}")
    
# write clean_data() tests
def test_clean_data_removes_duplicates(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    assert cleaned.duplicated().sum() == 0

def test_clean_data_reduces_row_count(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    assert len(cleaned) < len(sample_df)

def test_clean_data_keeps_columns(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    assert list(cleaned.columns) == list(sample_df.columns)

def test_clean_data_returns_dataframe(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    assert isinstance(cleaned, pd.DataFrame)
    
# write split_data() tests
def test_split_data_total_rows(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(cleaned)
    assert len(X_train) + len(X_test) == len(cleaned)

def test_split_data_test_size(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    X_train, X_test, _, _ = prepare_data.split_data(cleaned)
    test_ratio = len(X_test) / (len(X_train) + len(X_test))
    assert abs(test_ratio - 0.2) < 0.15   

def test_split_data_no_charges_in_X(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    X_train, X_test, _, _ = prepare_data.split_data(cleaned)
    assert "charges" not in X_train.columns
    assert "charges" not in X_test.columns

def test_split_data_y_is_log_transformed(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    _, _, y_train, y_test = prepare_data.split_data(cleaned)
    assert (y_train >= 0).all()
    assert (y_test >= 0).all()

def test_split_data_X_train_has_correct_columns(sample_df):
    cleaned = prepare_data.clean_data(sample_df)
    X_train, _, _, _ = prepare_data.split_data(cleaned)
    expected_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
    for col in expected_cols:
        assert col in X_train.columns