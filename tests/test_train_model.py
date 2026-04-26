import pytest
import numpy as np
import pandas as pd
import sys
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, "src")
import prepare_data
import train_model

@pytest.fixture
def data():
    df = prepare_data.load_data()
    df = prepare_data.clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(df)
    return X_train, X_test, y_train, y_test

# write build_pipeline tests
def test_build_pipeline_returns_pipeline(data):
    X_train, _, _, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    assert isinstance(pipeline, Pipeline)

def test_build_pipeline_not_fitted(data):
    X_train, _, _, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    assert not hasattr(pipeline, "feature_names_in_")

def test_build_pipeline_with_linear_regression(data):
    X_train, _, _, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    assert "model" in pipeline.named_steps

def test_build_pipeline_with_random_forest(data):
    X_train, _, _, _ = data
    pipeline = train_model.build_pipeline(X_train, RandomForestRegressor())
    assert "model" in pipeline.named_steps

def test_build_pipeline_has_preprocessing(data):
    X_train, _, _, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    assert "preprocessing" in pipeline.named_steps
    
# write train_model() tests
def test_train_model_returns_pipeline(data):
    X_train, _, y_train, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    fitted = train_model.train_model(pipeline, X_train, y_train)
    assert isinstance(fitted, Pipeline)

def test_train_model_can_predict(data):
    X_train, X_test, y_train, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    fitted = train_model.train_model(pipeline, X_train, y_train)
    predictions = fitted.predict(X_test)
    assert len(predictions) == len(X_test)

def test_train_model_predictions_are_numeric(data):
    X_train, X_test, y_train, _ = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    fitted = train_model.train_model(pipeline, X_train, y_train)
    predictions = fitted.predict(X_test)
    assert np.issubdtype(predictions.dtype, np.floating)

def test_train_model_r2_score_reasonable(data):
    X_train, X_test, y_train, y_test = data
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    fitted = train_model.train_model(pipeline, X_train, y_train)
    score = fitted.score(X_test, y_test)
    assert score > 0, f"R² score is so lower: {score}"
    
# write fine_tune_model() tests
def test_fine_tune_returns_pipeline(data):
    X_train, _, y_train, _ = data
    tuned = train_model.fine_tune_model(X_train, y_train)
    assert isinstance(tuned, Pipeline)

def test_fine_tune_can_predict(data):
    X_train, X_test, y_train, _ = data
    tuned = train_model.fine_tune_model(X_train, y_train)
    predictions = tuned.predict(X_test)
    assert len(predictions) == len(X_test)

def test_fine_tune_r2_better_than_baseline(data):
    X_train, X_test, y_train, y_test = data

    rf = train_model.build_pipeline(X_train, RandomForestRegressor(random_state=42))
    rf = train_model.train_model(rf, X_train, y_train)
    baseline_score = rf.score(X_test, y_test)

    tuned = train_model.fine_tune_model(X_train, y_train)
    tuned_score = tuned.score(X_test, y_test)
    assert tuned_score >= baseline_score - 0.05  