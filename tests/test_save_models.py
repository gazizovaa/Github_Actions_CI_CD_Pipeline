import pytest
import os
import sys
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

sys.path.insert(0, "src")
import prepare_data
import train_model
import save_models

@pytest.fixture
def fitted_pipeline():
    df = prepare_data.load_data()
    df = prepare_data.clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(df)
    pipeline = train_model.build_pipeline(X_train, LinearRegression())
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

@pytest.fixture
def temp_model_path(tmp_path):
    return tmp_path


# write save_model() tests
def test_save_model_creates_file(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, _, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)

    save_models.save_model(pipeline, "test_model.pkl")
    assert os.path.exists("models/test_model.pkl")

def test_save_model_file_not_empty(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, _, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)
    save_models.save_model(pipeline, "test_model.pkl")
    file_size = os.path.getsize("models/test_model.pkl")
    assert file_size > 0

def test_save_model_creates_models_directory(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, _, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)
    assert not os.path.exists("models")
    save_models.save_model(pipeline, "test_model.pkl")
    assert os.path.exists("models")

def test_save_model_correct_filename(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, _, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)
    save_models.save_model(pipeline, "my_model.pkl")
    assert os.path.exists("models/my_model.pkl")
    assert not os.path.exists("models/wrong_name.pkl")

# writw load_model() testləri
def test_load_model_returns_pipeline(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, _, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)
    save_models.save_model(pipeline, "test_model.pkl")
    loaded = save_models.load_model("test_model.pkl")
    assert isinstance(loaded, Pipeline)

def test_load_model_can_predict(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, X_test, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)
    save_models.save_model(pipeline, "test_model.pkl")
    loaded = save_models.load_model("test_model.pkl")
    predictions = loaded.predict(X_test)
    assert len(predictions) == len(X_test)

def test_load_model_predictions_match(fitted_pipeline, temp_model_path, monkeypatch):
    pipeline, X_test, _ = fitted_pipeline
    monkeypatch.chdir(temp_model_path)
    original_preds = pipeline.predict(X_test)
    save_models.save_model(pipeline, "test_model.pkl")
    loaded = save_models.load_model("test_model.pkl")
    loaded_preds = loaded.predict(X_test)
    np.testing.assert_array_almost_equal(original_preds, loaded_preds)

def test_load_model_file_not_found():
    with pytest.raises(Exception):
        save_models.load_model("nonexistent_model.pkl")