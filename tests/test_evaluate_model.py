import pytest
import numpy as np
import pandas as pd
import sys
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, "src")
import prepare_data
import train_model
import evaluate_model

@pytest.fixture
def fitted_lr():
    df = prepare_data.load_data()
    df = prepare_data.clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(df)
    pipeline = train_model.build_pipeline(X_train, 
                    __import__('sklearn.linear_model', 
                    fromlist=['LinearRegression']).LinearRegression())
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test

# write evaluate() tests
def test_evaluate_returns_dict(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    assert isinstance(result, dict)

def test_evaluate_has_required_keys(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    required_keys = ["label", "MAE", "MSE", "RMSE", "R2"]
    for key in required_keys:
        assert key in result, f"'{key}' açarı tapılmadı"

def test_evaluate_mae_is_positive(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    assert result["MAE"] >= 0

def test_evaluate_mse_is_positive(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    assert result["MSE"] >= 0

def test_evaluate_rmse_equals_sqrt_mse(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    assert abs(result["RMSE"] - np.sqrt(result["MSE"])) < 1e-6

def test_evaluate_r2_between_minus1_and_1(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    assert -1 <= result["R2"] <= 1

def test_evaluate_default_label(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    assert result["label"] == "Model"

def test_evaluate_custom_label(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test, 
                                     label="Linear Regression")
    assert result["label"] == "Linear Regression"

def test_evaluate_values_are_numeric(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    result = evaluate_model.evaluate(pipeline, X_test, y_test)
    for key in ["MAE", "MSE", "RMSE", "R2"]:
        assert isinstance(result[key], float), \
            f"'{key}' value is not found: {type(result[key])}"

# write plot_predictions() tests
def test_plot_predictions_runs_without_error(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    try:
        evaluate_model.plot_predictions(pipeline, X_test, y_test,
                                        label="Linear Regression")
    except Exception as e:
        pytest.fail(f"plot_predictions() gave an error: {e}")

def test_plot_predictions_with_default_label(fitted_lr):
    pipeline, X_test, y_test = fitted_lr
    try:
        evaluate_model.plot_predictions(pipeline, X_test, y_test)
    except Exception as e:
        pytest.fail(f"plot_predictions() gave an error: {e}")