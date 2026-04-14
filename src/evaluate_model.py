import argparse, os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import prepare_data
import train_model

def evaluate(pipeline, X_test, y_test, label="Model"):
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n── {label} ──")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    return {"label": label, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def plot_predictions(pipeline, X_test, y_test, label="Model"):
    y_pred = pipeline.predict(X_test)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual (log)")
    plt.ylabel("Predicted (log)")
    plt.title(f"{label}: Actual vs Predicted")
    plt.tight_layout()
    plt.show()
    
def run_evaluation(ci_check=False):
    df = prepare_data.load_data()
    df = prepare_data.clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(df)

    # Linear Model evaluate
    lr = train_model.build_pipeline(X_train, LinearRegression())
    lr = train_model.train_model(lr, X_train, y_train)
    lr_metrics = evaluate(lr, X_test, y_test, "Linear Regression")
    plot_predictions(lr, X_test, y_test, "Linear Regression")

    # RandomForest Model evaluate
    rf = train_model.build_pipeline(X_train, RandomForestRegressor(random_state=42))
    rf = train_model.train_model(rf, X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, "Random Forest")
    plot_predictions(rf, X_test, y_test, "Random Forest")

    # Fine-tuned RandomForest Model evaluate
    tuned_rf = train_model.fine_tune_model(X_train, y_train)
    tuned_metrics = evaluate(tuned_rf, X_test, y_test, "Fine-tuned Random Forest")
    plot_predictions(tuned_rf, X_test, y_test, "Fine-tuned Random Forest")

    if ci_check:
        os.makedirs("reports", exist_ok=True)
        with open("reports/ci_report.txt", "w") as f:
            for m in [lr_metrics, rf_metrics, tuned_metrics]:
                f.write(f"{m['label']}: R²={m['R2']:.2f}, MAE={m['MAE']:.2f}\n")
        print("Report saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ci-check', action='store_true')
    args = parser.parse_args()
    run_evaluation(ci_check=args.ci_check)