import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import argparse
import os
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
import prepare_data
import train_model

def run_evaluation(ci_check=False):
    df = prepare_data.load_data()
    X = df.drop('charges', axis=1)
    y_log = np.log1p(df['charges']).copy()

    X_train, X_test, y_log_train, y_log_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    full_pipeline = train_model.build_pipeline(X_train)
    full_pipeline.fit(X_train, y_log_train)

    # Cross-validation scores on training set
    cv_scores = cross_val_score(full_pipeline, X_train, y_log_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    print(f"CV R² scores: {cv_scores}")
    print(f"CV mean R²:   {cv_mean:.4f}")

    # Predictions on test set
    y_preds = cross_val_predict(full_pipeline, X_test, y_log_test, cv=5)
    print(f"First 10 predictions: {y_preds[:10]}")
    print(f"Predictions shape: {y_preds.shape}, Test shape: {y_log_test.shape}")

    # Test R²
    test_r2 = full_pipeline.score(X_test, y_log_test)
    print(f"Test R²: {test_r2:.2f}")
    if ci_check:
        report_lines = [
            f"CV Mean R²: {cv_mean:.2f}",
            f"Test R²: {test_r2:.2f}",
        ]
        os.makedirs("reports", exist_ok=True)
        with open("reports/ci_report.txt", "w") as f:
            f.write("\n".join(report_lines))
        print("Report saved to reports/ci_report.txt")
    return y_preds, cv_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ci-check', action='store_true', help='Run metric threshold checks for CI')
    args = parser.parse_args()
    run_evaluation(ci_check=args.ci_check)