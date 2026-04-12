import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import argparse
import joblib 
import os 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import prepare_data

# Create a full pipeline
def build_pipeline(X_train):
    # Defining numerik columns
    num_features = X_train.select_dtypes(include=[np.number]).columns
    print(num_features)

    # Defining categorica columns
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns
    print(cat_features)

    num_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    transformer = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ], remainder='passthrough')

    # estimator = LinearRegression()
    full_pipeline = Pipeline([
        ('preprocessing', transformer),
        ('model', LinearRegression())
    ])
    return full_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Build pipeline without fitting')
    args = parser.parse_args()
 
    df = prepare_data.load_data()
    X = df.drop('charges', axis=1)
    y_log = np.log1p(df['charges']).copy()
 
    X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
 
    # Fitting the Linear Regression model on training set 
    full_pipeline = build_pipeline(X_train)
 
    if args.dry_run:
        print("Dry-run: Pipeline built successfully (not fitted).")
    else:
        full_pipeline.fit(X_train, y_log_train)
        # Calculating train/test scores
        print(f"Train score: {full_pipeline.score(X_train, y_log_train)}\nTest score: {full_pipeline.score(X_test, y_log_test)}")
        
        # Saving ml model
        os.makedirs("models", exist_ok=True)
        joblib.dump(full_pipeline, "models/model.pkl")


