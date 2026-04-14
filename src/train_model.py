import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import joblib, os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import prepare_data

# Creating full pipeline
def build_pipeline(X_train, model):
    # Splitting training data into umeric/categoric columns
    num_features = X_train.select_dtypes(include=[np.number]).columns
    cat_features = X_train.select_dtypes(exclude=[np.number]).columns

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

    full_pipeline = Pipeline([
        ('preprocessing', transformer),
        ('model', model) 
    ])
    return full_pipeline

# Training models
def train_model(full_pipeline, X_train, y_train):
    full_pipeline.fit(X_train, y_train)
    return full_pipeline

# Fine-tune with GridSearchCV
def fine_tune_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    pipeline = build_pipeline(X_train, model)
    
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print("Best params:", grid_search.best_params_)
    print("Best CV R²:", grid_search.best_score_)
    return grid_search.best_estimator_

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()

    df = prepare_data.load_data()
    df = prepare_data.clean_data(df)
    X_train, X_test, y_train, y_test = prepare_data.split_data(df)

    if args.dry_run:
        build_pipeline(X_train, LinearRegression())
        print("Pipeline built (not fitted).")
    elif args.tune:
        fine_tune_model(X_train, y_train)
    else:
        # Linear Regression model
        lr_pipeline = build_pipeline(X_train, LinearRegression())
        lr_pipeline = train_model(lr_pipeline, X_train, y_train)
        print("LR Train R²:", lr_pipeline.score(X_train, y_train))
        print("LR Test  R²:", lr_pipeline.score(X_test, y_test))

        # Random Forest Regressor model
        rf_pipeline = build_pipeline(X_train, RandomForestRegressor(random_state=42))
        rf_pipeline = train_model(rf_pipeline, X_train, y_train)
        print("RF Train R²:", rf_pipeline.score(X_train, y_train))
        print("RF Test  R²:", rf_pipeline.score(X_test, y_test))
        
        # Saving model as .pkl for CI-CD pipeline
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(rf_pipeline, "models/model.pkl")
        print("Saved → models/model.pkl")


