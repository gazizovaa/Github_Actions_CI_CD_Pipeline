import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import prepare_data
import train_model

# Applying Cross-Validation method to prevent data leakage problem
scores = cross_val_score(train_model.full_pipeline, train_model.X_train, train_model.y_log_train, cv=5, scoring='r2')
print(scores)

y_preds = cross_val_predict(train_model.full_pipeline, train_model.X_test, train_model.y_log_test)
print(y_preds)

# Take the first 10 data and calculate predictions
print(y_preds[:10])

# Checking whether the shapes of predictions and log test set are compatible or not 
print(f"Shape of predictions: {y_preds.shape}\nShape of log test set: {train_model.y_log_test.shape}")
