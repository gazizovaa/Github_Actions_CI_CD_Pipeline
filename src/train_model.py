import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import prepare_data

# Splitting data into features (X) and target (y)
X = prepare_data.df.drop('charges', axis=1)
y_log = np.log1p(prepare_data.df['charges']).copy()

# Splittig each X and y into train/test sets
X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"Shape of training set: {X_train.shape}\nShape of log train set: {y_log_train.shape}")
print(f"Shape of test set: {X_test.shape}\nShape of log test set: {y_log_test.shape}")

# Defining numerik columns
num_features = X_train.select_dtypes(include=[np.number]).columns
print(num_features)

# Defining categorica columns
cat_features = X_train.select_dtypes(exclude=[np.number]).columns
print(cat_features)

# Create a full pipeline -> I transfer the ML model into the basic pipeline I created
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

# Fitting the Linear Regression model on training set 
print(full_pipeline.fit(X_train, y_log_train))

# Calculating train/test scores
print(f"Train score: {full_pipeline.score(X_train, y_log_train)}\nTest score: {full_pipeline.score(X_test, y_log_test)}")


