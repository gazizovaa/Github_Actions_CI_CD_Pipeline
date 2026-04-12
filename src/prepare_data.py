import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split

# Loading data
def load_data(path="src/data/medical-charges.csv"):
    df = pd.read_csv(path)
    return df 

# Dataset Overview
def validate_data(df):
    print(df.describe())
    print(df.columns)
    print(df.info())

    # Checking and deleting duplicated values
    print(df.duplicated().sum())
    print(df.drop_duplicates(inplace=True))
    print(df.shape)

# Adding Visualizations
def show_visualizations(df):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], bins=30, color='blue', kde=True)
    plt.title('Distribution of Charges')
    plt.xlabel('Charges')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.tight_layout()
    plt.show()

    # Detecting outliers
    sns.boxplot(df)
    plt.xlabel('Numeric Columns')
    plt.show()

def split_check(df):
    # Splitting data into features (X) and target (y)
    X = df.drop('charges', axis=1)
    y_log = np.log1p(df['charges']).copy()

    # Splittig each X and y into train/test sets
    X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    print(f"Shape of training set: {X_train.shape}\nShape of log train set: {y_log_train.shape}")
    print(f"Shape of test set: {X_test.shape}\nShape of log test set: {y_log_test.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true', help='Only validate data, no plots')
    parser.add_argument('--split-check', action='store_true', help='Check train/test split integrity')
    args = parser.parse_args()
 
    df = load_data()
    if args.validate_only:
        validate_data(df)
    elif args.split_check:
        validate_data(df)
        split_check(df)
    else:
        validate_data(df)
        show_visualizations(df)

