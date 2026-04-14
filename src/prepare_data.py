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
    print(f"Shape of data: {df.shape}")
    print(df.describe())
    print(df.info())
    # Checking duplicated values
    print(df.duplicated().sum())
    
# Data Cleaning
def clean_data(df):
    df = df.drop_duplicates()
    print(f"Shape of data after deleting duplicated values: {df.shape}")
    return df 

# Data Visualizations
def show_visualizations(df):
    # Distribution of "Charges"
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['charges'], bins=30, color='blue', kde=True)
    plt.title('Distribution of Charges')
    plt.tight_layout()
    plt.show()

    # The correlation between "BMI" and "Charges"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    plt.tight_layout()
    plt.show()
    
    # The correlation between "Age" and "Charges"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='age', y='charges', hue='smoker')
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.tight_layout()
    plt.show()

    # Detecting outliers
    num_cols = df.select_dtypes(include=np.number).columns 
    plt.figure(figsize=(10, 6))
    sns.boxplot(df[num_cols])
    plt.title("Outliers")
    plt.tight_layout()
    plt.show()
    
# Correlation Matrix
def show_correlation(df):
    plt.figure(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
def split_data(df):
    # Splitting data into features (X) and target (y)
    X = df.drop('charges', axis=1)
    y_log = np.log1p(df['charges'])

    # Splittig each X and y into train/test sets
    X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    print(f"Shape of training set: {X_train.shape}\nShape of log train set: {y_log_train.shape}")
    print(f"Shape of test set: {X_test.shape}\nShape of log test set: {y_log_test.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--split-check', action='store_true')
    args = parser.parse_args()
 
    df = load_data()
    df = clean_data(df)
    
    if args.validate_only:
        validate_data(df)
    elif args.split_check:
        validate_data(df)
        split_data(df)
    else:
        validate_data(df)
        show_visualizations(df)
        show_correlation(df) 

