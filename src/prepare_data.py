import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# Loading data
medical_costs = "src\data\medical-charges.csv"
df = pd.read_csv(medical_costs)
print(df)

# Dataset Overview
print(df.describe())
print(df.columns)
print(df.info())

# Checking and deleting duplicated values
print(df.duplicated().sum())
print(df.drop_duplicates(inplace=True))
print(df.shape)

# Adding Visualizations
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



