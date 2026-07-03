#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =====================================================================
# MACHINE LEARNING & AI COURSE - WEEK 1 ASSIGNMENT
# Name: Pallavi Ghugardare
# Topic: Basic Data Preprocessing and Data Cleaning
# =====================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

print("=== WEEK 1 ASSIGNMENT START ===\n")

# ---------------------------------------------------------------------
# ASSIGNMENT 1: Loading Dataset and Checking Summary Statistics
# ---------------------------------------------------------------------
# Loading the Titanic dataset directly from the online URL
dataset_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(dataset_url)

print("--- 1. Basic Dataset Info ---")
df.info()

print("\n--- 2. Statistical Description of Data ---")
print(df.describe())
print("\n" + "-" * 50 + "\n")


# ---------------------------------------------------------------------
# ASSIGNMENT 2: Handling Missing Data (Data Cleaning)
# ---------------------------------------------------------------------
print("--- 3. Missing Values Count Before Cleaning ---")
print(df.isnull().sum())

# Filling missing values in Age column with the Median value
df["Age"] = df["Age"].fillna(df["Age"].median())

# Filling missing values in Embarked column with the Mode (most frequent value)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Dropping the Cabin column because it has too many missing values
df.drop(columns=["Cabin"], inplace=True)

print("\n--- 4. Missing Values Count After Cleaning ---")
print(df.isnull().sum())
print("\n" + "-" * 50 + "\n")


# ---------------------------------------------------------------------
# ASSIGNMENT 3: Categorical Encoding
# ---------------------------------------------------------------------
print("--- 5. Encoding Categorical Variables ---")

# Encoding 'Sex' column: Male/Female to 1/0 using LabelEncoder
label_enc = LabelEncoder()
df["Sex"] = label_enc.fit_transform(df["Sex"])
print("Successfully converted 'Sex' column to numbers.")

# Encoding 'Embarked' column using simple pandas get_dummies (One-Hot Encoding)
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)
print("Successfully applied One-Hot Encoding to 'Embarked' column.")
print("\n" + "-" * 50 + "\n")


# ---------------------------------------------------------------------
# MINI PROJECT 1: Visualizing Age Distribution & Exporting Cleaned Data
# ---------------------------------------------------------------------
print("--- 6. Plotting Age Distribution Graph ---")

# Creating a basic histogram to see passenger age groups
plt.figure(figsize=(7, 4))
sns.histplot(df["Age"], kde=True, color="blue", bins=25)
plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age of Passengers")
plt.ylabel("Number of Passengers")
plt.grid(axis="y", alpha=0.4)
plt.show()

# Saving the final cleaned dataframe into a new CSV file
output_file = "titanic_cleaned_survival.csv"
df.to_csv(output_file, index=False)

print("\n--- 7. Mini Project Output Info ---")
print(f"Cleaned CSV file saved as: {output_file}")
print("\nPrinting top 5 rows of final cleaned dataset:")
print(df.head())

print("\n=== ASSIGNMENT COMPLETED ===")


# In[ ]:




