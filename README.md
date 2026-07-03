# Titanic Dataset - Data Cleaning & Preprocessing

This repository contains a Python-based Data Preprocessing and Data Cleaning project using the famous **Titanic Dataset** from Kaggle. Data preprocessing is a crucial first step before building any Machine Learning model.

## 📌 Project Overview
The main goal of this project is to handle missing values, clean the dataset, and encode categorical features into a format suitable for Machine Learning algorithms.

## 🛠️ Key Tasks Achieved
- **Exploratory Data Analysis (EDA):** Analyzed data shapes, types, and descriptive statistics.
- **Handling Missing Values:** Imputed missing values in the `Age` column using the Median and the `Embarked` column using the Mode.
- **Feature Selection:** Dropped the `Cabin` column due to an excessive amount of missing data (over 70%).
- **Categorical Encoding:** Converted `Sex` column to binary integers using `LabelEncoder` and applied One-Hot Encoding (`pd.get_dummies`) to the `Embarked` column.
- **Data Visualization:** Plotted an Age Distribution graph using Matplotlib and Seaborn to analyze passenger demographics.

## 🚀 Technologies Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

## 📂 Output
The project successfully cleans the raw data and exports it as a new file named `titanic_cleaned_survival.csv`, which is completely ready for training Machine Learning models.
