# Titanic: Machine Learning from Disaster (End-to-End Project)

This repository contains an end-to-end Machine Learning project on the classic Titanic dataset. The project is divided into two main phases: **Data Cleaning & Preprocessing** (Week 1) and **Predictive Modeling using Classification Algorithms** (Week 2).

---

## Project Overview
The goal of this project is to predict whether a passenger survived the Titanic disaster based on features like age, gender, passenger class, and fare.

---

## Phase 1: Data Cleaning & Preprocessing
In the first phase (`titanic_data_cleaning.ipynb`), the raw dataset was processed to make it ready for machine learning models.
* **Handling Missing Values:** Filled missing values in age and fare columns using median strategies and handled categorical gaps.
* **Feature Engineering:** Extracted useful insights and dropped irrelevant columns (like PassengerId, Name, Ticket, Cabin) to reduce noise.
* **Categorical Encoding:** Converted text-based features (like Gender and Embarked) into numerical values using One-Hot Encoding (`pd.get_dummies`).

---

## Phase 2: Predictive Modeling (Classification)
In the second phase (`titanic_classification_models.ipynb`), three different Supervised Machine Learning algorithms were trained and evaluated on an 80-20 train-test split.

### Model Performance & Accuracy:
| Model | Accuracy Score | Key Highlights |
| :--- | :--- | :--- |
| **Logistic Regression** | **82.12%** | Best baseline performance with standard iterations. |
| **Decision Tree (Tuned)** | **81.56%** | Optimized at `max_depth=5` to prevent overfitting. |
| **Decision Tree (Default)**| **80.45%** | Unlimited depth led to slight overfitting. |
| **K-Nearest Neighbors (KNN)**| **65.92%** | Lower accuracy, sensitive to feature scaling. |

### Key Insights & Evaluation
* **Hyperparameter Tuning:** Tuning the `max_depth` of the Decision Tree significantly improved the model's test accuracy (from 80.45% to 81.56%).
* **Confusion Matrix:** Implemented visual `ConfusionMatrixDisplay` for Logistic Regression to evaluate True Positives and False Positives effectively.

---

## Repository Structure
* `titanic_data_cleaning.ipynb`: Jupyter Notebook for Data Preprocessing.
* `titanic_classification_models.ipynb`: Jupyter Notebook for Model Training, Evaluation, and Tuning.
* `titanic_cleaned_survival.csv`: The cleaned dataset used for training models.

---

## Tech Stack Used
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
