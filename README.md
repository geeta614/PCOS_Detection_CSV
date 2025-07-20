# 🧠 PCOS Detection Using Machine Learning (CSV-Based Analysis)

This repository contains a machine learning-based analysis for detecting **Polycystic Ovary Syndrome (PCOS)** using a structured CSV dataset. The study explores the effectiveness of various traditional machine learning models and evaluates the impact of techniques like **SMOTE** for class balancing, and **Select KBest** and **Recursive Feature Elimination (RFE)** for feature selection in the dataset.

---

## 📂 Dataset

- Source: [CSV file](https://www.kaggle.com/datasets/shreyasvedpathak/pcos-dataset)
- Contains a variety of health-related features such as:
  - Follicle count (left/right ovary)
  - Weight gain
  - Hair growth
  - BMI, age, insulin levels, etc.
- Target Variable: `PCOS (Y/N)`

---

## 🧪 Methodology

This study follows a step-by-step pipeline:

1. **Data Loading and Preprocessing**  
   - Handling missing values using `SimpleImputer`  
   - Standardizing features with `StandardScaler`

2. **Handling Class Imbalance**  
   - Applied **SMOTE** to achieve balanced distribution:  
     - Final class distribution: (0, 1) = (364, 364)

3. **Feature Selection**  
   - Using `SelectKBest` and `Recursive Feature Elimination (RFE)`

5. **Model Training**  
   Implemented 5 traditional classification models:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Random Forest
   - Decision Tree

6. **Evaluation Metrics**  
   - Accuracy
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)

---

## 📊 Results Summary
**Random Forest** achieved the highest accuracy of **95.89%** using the top 10 selected features.

---

## 🧰 Libraries Used

- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Data visualization
- `scikit-learn` – Modeling, evaluation, preprocessing, and feature selection
- `imblearn` – Class imbalance handling using SMOTE

---
