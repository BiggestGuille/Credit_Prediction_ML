# Risk Score and Credit Approval Prediction Project

This project focuses on building machine learning models to predict two key targets:
- **RiskScore** (Regression problem)
- **CreditApproved** (Classification problem)

We have a dataset of **50,000 records**, and we will conduct an **Exploratory Data Analysis (EDA)**, followed by **preprocessing** steps to clean and prepare the data. Multiple machine learning models will be trained and evaluated for both regression and classification tasks to find the best-performing models.

## Project Overview

### Dataset
The dataset contains **50,000 samples** with features related to financial behavior and other relevant information. The target variables are:
- **ScoreRiesgo**: A continuous value representing the risk level (Regression problem).
- **CreditoAprobado**: A binary indicator (0/1) of whether a credit application is approved (Classification problem).

### Steps Involved
1. **Exploratory Data Analysis (EDA)**:
   - Understand the data distribution, correlations, and missing values.
   - Perform necessary visualizations and summary statistics.
2. **Preprocessing**:
   - Handle missing data, outliers, and categorical variables.
   - Feature scaling and transformation where needed.
3. **Modeling**:
   - **Regression Models** for predicting `ScoreRiesgo`, including:
     - Linear Regression (and variants)
     - Random Forest Regressor
     - Gradient Boosting Regressor (HistGradientBoosting and XGBoost)
     - MLP Regressor
   - **Classification Models** for predicting `CreditoAprobado`, including:
     - Logistic Regression (and variants)
     - Random Forest Classifier
     - Boosting Classifier
     - MLP Classifier
     - Voting Classifier
4. **Model Evaluation**:
   - Performance metrics:
     - For regression: Mean Squared Error (MSE), R².
     - For classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC.

5. **Best Models**:
   - After evaluation, the top-performing models for both tasks will be selected.


