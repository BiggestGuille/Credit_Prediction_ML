# Risk Score and Credit Approval Prediction Project 💰

This project focuses on building machine learning models to predict two key targets:
- **RiskScore** (Regression problem)
- **CreditApproved** (Classification problem)

With a dataset of **50,000 records**, we will conduct an **Exploratory Data Analysis (EDA)**, followed by **preprocessing** steps to clean and prepare the data. Then, multiple machine learning models will be trained and evaluated for both regression and classification tasks to find the best-performing models.

## Project Overview 📊

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
   - Feature scaling or transformations.
3. **Modeling**:
   - **Regression Models** for predicting `ScoreRiesgo`, including:
     - Linear Regression (and variants)
     - Random Forest Regressor
     - Gradient Boosting Regressor (HistGradientBoosting and XGBoost)
     - MLP Regressor
     - Voting Regressor
     - Bagging Regressor
   - **Classification Models** for predicting `CreditoAprobado`, including:
     - Logistic Regression (and variants)
     - Random Forest Classifier
     - Boosting Classifier (HistGradientBoosting and XGBoost)
     - Bagging Classifier
     - MLP Classifier
     - Voting Classifier
4. **Model Evaluation**:
   - Performance metrics:
     - For regression: Mean Squared Error (MSE), R².
     - For classification: Accuracy , ROC-AUC.
   - Techniques used:
     - Cross Validation with Randomized Search
     - 80-20 Train-Test Split 

5. **Competition: Best Models**
   - After evaluation, the top-performing models for both tasks were selected to participate in a competition. In this competition, teams tested their machine learning solutions against one another to achieve the best performance in a new test dataset. Our group excelled, demonstrating exceptional understanding and application of machine learning techniques, and proudly secured the first place in the challenge. As a result of our performance, we were awarded the **Highest Honors** in the Machine Learning course.
     
![WhatsApp Image 2024-11-23 at 13 04 29](https://github.com/user-attachments/assets/98a7be05-5c9b-4711-907b-41545d19a72b)


