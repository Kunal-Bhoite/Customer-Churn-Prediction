Customer Churn Prediction using Random Forest & Support Vector Machine

This project focuses on predicting customer churn using two powerful classification models — Random Forest Classifier and Support Vector Machine (SVM).
The goal is to preprocess the data, handle imbalance using SMOTE, perform hyperparameter tuning, and compare model performance based on precision, which is the chosen evaluation metric.

📁 Project Structure
CustomerChurn.csv               - Dataset used for training & evaluation
Code_Kunal_Bhoite.py            - Full preprocessing, tuning, and model training code
Report_Kunal_Bhoite.pdf         - Detailed project documentation/report

📊 Dataset Overview

The dataset contains customer attributes related to demographics, financial behavior, credit card usage, and churn status.

Key features include:

Customer_Age

Gender

Dependent_count

Education_Level

Marital_Status

Income_Category

Card_Category

Months_on_book

Total_Relationship_Count

Months_Inactive

Contacts_Count

Credit_Limit

Total_Revolving_Bal

Total_Trans_Amt

Total_Trans_Ct

Attrition_Flag (Target Variable)

🛠️ Data Preparation Steps

The following steps were performed during preprocessing:

Imported necessary libraries (pandas, sklearn, imblearn)

Loaded dataset and inspected missing values

Applied Label Encoding to binary or ordinal categorical features

Applied One-Hot Encoding to non-rankable features like Marital_Status

Scaled the numerical data using StandardScaler

Split dataset into 70% Training and 30% Testing

Applied SMOTE to handle class imbalance

Prepared final X (features) and Y (target)

🚀 Machine Learning Models Implemented
1. Random Forest Classifier

Implemented through an imblearn Pipeline (SMOTE + Classifier)

Hyperparameter tuned using GridSearchCV

Parameters tuned:

Number of trees (n_estimators)

Criterion = "entropy"

max_features = "sqrt"

Evaluated using Precision

2. Support Vector Machine (SVM)

SMOTE + SVM implemented using Pipeline

Hyperparameter tuned using GridSearchCV

Parameters tuned:

kernel = [linear, poly, rbf, sigmoid]

C = [.001, .01, .1, 1, 10, 100]

Evaluated using Precision

📈 Model Performance Summary
Model	Best Precision Score	Notes
Random Forest Classifier	~94.31%	Precision-based evaluation after hyperparameter tuning
Support Vector Machine	~96.56%	Best performer; RBF kernel selected by GridSearchCV
✔ Recommendation

Based on the results, Support Vector Machine (SVM) is recommended for real-world deployment because it achieved the highest precision score.

📉 Underfitting / Overfitting Analysis

No model showed underfitting.

Both Random Forest and SVM demonstrated good generalization.

SMOTE helped prevent bias caused by class imbalance.

Proper normalization and encoding avoided noisy model behavior.

🧪 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git


Install necessary dependencies:

pip install pandas numpy scikit-learn imbalanced-learn


Run the Python file:

python Code_Kunal_Bhoite.py

📄 Evaluation Metrics Used

Precision (Primary metric — to measure how accurately we predict “existing customer” or “attrited customer”)

Accuracy

Recall

Confusion Matrix

Precision chosen because correctly identifying churned customers is more important than overall accuracy.
