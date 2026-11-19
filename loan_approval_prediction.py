
# LOAN APPROVAL PREDICTION USING MACHINE LEARNING

# Step 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Step 2: Create a sample loan dataset
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    'income': np.random.randint(15000, 150000, n),
    'credit_score': np.random.randint(300, 851, n),
    'loan_amount': np.random.randint(10000, 500000, n),
    'employment_years': np.random.randint(0, 31, n),
    'existing_loans': np.random.randint(0, 6, n),
    'debt_to_income': np.random.uniform(0.05, 0.8, n),
    'education': np.random.choice(
        ['Graduate', 'Not Graduate'], n
    )
})

# Step 3: Generate target variable
# 1 = Approved, 0 = Rejected

score = (
    (data['credit_score'] >= 650) * 2 +
    (data['income'] >= 40000) * 2 +
    (data['loan_amount'] <= 200000) * 1 +
    (data['employment_years'] >= 2) * 1 +
    (data['existing_loans'] <= 2) * 1 +
    (data['debt_to_income'] <= 0.4) * 2 +
    (data['education'] == 'Graduate') * 1
)

# Add some randomness
noise = np.random.normal(0, 1.5, n)

data['loan_status'] = (
    (score + noise >= 6).astype(int)
)

# Step 4: Separate input and output
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Step 5: Convert categorical data into numerical data
X = pd.get_dummies(X, columns=['education'], drop_first=True)

# Step 6: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 7: Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Step 8: Train the model
model.fit(X_train, y_train)

# Step 9: Make predictions
y_pred = model.predict(X_test)

# Step 10: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Loan Approval Prediction Model")
print("--------------------------------")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Step 11: Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 12: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Rejected', 'Approved'],
    yticklabels=['Rejected', 'Approved']
)

plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")
plt.title("Loan Approval Confusion Matrix")
plt.show()

# Step 13: Predict a new loan application

new_applicant = pd.DataFrame([{
    'income': 75000,
    'credit_score': 750,
    'loan_amount': 150000,
    'employment_years': 5,
    'existing_loans': 1,
    'debt_to_income': 0.25,
    'education_Graduate': 1
}])

prediction = model.predict(new_applicant)

print("\nNew Loan Application Result:")

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
