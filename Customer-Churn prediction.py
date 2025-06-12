# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Drop unnecessary columns
dataset = dataset.drop(['customerID'], axis=1)

# Handle categorical variables
for column in dataset.columns:
    if dataset[column].dtype == object:
        dataset[column] = LabelEncoder().fit_transform(dataset[column])

# Define features and target
X = dataset.drop('Churn', axis=1)
y = dataset['Churn']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict churn for the entire dataset
dataset['Predicted_Churn'] = model.predict(X)

# List high-risk customers (predicted to churn)
high_risk_customers = dataset[dataset['Predicted_Churn'] == 1]

# Display first 5 high-risk customers
print("\nHigh-Risk Customers:\n")
print(high_risk_customers[['Predicted_Churn']].head())

# Save high-risk customers to a CSV file
high_risk_customers.to_csv('High_Risk_Customers.csv', index=False)
print("\nHigh-risk customer list saved as 'High_Risk_Customers.csv'")
