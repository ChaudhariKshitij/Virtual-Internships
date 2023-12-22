import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the loan data from the CSV file
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Define features (X) and target variable (y)
X = df[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
y = df['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to estimate the probability of default and calculate expected loss
def calculate_expected_loss(loan_properties, recovery_rate=0.1):
    # Convert loan_properties to a DataFrame with the same columns as the training data
    loan_df = pd.DataFrame(loan_properties, index=[0])

    # Use the trained model to predict the probability of default
    probability_of_default = model.predict_proba(loan_df)[0, 1]

    # Calculate expected loss: probability of default * (1 - recovery rate)
    expected_loss = probability_of_default * (1 - recovery_rate)

    return expected_loss

# Example usage:
loan_properties = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000.0,
    'total_debt_outstanding': 10000.0,
    'income': 60000.0,
    'years_employed': 5,
    'fico_score': 700
}

expected_loss = calculate_expected_loss(loan_properties)
print(f"Estimated Expected Loss: ${expected_loss:.2f}")
