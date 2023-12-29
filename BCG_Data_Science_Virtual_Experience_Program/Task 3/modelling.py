from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Load the cleaned and engineered dataset
data = pd.read_csv('cleaned_and_engineered_data.csv')
# Define features and target
X = data.drop('churn_indicator', axis=1)
y = data['churn_indicator']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
# Train the classifier
rf_classifier.fit(X_train, y_train)
# Predict on the test set
y_pred = rf_classifier.predict(X_test)