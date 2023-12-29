port pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the data
customer_data = pd.read_csv('historical_customer_data.csv')
pricing_data = pd.read_csv('historical_pricing_data.csv')
churn_data = pd.read_csv('churn_indicator.csv')
# Merge data
merged_data = pd.merge(customer_data, pricing_data, on='customer_id')
merged_data = pd.merge(merged_data, churn_data, on='customer_id')
# Display basic information about the data
print(merged_data.info())
# Summary statistics
print(merged_data.describe())
# Distribution of churn
sns.countplot(x='churn_indicator', data=merged_data)
plt.title('Distribution of Churn')
plt.show()
# Explore variable distributions
numeric_features = ['usage', 'forecasted_usage', 'variable_price', 'fixed_price']
for feature in numeric_features:
 plt.figure(figsize=(8, 4))
 sns.histplot(merged_data[feature], bins=20)
 plt.title(f'Distribution of {feature}')
 plt.show()
# Calculate price sensitivity
merged_data['price_change_percentage'] = ((merged_data['variable_price'] -
merged_data['fixed_price']) / merged_data['fixed_price']) * 100
# Explore the relationship between price sensitivity and churn
plt.figure(figsize=(10, 6))
sns.boxplot(x='churn_indicator', y='price_change_percentage', data=merged_data)
plt.title('Price Sensitivity vs. Churn')
plt.ylabel('Price Change Percentage')
plt.xlabel('Churn Indicator')
plt.show() 