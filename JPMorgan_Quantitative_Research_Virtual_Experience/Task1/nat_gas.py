import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Load the data from the CSV file
df = pd.read_csv('Nat_Gas.csv')

# Convert the 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort the DataFrame by date
df.sort_values(by='Date', inplace=True)

# Visualize the historical gas prices
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], marker='o', linestyle='-', color='b')
plt.title('Historical Gas Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Gas Price')
plt.grid(True)
plt.show()

# Feature engineering: Extract month and year from the 'Date' column
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Create a linear regression model to predict gas prices
model = LinearRegression()

# Prepare features and target variable
X = df[['Year', 'Month']]
y = df['Price']

# Fit the model
model.fit(X, y)

# Function to estimate gas price for a given date
def estimate_price(input_date):
    year = input_date.year
    month = input_date.month

    # Use the model to predict the gas price
    price_estimate = model.predict([[year, month]])
    return price_estimate[0]

# Example: Estimate gas price for a specific date
input_date = datetime(2023, 6, 30)
estimated_price = estimate_price(input_date)
print(f"Estimated gas price on {input_date}: ${estimated_price:.2f}")
