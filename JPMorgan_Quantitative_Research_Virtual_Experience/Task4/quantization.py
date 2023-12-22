import pandas as pd
import numpy as np

# Print the current working directory
cwd = os.getcwd()
print("Current working directory: {0}".format(cwd))

# Load the loan data from the CSV file
# Replace '________' with the actual path to the 'loan_data_created.csv' file
os.chdir("________")
df = pd.read_csv('loan_data_created.csv')

# Extract 'default' and 'fico_score' columns
x = df['default'].to_list()
y = df['fico_score'].to_list()
n = len(x)
print("Number of data points:", n)

# Initialize lists to keep track of defaults and totals for each FICO score
default = [0 for _ in range(851)]
total = [0 for _ in range(851)]

# Count defaults and totals for each FICO score
for i in range(n):
    y[i] = int(y[i])
    default[y[i] - 300] += x[i]
    total[y[i] - 300] += 1

# Calculate cumulative sums for defaults and totals
for i in range(1, 551):
    default[i] += default[i - 1]
    total[i] += total[i - 1]

# Define a function to calculate log-likelihood
def log_likelihood(n, k):
    p = k / n
    if p == 0 or p == 1:
        return 0
    return k * np.log(p) + (n - k) * np.log(1 - p)

# Set the number of rounds (r) for the algorithm
r = 10

# Initialize a 3D array to store intermediate results
dp = [[[-10**18, 0] for _ in range(551)] for _ in range(r + 1)]

# Dynamic programming to calculate the log-likelihood
for i in range(r + 1):
    for j in range(551):
        if i == 0:
            dp[i][j][0] = 0
        else:
            for k in range(j):
                if total[j] == total[k]:
                    continue
                if i == 1:
                    dp[i][j][0] = log_likelihood(total[j], default[j])
                else:
                    temp = dp[i - 1][k][0] + log_likelihood(total[j] - total[k], default[j] - default[k])
                    if dp[i][j][0] < temp:
                        dp[i][j][0] = temp
                        dp[i][j][1] = k

# Print the final result
print("Maximum Log-Likelihood:", round(dp[r][550][0], 4))

# Reconstruct the path to find FICO scores corresponding to maximum log-likelihood
k = 550
path = []
while r >= 0:
    path.append(k + 300)
    k = dp[r][k][1]
    r -= 1

# Print the FICO scores corresponding to the maximum log-likelihood
print("Optimal FICO Scores:", path)
