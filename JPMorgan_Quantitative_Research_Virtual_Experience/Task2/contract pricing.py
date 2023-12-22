import pandas as pd

def calculate_contract_value(injection_dates, withdrawal_dates, injection_prices, withdrawal_prices,
                              injection_rate, withdrawal_rate, max_storage_volume, storage_cost):
    """
    Calculate the value of the contract based on input parameters.

    Parameters:
    - injection_dates: List of dates for gas injection.
    - withdrawal_dates: List of dates for gas withdrawal.
    - injection_prices: List of prices at which the gas is injected on corresponding injection dates.
    - withdrawal_prices: List of prices at which the gas is withdrawn on corresponding withdrawal dates.
    - injection_rate: Rate at which the gas is injected.
    - withdrawal_rate: Rate at which the gas is withdrawn.
    - max_storage_volume: Maximum volume that can be stored.
    - storage_cost: Cost associated with storing gas.

    Returns:
    - Total value of the contract.
    """
    # Create a DataFrame to store injection and withdrawal information
    data = {'Date': [], 'Action': [], 'Volume': [], 'Price': []}
    for date, price in zip(injection_dates, injection_prices):
        data['Date'].append(date)
        data['Action'].append('Injection')
        data['Volume'].append(injection_rate)
        data['Price'].append(price)

    for date, price in zip(withdrawal_dates, withdrawal_prices):
        data['Date'].append(date)
        data['Action'].append('Withdrawal')
        data['Volume'].append(withdrawal_rate)
        data['Price'].append(price)

    df = pd.DataFrame(data)

    # Sort the DataFrame by date
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)

    # Calculate the cumulative volume and value
    df['Cumulative_Volume'] = df.groupby('Action')['Volume'].cumsum()
    df['Value'] = df['Volume'] * df['Price']

    # Calculate the total value of the contract
    total_value = df['Value'].sum()

    # Check if the storage volume constraint is violated
    if df['Cumulative_Volume'].max() > max_storage_volume:
        print("Warning: Maximum storage volume constraint violated!")

    # Add storage cost to the total value
    total_value -= storage_cost * max(0, df['Cumulative_Volume'].max() - max_storage_volume)

    return total_value

# Example usage:
injection_dates = ['2023-01-15', '2023-04-15']
withdrawal_dates = ['2023-06-15', '2023-09-15']
injection_prices = [3.0, 3.5]
withdrawal_prices = [4.0, 4.5]
injection_rate = 1000  # assumed units
withdrawal_rate = 800  # assumed units
max_storage_volume = 5000  # assumed units
storage_cost = 0.02  # assumed cost per unit volume

contract_value = calculate_contract_value(injection_dates, withdrawal_dates, injection_prices, withdrawal_prices,
                                          injection_rate, withdrawal_rate, max_storage_volume, storage_cost)

print(f"Estimated contract value: ${contract_value:.2f}")
