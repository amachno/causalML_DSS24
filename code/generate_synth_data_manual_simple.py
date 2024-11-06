import numpy as np
import pandas as pd
import datetime
from numpy.random import default_rng

PATH_DATA = "C:/Users/artur.machno/git_repos/causalML_DSS24/data"

# Set a seed for reproducibility
rng = default_rng(seed=42)

# Define the time range
start_date = datetime.datetime(2018, 1, 1)
end_date = datetime.datetime(2023, 12, 31)
dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')  # Weekly data on Mondays

# Define store information
store_ids = np.arange(1, 11)  # 10 stores
store_types = ['urban'] * 3 + ['suburban'] * 4 + ['rural'] * 3  # Assign store types
store_info = pd.DataFrame({'store_id': store_ids, 'store_type': store_types})

# Create the base DataFrame
df = pd.DataFrame({
    'date': np.repeat(dates, len(store_ids)),
    'store_id': np.tile(store_ids, len(dates))
})

# Merge store information
df = df.merge(store_info, on='store_id')

# Extract year, month, and week number
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week_number'] = df['date'].dt.isocalendar().week
df['week_number_trend'] = (df['date'] - start_date).dt.days // 7 + 1

# Simulate 'holiday_days'
def simulate_holiday_days(month):
    if month == 12:
        return rng.poisson(2)
    elif month in [7, 8]:
        return rng.poisson(1)
    else:
        return rng.poisson(0.5)

df['holiday_days'] = df['month'].apply(simulate_holiday_days)

# Simulate 'is_lockdown' and 'curfew'
def simulate_lockdown(date):
    if datetime.datetime(2020, 3, 1) <= date <= datetime.datetime(2020, 6, 30):
        return 1
    elif datetime.datetime(2020, 10, 1) <= date <= datetime.datetime(2021, 3, 31):
        return 1
    else:
        return 0

df['is_lockdown'] = df['date'].apply(simulate_lockdown)
df['curfew'] = df.apply(
    lambda row: rng.choice([0, 1], p=[0.9, 0.1]) if row['is_lockdown'] == 0 else 1,
    axis=1
)

store_type_sales_effect = {'urban': 2, 'suburban': 1, 'rural': 0}
df['store_type_sales_effect'] = df['store_type'].map(store_type_sales_effect)

initial_base_price = 100
inflation_rate = 0.02  # Simulate natural rise of base price over time
promotion_amount = 10  # Amount to reduce price during promotions
minimum_price = 50  # Set a minimum acceptable price
maximum_price = 150  # Set a maximum acceptable price

def round_to_5(n):
    return 5 * round(n / 5)

def simulate_price(store_df):
    store_df = store_df.sort_values('date').reset_index(drop=True)
    prices = []
    # Introduce store-specific random starting price
    store_price_offset = rng.normal(0, 2)
    current_price = initial_base_price + store_price_offset

    for idx, row in store_df.iterrows():
        # Compute base price for the date (inflation over time)
        base_price = initial_base_price + inflation_rate * row['week_number_trend']
        base_price += store_price_offset  # Keep store-specific offset

        if idx > 0:
            previous_price = prices[-1]
            # Adjust probabilities to prevent rapid decrease to minimum price
            # High probability to stay the same, equal chance to increase or decrease slightly
            prob_increase = 0.15
            prob_same = 0.8
            prob_decrease = 0.05

            action = rng.choice(['increase', 'same', 'decrease'], p=[prob_increase, prob_same, prob_decrease])
            if action == 'increase':
                current_price = previous_price + rng.uniform(10, 15)  #  increase
            elif action == 'decrease':
                current_price = previous_price - rng.uniform(0, 1)  #  decrease
            else:
                current_price = previous_price

            # Ensure price does not drift too far from base price
            # Pull price towards base price
            drift = (base_price - current_price) * 0.1
            current_price += drift

            # Ensure price stays within bounds
            current_price = max(min(current_price, maximum_price), minimum_price)

        # Apply promotion during holidays randomly
        if row['holiday_days'] > 0 and rng.random() < 0.5:  # 50% chance of promotion during holidays
            price = current_price - promotion_amount
            price = max(price, minimum_price)  # Ensure price doesn't go below minimum
        else:
            price = current_price

        prices.append(price)

    store_df['price'] = prices
    store_df['price'] = store_df['price'].apply(round_to_5)
    return store_df

# Apply the simulate_price function to each store
df = df.groupby('store_id').apply(simulate_price).reset_index(drop=True)

# Simulate 'sales'
base_sales = 1000
gamma_holiday = 20
gamma_lockdown = -100
gamma_curfew = -50
gamma_month = 10
gamma_store_sales = 50
gamma_trend_sales = 0.5
gamma_price_effect = -3  # Negative effect since higher price reduces sales

# Month effect to capture seasonality
df['month_effect'] = np.sin(2 * np.pi * df['month'] / 12)

df['sales'] = (
    base_sales
    + gamma_holiday * df['holiday_days']
    + gamma_lockdown * df['is_lockdown']
    + gamma_curfew * df['curfew']
    + gamma_month * df['month_effect']
    + gamma_store_sales * df['store_type_sales_effect']
    + gamma_trend_sales * df['week_number_trend']
    + gamma_price_effect * df['price']
    + rng.normal(0, 20, size=len(df))
)

# Simulate 'competitor_sales'
delta_price_competitor = 2
delta_sales_competitor = 0.1
df['competitor_sales'] = (
    1000
    + delta_price_competitor * df['price']
    + delta_sales_competitor * df['sales']
    + rng.normal(0, 1, size=len(df))
)

# Ensure sales are non-negative
df['sales'] = df['sales'].clip(lower=0)
df['competitor_sales'] = df['competitor_sales'].clip(lower=0)

# Display the first few rows
print(df.head())

# Save the DataFrame to CSV
df.to_csv(f"{PATH_DATA}/df_synth_manual.csv", index=False)
