import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop\projects/202410_DSSummit/data"

def plot_sales_timeseries(df, store_id, product_id):
    # Filter the dataframe for the given store_id and product_id
    df_filtered = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)]
    
    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print(f"No data available for store_id {store_id} and product_id {product_id}")
        return
    
    # Plot the timeseries
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered['date'].values, df_filtered['sales'].values, marker='o', linestyle='-')
    plt.title(f'Sales Time Series for Store {store_id} and Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

#### check if prices vary for the same SKU vary at the same time (across different stores)
def plot_min_max_price_over_time(df, product_id, price_col='price'):
    # Filter the dataframe for the given product_id
    df_filtered = df[df['product_id'] == product_id]
    
    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print(f"No data available for product_id {product_id}")
        return

    # Calculate minimum and maximum price for every date   
    min_max_price_per_date = df_filtered.groupby('date')[price_col].agg(['min', 'max']).reset_index().sort_values('date')

    plt.scatter(min_max_price_per_date['date'].values, min_max_price_per_date['min'].values, color='green', label='Min Price', alpha=0.1, s=5)
    plt.scatter(min_max_price_per_date['date'].values, min_max_price_per_date['max'].values, color='red', label='Max Price', alpha=0.1, s=5)
    
    plt.title(f'Min and Max Price Over Time for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calc_prc_rows_by_combination(df, list_cols):
    combination_counts = df.groupby(list_cols, dropna=False).size()
    total_rows = len(df)
    percentage_combination = (combination_counts / total_rows) * 100
    percentage_combination = percentage_combination.reset_index(name='percentage')
    return percentage_combination


def plot_sales_and_price_timeseries(df, store_id, product_id):
    # Filter the dataframe for the given store_id and product_id
    df_filtered = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)]
    
    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print(f"No data available for store_id {store_id} and product_id {product_id}")
        return
    
    # Plot the timeseries
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.scatter(df_filtered['date'].values, df_filtered['sales'].values, marker='o', linestyle='-', label='Sales')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales')
    ax1.set_title(f'Sales and Price Time Series for Store {store_id} and Product {product_id}')

    # Create a secondary y-axis for the price
    ax2 = ax1.twinx()
    ax2.plot(df_filtered['date'].values, df_filtered['price'].values, color='red', label='Price')
    ax2.set_ylabel('Price')

    # Set date format on x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

