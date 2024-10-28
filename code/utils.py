import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop\projects/202410_DSSummit/data"

def return_regex_cols(df, rgx):
    """returns list of columns"""
    comp = re.compile(rgx)
    cols = list(filter(comp.match, df.columns))
    return cols

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


def plot_sales_and_price_timeseries(df, store_id, product_id, price_col='price'):
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
    ax2.plot(df_filtered['date'].values, df_filtered[price_col].values, color='red', label=price_col)
    ax2.set_ylabel(price_col)

    # Set date format on x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

def remove_price_outliers_groupby(df, groupby_cols, price_col, threshold=1.5):
    """
    Remove price outliers from a DataFrame by grouping based on specified columns.
    This function removes outliers from the price column of a DataFrame by grouping
    the data based on the specified columns and applying the IQR (Interquartile Range) method.
    Outliers are defined as values that lie outside the range [Q1 - threshold * IQR, Q3 + threshold * IQR].
    """
    def remove_outliers(group):
        Q1 = group[price_col].quantile(0.25)
        Q3 = group[price_col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR < group[price_col].mean() * 0.1:
            return group
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return group[(group[price_col] >= lower_bound) & (group[price_col] <= upper_bound)]
    
    df_filtered = df.groupby(groupby_cols, group_keys=False).apply(remove_outliers)
    print(f"Removed {len(df) - len(df_filtered)} price outliers")
    return df_filtered


def remove_sales_outliers_groupby(df, groupby_cols, sales_col='sales', threshold=1.5):
    """
    Removes sales outliers from a DataFrame based on a specified threshold and grouping columns.
    This function groups the DataFrame by the specified columns and removes sales outliers
    within each group. Outliers are defined as sales values greater than the 95th percentile
    plus a multiple of the interquartile range (IQR), excluding rows where sales_col=0.
    """
    def remove_outliers(group):
        group_stats_calc = group[group[sales_col] > 0]
        Q95 = group_stats_calc[sales_col].quantile(0.95)
        Q1 = group_stats_calc[sales_col].quantile(0.25)
        Q3 = group_stats_calc[sales_col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q95 + threshold * IQR
        return group[group[sales_col] <= upper_bound]
    
    df_filtered = df.groupby(groupby_cols, group_keys=False).apply(remove_outliers)
    print(f"Removed {len(df) - len(df_filtered)} sales outliers")
    return df_filtered

def summarize_and_rank(df, grp_cols, get_elast_coefs=False):

    df_summary = df.groupby(grp_cols).agg(
        avg_sales=('sales', 'mean'),
        std_sales=('sales', 'std'),
        std_unit_price=('unit_price', 'std')
    ).reset_index()

    df_non_zero_sales = df[df['sales'] > 0].groupby(grp_cols).size().reset_index(name='non_zero_sales_count')
    df_summary = pd.merge(df_summary, df_non_zero_sales, on=grp_cols, how='left')

    if get_elast_coefs:
        coefficients = {}
        n_skus = len(df['product_id'].unique())
        for i, product_id in enumerate(df['product_id'].unique()):
            if i % (n_skus // 10) == 0:
                print(f"Processing {i}/{n_skus} SKUs ({(i / n_skus) * 100:.1f}%)")
            subset = df[df['product_id'] == product_id]
            if not subset.empty:
                model = LinearRegression().fit(subset[['log_price', 'month', 'year', 'store_size', 'trend']], subset['log_sales'])
                coefficients[product_id] = model.coef_[0]

        df_coefs = pd.DataFrame(list(coefficients.items()), columns=['product_id', 'coefficient']).sort_values(by='coefficient', ascending=True)
        df_coefs['coefficient'] = -df_coefs['coefficient']

        df_summary = pd.merge(df_summary, df_coefs, on=grp_cols, how='left')
        # Standardize all values in df_summary to have values between 0 and 1
        scaler = MinMaxScaler()
        df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count', 'coefficient']] = scaler.fit_transform(
            df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count', 'coefficient']]
        )
        df_summary['ranking'] = df_summary['avg_sales'] * df_summary['std_sales'] * df_summary['std_unit_price'] * df_summary['non_zero_sales_count'] * df_summary['coefficient']
        df_summary.sort_values(by=['ranking'], ascending=False, inplace=True)

        print("Returning dataframe with coefficients")
        return df_summary

    # Standardize all values in df_summary to have values between 0 and 1
    scaler = MinMaxScaler()
    df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count']] = scaler.fit_transform(
        df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count']]
    )
    df_summary['ranking'] = df_summary['avg_sales'] * df_summary['std_sales'] * df_summary['std_unit_price'] * df_summary['non_zero_sales_count']
    df_summary.sort_values(by=['ranking'], ascending=False, inplace=True)
    
    print("Returning dataframe without coefficients")
    return df_summary
