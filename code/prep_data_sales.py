import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from  utils import *

PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop\projects/202410_DSSummit/data"

df_sales = pd.read_csv(f"{PATH_DATA}/sales.csv")

# drop rows with missing values in sales, revenue, price
print(df_sales.shape[0])
df_sales.dropna(subset=['sales', 'revenue', 'price'], how='any', inplace=True)
print(df_sales.shape[0])


# remove store_id product_id combinations where min and max price are equal
price_stats = df_sales.groupby(['product_id', 'store_id'])['price'].describe(percentiles=[.25, .5, .75])[['min', '25%', '50%', '75%', 'max']]
equal_min_max = price_stats[price_stats['min'] == price_stats['max']]
combinations_to_remove = equal_min_max.index
df_sales = df_sales[~df_sales.set_index(['product_id', 'store_id']).index.isin(combinations_to_remove)]
print(df_sales.shape[0])

# filter out combinations where number of rows is lower than 100
combination_counts = df_sales.groupby(['store_id', 'product_id']).size()
combinations_to_keep = combination_counts[combination_counts >= 100].index
df_sales = df_sales[df_sales.set_index(['store_id', 'product_id']).index.isin(combinations_to_keep)]
print(df_sales.shape[0])

# Filter out combinations where number of rows with sales > 0 is lower than 50
positive_sales_counts = df_sales[df_sales['sales'] > 0].groupby(['product_id', 'store_id']).size()
combinations_to_keep_positive_sales = positive_sales_counts[positive_sales_counts >= 50].index
df_sales = df_sales[df_sales.set_index(['product_id', 'store_id']).index.isin(combinations_to_keep_positive_sales)]
print(df_sales.shape[0])

if False:
    df_sales.to_csv(f"{PATH_DATA}/sales_cleaned_v1.csv", index=False)




#### Check what promo_bin_1 means
# Identify combinations of store_id and product_id with different values of promo_bin_1 without considering missing or NaN values
# promo_bin_1_variations = df_sales.dropna(subset=['promo_bin_1']).groupby(['store_id', 'product_id'])['promo_bin_1'].nunique()
# combinations_with_different_promo_bin_1 = promo_bin_1_variations[promo_bin_1_variations > 1].index

# # Filter out combinations with different values of promo_bin_1
# df_sales_filtered = df_sales[df_sales.set_index(['store_id', 'product_id']).index.isin(combinations_with_different_promo_bin_1)]
# print(df_sales_filtered.shape[0])

