import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

import networkx as nx

from utils import *

### 1) Import data
PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop/projects/202410_DSSummit/repo/causalML_DSS24/data/"

df = pd.read_csv(f"{PATH_DATA}/data_clean.csv")

# select one SKU
df = df[df['PROD_LONG']=='TOTAL GIN GIN BOMBAY SAPPHIRE 1 L STANDARD']

df['period_start'] = pd.to_datetime(df['period_start'])

# drop unncecessary columns
df.columns
list_cols_drop = ['log_Sales_Units_normalized_weighted',
                  'household_mixing_indoors_banned', 
                  'pubs_closed', 
                  'shops_closed',
                  'eat_out_to_help_out',
                  'MKT_SHORT', 
                  'PROD_LONG',
                  'BRAND',
                  'WEIGHT_VOLUME',
                  'week_of_year',
                  'N_stores_weighted_dist_w',
                  'max_N_stores_weighted_dist_w', 
                  'Sales_Units_normalized_weighted',
                  'wfh',
                  'PROD_TAG',
                  'stay_at_home',
                  'Sales_Volume']
df.drop(columns=list_cols_drop, inplace=True)

# rename mkt_tag and change values
df.rename(columns={'MKT_TAG': 'store_type'}, inplace=True)
# replace each unique value in store_type with letters starting from A
unique_store_types = df['store_type'].unique()
store_type_mapping = {store_type: chr(65 + i) for i, store_type in enumerate(unique_store_types)}
df['store_type'] = df['store_type'].map(store_type_mapping)
df.groupby(['store_type']).size()

df.rename(columns={'is_holiday': 'holiday_days'}, inplace=True)
df.rename(columns={'period_start': 'week_start'}, inplace=True)

# scale sales_volume:
df['Sales_Units'] = np.round(df['Sales_Units'] / df['Weighted_Distribution_w'] * 100)
df['Sales_Value'] = np.round(df['Sales_Value'] / df['Weighted_Distribution_w'] * 100)

# recalculate sales and price
df['Sales_Units'] = np.round(df['Sales_Units'] * 0.13)
df['Sales_Value'] = np.round(df['Sales_Value'] * 0.13)
df['avg_price'] = np.round(df['avg_price'] * 1.2 + 3)
df.groupby(['avg_price']).size()

# round values in avg_price to have at least 20 rows for each price
price_counts = df['avg_price'].value_counts()

# Identify the prices with fewer than 20 rows
prices_to_update = price_counts[price_counts < 20].index

# Update the 'avg_price' values
for price in prices_to_update:
    # Find the nearest price with at least 20 rows
    nearest_price = price_counts[price_counts >= 20].index[np.abs(price_counts[price_counts >= 20].index - price).argmin()]
    df.loc[df['avg_price'] == price, 'avg_price'] = nearest_price
df.groupby(['avg_price']).size()

#rename all columns to lower case
df.columns = [col.lower() for col in df.columns]

df.drop(columns=['weighted_distribution_w'], inplace=True)
df.drop(columns=['log_price', 'log_sales_units'], inplace=True)

df['log_price'] = np.log(df['avg_price'] + 1)  # Adding 1 to avoid log(0)
df['log_sales'] = np.log(df['sales_units'] + 1)  # Adding 1 to avoid log(0)
df.rename(columns={'avg_price': 'price'}, inplace=True)
df.rename(columns={'sales_units': 'sales'}, inplace=True)

print(f"Print a few rows of df \n{df.head()}")

print("Saving df to csv:")
df.to_csv(f"{PATH_DATA}/df_prepared.csv", index=False)
print("df saved.")

if False:
    df = pd.read_csv(f"{PATH_DATA}/df_prepared.csv")
