## Generate syntethic data based on actual data from Kaggle
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

import networkx as nx

import random

from sklearn.preprocessing import MinMaxScaler

from utils import *

### 1) Import data
PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop/projects/202410_DSSummit/repo/causalML_DSS24/data"

df_sales = pd.read_csv(f"{PATH_DATA}/sales_cleaned_v1.csv")
df_store_cities = pd.read_csv(f"{PATH_DATA}/store_cities.csv")
df_product_hierarchy = pd.read_csv(f"{PATH_DATA}/product_hierarchy.csv")

### 2) Preprocess
# drop last 4 columns:
list_cols_drop = ['promo_type_2', 'promo_bin_2', 'promo_discount_2', 'promo_discount_type_2']
df_sales.drop(columns=list_cols_drop, inplace=True)
df_sales.shape

df_store_cities.groupby(['storetype_id']).size()
df_store_cities.groupby(['city_id']).size().sort_values(ascending=False)

df = pd.merge(df_sales, df_store_cities, on='store_id', how='left')
# df = pd.merge(df, df_product_hierarchy, on='product_id', how='left')
df.shape

# Convert 'date' column to datetime type
df['date'] = pd.to_datetime(df['date'])


### 3) create some new helpful variables
# Create trend variabls
all_dates = pd.DataFrame({'date': pd.date_range(start=df['date'].min(), end=df['date'].max())})
all_dates['trend'] = np.arange(1, len(all_dates) + 1)
df = pd.merge(df, all_dates, on='date', how='left')

df['unit_price'] = df['revenue'] / df['sales']
# replace missing values
list_cols_select = ['store_id', 'product_id', 'date', 'unit_price']
df.sort_values(by=['store_id', 'product_id', 'date'], inplace=True)
df[list_cols_select].isna().sum()
df['unit_price'] = df.groupby(['store_id', 'product_id'])['unit_price'].transform(lambda x: x.fillna(method='ffill'))
df[list_cols_select].isna().sum()
df['unit_price'] = df.groupby(['store_id', 'product_id'])['unit_price'].transform(lambda x: x.fillna(method='bfill'))
df[list_cols_select].isna().sum()

df['stock_init'] = df.groupby(['store_id', 'product_id'])['stock'].shift(+1)
df['stock_init'].fillna(df['stock'] + df['sales'], inplace=True)

# analyze stock levels and its implications
df['is_oversales'] = df['stock_init'] - df['sales'] < 0
df[df['is_oversales']==1] 
# Conclusion: there are rows where stock at the end and at the beggining of the day 
# is 0 and there is sales

#TODO: can we create variable 'out-of-stock' where we know that all was sold out
# and sales potential could not be fully realized which means restricted sales?

# how many stores there are in a city: this potentially can mean how big the city is
# size of the city can inflience sales as it can be proxy for ppl density
# potentially it can also inflience price levels
df['n_cities'] = df.groupby('city_id')['store_id'].transform('nunique')

### 4) Select Top SKUs for data to create Data Generating Process
### 4a) First, remove price outliers
df_no_outliers = df.copy()

df_no_outliers.shape
df_no_outliers = remove_price_outliers_groupby(df_no_outliers, ['store_id', 'product_id'], 'unit_price', threshold=1.5)
df_no_outliers.shape

# check
df_tmp = df[list_cols_select].copy()
df_tmp = df_tmp.merge(df_no_outliers[list_cols_select], on=list_cols_select[:3], how='left', suffixes=('_orig', '_filtered'))

# plot some examples
if False:
    df_tmp[df_tmp['unit_price_orig'] != df_tmp['unit_price_filtered']]
    plot_sales_and_price_timeseries(df, 'S0001', 'P0042', price_col='unit_price')
    plot_sales_and_price_timeseries(df, 'S0144', 'P0704', price_col='unit_price')

### 4b) Second, remove sales outliers
df_no_outliers.shape
df_no_outliers = remove_sales_outliers_groupby(df_no_outliers, ['store_id', 'product_id'])
df_no_outliers.shape


### 4c) Finally, select Top SKUs for data to create Data Generating Process
df_summary_store_sku = summarize_and_rank(df, ['store_id', 'product_id'])
df_summary_sku = summarize_and_rank(df, ['product_id'])

# select top SKUs:
list_top_sku = df_summary_sku['product_id'].iloc[:1].tolist()
df_summary_store_sku = df_summary_store_sku[df_summary_store_sku['product_id'].isin(list_top_sku)]

# Select top 5 store_id within each product_id based on ranking
df_summary_top_sku = df_summary_store_sku.groupby('product_id').apply(lambda x: x.nlargest(5, 'ranking')).reset_index(drop=True)

### 5) plot top 15 store-product combinations:
row = 0
plot_sales_and_price_timeseries(df, df_summary_top_sku['store_id'].iloc[row], df_summary_top_sku['product_id'].iloc[row], price_col='unit_price')
print(row)
row += 1

### 6) Select only top 15 store-product combinations for DGP
df_selected = df_no_outliers[df_no_outliers['product_id'].isin(list_top_sku)]
df_selected.shape

# look at variance of key variables in selected dataframe 
# there shoul be reasonable (not too big) variance, given it's 1 SKU
# which means source of variance is mainly stores and trend and native promo & sales
df_selected[['sales', 'unit_price']].describe()

# basic data quality check
df_selected.groupby(['store_id']).size().sort_values()

if False:
    df_selected.to_csv(f"{PATH_DATA}/sales_top_sku_selected.csv", index=False)
if False:
    df_selected = pd.read_csv(f"{PATH_DATA}/sales_top_sku_selected.csv")

### 7) Create causal graph
df.columns

causal_graph = nx.DiGraph([('trend', 'sales'), #CF
                           ('trend', 'unit_price'),
                           ('unit_price', 'sales'), #T
                           ('stock_init', 'sales'),
                           ('stock', 'sales'),
                           ('n_cities', 'sales'), #CF
                           ('n_cities', 'unit_price'), # ?
                           ('store_size', 'sales'), #CF
                           ('store_size', 'unit_price'), # ? the bigger store the lower price?
                           ('storetype_id', 'sales'), #CF
                           ('storetype_id', 'unit_price'), # ?
                           ])
# potentially include city_id as categorical variable as well?

pos = nx.spring_layout(causal_graph, seed=10)
nx.draw(causal_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, arrowsize=20)
plt.show()

plot(causal_graph)

df_selected.describe()
df_selected.groupby(['store_size', 'n_cities', 'storetype_id'])[['sales', 'unit_price']].mean()

### 8) Instantiate GCM and fit the model
# Create the structural causal model object
scm = gcm.StructuralCausalModel(causal_graph)

# make sure relationships with Target node are linear? But I want only price->sales to be linear
scm.set_causal_mechanism('sales', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

# Automatically assign generative models to each node based on the given data
# auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm, df_selected, override_models=True, quality=gcm.auto.AssignmentQuality.GOOD)
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm, df_selected, override_models=False, quality=gcm.auto.AssignmentQuality.GOOD)
if False:
    print(auto_assignment_summary)

gcm.fit(scm, df_selected)
if False:
    print(gcm.evaluate_causal_model(scm, df_selected, compare_mechanism_baselines=True, evaluate_invertibility_assumptions=False))

# draw random samples
df_generated = gcm.draw_samples(scm, num_samples=1000)
# compare generated with original
df_selected[['sales', 'unit_price']].mean()
df_generated[['sales', 'unit_price']].mean()

# draw interventional samples: hard intervention: set price to min in generated data
intervention = {'unit_price': lambda unit_price: df_generated['unit_price'].min()}
df_intervention1 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['unit_price', 'sales']].mean(), df_intervention1[['unit_price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})


# draw interventional samples 2: hard intervention: set price to average in generated data
intervention = {'unit_price': lambda unit_price: df_generated['unit_price'].mean()}
df_intervention2 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['unit_price', 'sales']].mean(), df_intervention2[['unit_price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})


# draw interventional samples 3: hard intervention: set price to maximum in generated data
intervention = {'unit_price': lambda unit_price: df_generated['unit_price'].max()}
df_intervention3 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['unit_price', 'sales']].mean(), df_intervention3[['unit_price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})

# compare all interventions:
pd.concat([df_intervention1[['unit_price', 'sales']].mean(), 
           df_intervention2[['unit_price', 'sales']].mean(),
           df_intervention3[['unit_price', 'sales']].mean()], 
           axis=1).rename(columns={0: 'min', 1: 'average', 2: 'max'})
# seams reasonable linear relationsheep between price and sales

2+2 # test

# draw counterfacutal samples
if False:
    dowhy.gcm.whatif.counterfactual_samples(scm, intervention, df_generated)