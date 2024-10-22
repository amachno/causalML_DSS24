## Generate syntethic data based on actual data from Kaggle
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

import networkx as nx

from utils import *

### 1) Import data
PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop/projects/202410_DSSummit/repo/causalML_DSS24/data"

df_selected = pd.read_csv(f"{PATH_DATA}/sales_top_sku_selected.csv")
df_store_cities = pd.read_csv(f"{PATH_DATA}/store_cities.csv")


df_selected['date'] = pd.to_datetime(df_selected['date'])
list_cols_drop = ['product_id', 'storetype_id', 'city_id', 'store_size', 'trend', 'is_oversales', 'n_cities',
                  'price', 'promo_type_1', 'promo_bin_1', 'stock', 'stock_init', 'revenue']
df_selected.drop(columns=list_cols_drop, inplace=True)

# 2) aggregate data to weekly level
# 2a) prepare dates
# select all possible dates between min and max date
df_dates = pd.DataFrame({'date': pd.date_range(start=df_selected['date'].min(), end=df_selected['date'].max(), freq='D')})
# Create a new variable "week_num" that aggregates full weeks between Monday and Sunday
df_dates['week_num'] = ((df_dates['date'] - df_dates['date'].min()).dt.days // 7) + 1
# check how many weeks are not full and remove
np.sum(df_dates.groupby(['week_num']).size()!=7)
# Remove rows where the number of rows for a given week_num is less than 7
week_counts = df_dates['week_num'].value_counts()
valid_weeks = week_counts[week_counts == 7].index
df_dates = df_dates[df_dates['week_num'].isin(valid_weeks)]

df_start_dates = df_dates.groupby(['week_num'])['date'].min().reset_index()

# 2a) prepare stores
df_stores = df_selected[['store_id']].drop_duplicates().reset_index(drop=True)
df = df_dates.assign(key=1).merge(df_stores.assign(key=1), on='key').drop('key', axis=1)
df = pd.merge(df, df_selected, on=['date', 'store_id'], how='left')
# fill missing values for sales with 0 and price with 0
df['sales'].fillna(0, inplace=True)
df['price'].fillna(0, inplace=True)
df['revenue'] = df['sales'] * df['price']
df['is_sales_zero'] = (df['sales'] == 0).astype(int)
# aggregate to weekly level
df_week = df.groupby(['store_id', 'week_num']).agg({'sales': 'sum', 'revenue': 'sum', 'is_sales_zero': 'sum'}).reset_index()
df_week['price'] = np.round(df_week['revenue'] / df_week['sales'], 2)
df_week['n_days_sales'] = 7 - df_week['is_sales_zero']
df_week.drop(columns=['is_sales_zero'], inplace=True)
df_week = pd.merge(df_week, df_start_dates, on='week_num', how='left')

# 3) feature engineering
df_week['month'] = df_week['date'].dt.month
df_week = pd.merge(df_week, df_store_cities, on='store_id', how='left')
df_week['n_cities'] = df_week.groupby('city_id')['store_id'].transform('nunique')
df_week['sin_month'] = np.sin(2 * np.pi * df_week['month'] / 12)
df_week['cos_month'] = np.cos(2 * np.pi * df_week['month'] / 12)
df_week[['month', 'sin_month', 'cos_month']].drop_duplicates().round(2)

# drop rows with missing price
df_week.shape[0]
df_week.dropna(subset=['price'], how='any', inplace=True)
df_week.shape[0]

# check
if df_week[df_week['sales'] <= 0].shape[0] > 0:
    raise ValueError("There are rows with 'sales' <= 0 in df_week")

### 7) Create causal graph
df.columns

causal_graph = nx.DiGraph([('week_num', 'sales'),
                           ('week_num', 'price'),
                           ('price', 'sales'),
                           ('n_cities', 'sales'),
                           ('n_cities', 'price'),
                           ('store_size', 'sales'),
                           ('store_size', 'price'),
                           ('storetype_id', 'sales'),
                           ('storetype_id', 'price'),
                           ('sin_month', 'price'),
                           ('sin_month', 'sales'),
                           ('cos_month', 'price'),
                           ('cos_month', 'sales'),
                           ('n_days_sales', 'sales'),
                           ])
# potentially include city_id as categorical variable as well?

# pos = nx.spring_layout(causal_graph, seed=10)
# nx.draw(causal_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, arrowsize=20)
# plt.show()

plot(causal_graph)

df_week.describe()
df_week.groupby(['store_size', 'n_cities', 'storetype_id'])[['sales', 'price']].mean()

### 8) Instantiate GCM and fit the model
# Create the structural causal model object
scm = gcm.StructuralCausalModel(causal_graph)

# make sure relationships with Target node are linear? But I want only price->sales to be linear
scm.set_causal_mechanism('sales', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

# Automatically assign generative models to each node based on the given data
# auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm, df_week, override_models=True, quality=gcm.auto.AssignmentQuality.GOOD)
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm, df_week, override_models=False, quality=gcm.auto.AssignmentQuality.GOOD)
if False:
    print(auto_assignment_summary)

gcm.fit(scm, df_week)
if False: # WARNING - this takes probably more than 10 minutes!
    print(gcm.evaluate_causal_model(scm, df_week, compare_mechanism_baselines=True, evaluate_invertibility_assumptions=False))

# draw random samples
df_generated = gcm.draw_samples(scm, num_samples=1000)
# compare generated with original
df_week[['sales', 'price']].mean()
df_generated[['sales', 'price']].mean()

# draw interventional samples: hard intervention: set price to min in generated data
intervention = {'price': lambda price: df_generated['price'].min() * 0.5}
df_intervention0 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['price', 'sales']].mean(), df_intervention0[['price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})

intervention = {'price': lambda price: df_generated['price'].min()}
df_intervention1 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['price', 'sales']].mean(), df_intervention1[['price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})

# draw interventional samples 2: hard intervention: set price to average in generated data
intervention = {'price': lambda price: df_generated['price'].mean()}
df_intervention2 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['price', 'sales']].mean(), df_intervention2[['price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})

# draw interventional samples 3: hard intervention: set price to maximum in generated data
intervention = {'price': lambda price: df_generated['price'].max()}
df_intervention3 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['price', 'sales']].mean(), df_intervention3[['price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})

intervention = {'price': lambda price: df_generated['price'].max() * 1.5}
df_intervention4 = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[['price', 'sales']].mean(), df_intervention4[['price', 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})


# compare all interventions:
pd.concat([df_intervention0[['price', 'sales']].mean(),
           df_intervention1[['price', 'sales']].mean(), 
           df_intervention2[['price', 'sales']].mean(),
           df_intervention3[['price', 'sales']].mean(),
           df_intervention4[['price', 'sales']].mean()], 
           axis=1).rename(columns={0: 'min*0.5', 1: 'min', 2: 'average', 3: 'max', 4: 'max*1.5'})
# seams reasonable linear relationsheep between price and sales

# Results for non-linear:
# 	    min*0.5	    min	        average	    max	        max*1.5
# price	1.215490	2.430980	3.269062	4.040582	6.060873
# sales	54.540233	46.153052	38.537486	32.853817	13.540768

# results for linear:
#       min*0.5	    min	        average	    max	        max*1.5
# price	1.215940	2.431881	3.281490	3.998177	5.997265
# sales	55.374669	44.991868	38.854952	32.070583	15.123097

# Results for both linear and non-linear are very similar and both EXTRAPOLATE ?well?
