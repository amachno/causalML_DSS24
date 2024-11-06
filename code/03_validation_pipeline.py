## Generate syntethic data based on actual data from Kaggle
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from econml.dml import LinearDML

import networkx as nx

from utils import *



### 1) Import data
PATH_DATA = "C:/Users/artur.machno/git_repos/causalML_DSS24/data"

df = pd.read_csv(f"{PATH_DATA}/df_prepared.csv")

df_rescaling = df[['price']].drop_duplicates().copy()
df_rescaling['price_rescaled'] = 1.5 - (df_rescaling['price'] - df_rescaling['price'].min()) / (df_rescaling['price'].max() - df_rescaling['price'].min()) * 1.5
# df_rescaling['price_rescaled'] = 1 + (df_rescaling['price'] - df_rescaling['price'].min()) / (df_rescaling['price'].max() - df_rescaling['price'].min())
df_rescaling['mult_factor'] = np.exp(df_rescaling['price_rescaled'])
df_rescaling.sort_values('price')


# rescale price:
df = pd.merge(df, df_rescaling[['price', 'mult_factor']], on='price', how='left')
df['sales'] = df['sales'] * df['mult_factor']
# recreate log_sales
df.drop(columns=['mult_factor', 'log_sales'], inplace=True)
df['log_sales'] = np.log(df['sales'] + 1)  # Adding 1 to avoid log(0)

### 2) Create causal graph
df.columns

causal_graph = nx.DiGraph([('holiday_days', 'sales'),
                           ('holiday_days', 'price'),
                           ('is_lockdown', 'sales'),
                           ('curfew', 'sales'),
                           ('year_2021', 'sales'),
                           ('year_2022', 'sales'),
                           ('year_2023', 'sales'),
                           ('month_2', 'sales'),
                           ('month_3', 'sales'),
                           ('month_4', 'sales'),
                           ('month_5', 'sales'),
                           ('month_6', 'sales'),
                           ('month_7', 'sales'),
                           ('month_8', 'sales'),
                           ('month_9', 'sales'),
                           ('month_10', 'sales'),
                           ('month_11', 'sales'),
                           ('month_12', 'sales'),
                           ('store_type', 'price'),
                           ('store_type', 'sales'),
                           ('week_number_trend', 'price'),
                           ('week_number_trend', 'sales'),
                           ('price', 'sales')
                           ])

graph_simple_for_plot = nx.DiGraph([('holiday_days', 'sales'),
                           ('holiday_days', 'price'),
                           ('is_lockdown', 'sales'),
                           ('curfew', 'sales'),
                           ('year', 'sales'),
                           ('month', 'sales'),
                           ('store_type', 'price'),
                           ('store_type', 'sales'),
                           ('week_number_trend', 'price'),
                           ('week_number_trend', 'sales'),
                           ('price', 'sales')
                           ])

# potentially include city_id as categorical variable as well?

plot(graph_simple_for_plot)

month_columns = [col for col in df.columns if col.startswith('month_')]
year_columns = [col for col in df.columns if col.startswith('year_')]
print(month_columns)

### 3) Quickly run linear regression to check price elasticity coefficient
LinearRegression().fit(df[['price']], df['sales']).coef_
LinearRegression().fit(df[['log_price']], df['log_sales']).coef_    
# seems very low price elasticity coefficient
LinearRegression().fit(df[['log_price', 'week_number_trend'] + month_columns + year_columns], df['log_sales']).coef_    
# still quite low

### 4) Instantiate GCM and fit the model
# Create the structural causal model object
scm_data_generator = gcm.StructuralCausalModel(causal_graph)

# make sure relationships with Target node are linear? But I want only price->sales to be linear
scm_data_generator.set_causal_mechanism('sales', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
# scm_data_generator.set_causal_mechanism('price', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

# Automatically assign generative models to each node based on the given data
df.groupby('price').size()
plt.show()
df.groupby('price').size().plot()
plt.show()

# Drop 50% of rows of each of the two highest values of 'price'
if True:
    print(df.shape)
    highest_price_values = np.sort(df['price'].unique())[-2:]
    for value in highest_price_values:
        value_indices = df[df['price'] == value].index
        drop_indices = np.random.choice(value_indices, size=int(len(value_indices) * 0.5), replace=False)
        df = df.drop(drop_indices)
    print(df.shape)

df.groupby('price').size().plot()
plt.show()
df.groupby('price')['sales'].mean().plot()
plt.show()
    
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm_data_generator, df, override_models=False, quality=gcm.auto.AssignmentQuality.BETTER)
if False:
    print(auto_assignment_summary)

gcm.fit(scm_data_generator, df)
if False: # WARNING - this takes probably more than 10 minutes!
    print(gcm.evaluate_causal_model(scm_data_generator, df, compare_mechanism_baselines=True, evaluate_invertibility_assumptions=False))

scm_data_generator.causal_mechanism('sales')
scm_data_generator.causal_mechanism('price')
scm_data_generator.causal_mechanism('holiday_days')

### 5) Draw random samples
df_generated = gcm.draw_samples(scm_data_generator, num_samples=10000)
list_cols_df_generated = df_generated.columns.tolist()
df_generated['sales'] = df_generated['sales'] + abs(df_generated['sales'].min())

# prepare variables used in modelling
treatment_name='log_price'
outcome_name='log_sales'
list_cols_treatment_outcome = return_regex_cols(df_generated, 'price|sales')
X_cols = [name for name in df_generated.columns.tolist() if name not in list_cols_treatment_outcome]
df_generated['log_price'] = np.log(df_generated['price'] + 1)  # Adding 1 to avoid log(0)
df_generated['log_sales'] = np.log(df_generated['sales'] + 1)  # Adding 1 to avoid log(0)
# scale sales to be non zero:

plt.scatter(df_generated['price'], df_generated['sales'], color='blue')
plt.scatter(df['price'], df['sales'], color='orange')
plt.show()

### 6) Estimate ATE for reference
est = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = df_generated[outcome_name]
T = df_generated[treatment_name]
df_generated['store_type_numeric'] = df_generated['store_type'].astype('category').cat.codes + 1
X_cols_store_type_numeric = X_cols + ['store_type_numeric']
X_cols_store_type_numeric.remove('store_type')
W = df_generated[X_cols_store_type_numeric]
est.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
print(f"calculated price elasticity coeficient: {est.effect(T0=0, T1=1)[0]}")

# Calculate model performance for models_y and models_t
y_model_performance = [mdl.score(W, Y) for mdls in est.models_y for mdl in mdls]
t_model_performance = [mdl.score(W, T) for mdls in est.models_t for mdl in mdls]
print(f"Performance for Y first stage models (R^2): {np.mean(y_model_performance)}")
print(f"Performance for T first stage models (R^2): {np.mean(t_model_performance)}")
# Models for Treatment are almost perfect. This esentially means that residuals from model T are 0  .....
print(np.mean(est.nuisance_scores_t))
print(np.mean(est.nuisance_scores_y))


####################################################################################
### VALIDATIOJN PIPELINE:
### 1) Generate datasets
df_generated = gcm.draw_samples(scm_data_generator, num_samples=10000)
plt.scatter(df_generated['price'], df_generated['sales'], color='blue')
plt.show()
# Split the generated data into train and test sets
train_df, test_df = train_test_split(df_generated, test_size=0.2, random_state=42)
plt.scatter(train_df['price'], train_df['sales'], color='blue')
plt.scatter(test_df['price'], test_df['sales'], color='red')
plt.show()

avg_price = np.round(np.sum(train_df['sales'] * train_df['price']) / np.sum(train_df['sales']), 2)
intervention_price_avg = {'price': lambda price: avg_price}
df_test_price_avg = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_avg, test_df[list_cols_df_generated])

intervention_price_min = {'price': lambda price: df_generated['price'].min()}
df_test_price_min = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_min, test_df[list_cols_df_generated])

intervention_price_max = {'price': lambda price: df_generated['price'].max()}
df_test_price_max = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_max, test_df[list_cols_df_generated])

### 2) Fit models and predict outcomes
# .drop(columns=['sales'])
# 2a) predict using SCM:

################## HOW TO MAKE PREDICTIONS ON TEST DATA USING SCM? ##################
df_test_price_avg['sales_scm'] = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_avg, test_df[list_cols_df_generated])['sales']
df_test_price_min['sales_scm'] = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_min, test_df[list_cols_df_generated])['sales']
df_test_price_max['sales_scm'] = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_max, test_df[list_cols_df_generated])['sales']

# 2b) fit and predict using ML:
# preprocess for ML models
def assign_store_type_numeric(df):
    df_c = df.copy()
    df_c['store_type_numeric'] = df_c['store_type'].astype('category').cat.codes + 1
    df_c['store_type_numeric'] = df_c['store_type_numeric'].astype(int)
    return df_c

train_df_numeric = train_df.copy()
train_df_numeric = assign_store_type_numeric(train_df_numeric)
test_df_numeric = assign_store_type_numeric(test_df)
df_test_price_avg = assign_store_type_numeric(df_test_price_avg)
df_test_price_min = assign_store_type_numeric(df_test_price_min)
df_test_price_max = assign_store_type_numeric(df_test_price_max)

# fit and predict
ml_model = GradientBoostingRegressor()
list_cols_x_ml = list_cols_df_generated.copy()
list_cols_x_ml.append('store_type_numeric') 
list_cols_x_ml.remove('store_type')
list_cols_x_ml.remove('sales')
ml_model.fit(train_df_numeric[list_cols_x_ml], train_df_numeric['sales'])

df_test_price_avg['sales_ml'] = ml_model.predict(df_test_price_avg[list_cols_x_ml])
df_test_price_min['sales_ml'] = ml_model.predict(df_test_price_min[list_cols_x_ml])
df_test_price_max['sales_ml'] = ml_model.predict(df_test_price_max[list_cols_x_ml])

# 3) evaluate both approaches:
# Calculate MAPE and MSE for SCM predictions
def calculate_metrics(df, df_type):
    mape_scm = mean_absolute_percentage_error(df['sales'], df[f'sales_scm'])
    mse_scm = mean_squared_error(df['sales'], df['sales_scm'])
    mape_ml = mean_absolute_percentage_error(df['sales'], df[f'sales_ml'])
    mse_ml = mean_squared_error(df['sales'], df['sales_ml'])
    return pd.DataFrame({'model_type': ['SCM', 'ML'], 
                         'df_type' : [df_type, df_type],
                         'MAPE': [mape_scm, mape_ml], 
                         'MSE': [mse_scm, mse_ml]})

list_dfs_res = []
for i, name in zip([df_test_price_avg, df_test_price_min, df_test_price_max], ['avg', 'min', 'max']):
    list_dfs_res.append(calculate_metrics(i, name))

df_res = pd.concat(list_dfs_res)
df_res['MSE'] = df_res['MSE'] / 1000000
df_res.pivot_table(index='model_type', columns='df_type', values=['MAPE', 'MSE'])

