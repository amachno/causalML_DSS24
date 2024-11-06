## Generate syntethic data based on actual data from Kaggle
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import cross_val_score

from econml.dml import LinearDML
import optuna
import lightgbm as lgb
import networkx as nx

from utils import *

# PARAMS
n_trials = 10 # number of trials for hyperparameter tuning for both first-stage DML and ML model


### 1) Import data
PATH_DATA = "C:/Users/artur.machno/git_repos/causalML_DSS24/data"

df = pd.read_csv(f"{PATH_DATA}/df_prepared.csv")
df = change_store_type_type(df)

df[['sales']].describe()

# Rescale 'sales' column to be between 10000 and 50000 (to ensure generated data has no negative sales values)
min_sales = df['sales'].min()
max_sales = df['sales'].max()
df['sales'] = 10000 + (df['sales'] - min_sales) * (50000 - 10000) / (max_sales - min_sales)


# drop OHE year and month columns:
list_cols_drop = return_regex_cols(df, 'year_|month_')
df.drop(columns=list_cols_drop, inplace=True)

# create collider variable
df['competitor_sales'] = df['sales'] * np.random.uniform(0.4, 0.6) * (df['price'] * np.random.uniform(0.02, 0.1)) + np.random.normal(0, 0.1, df.shape[0]) * 100

# plt.scatter(df['sales'], df['competitor_sales'])

### 2) Create causal graph
df.columns

causal_graph = nx.DiGraph([
                ('holiday_days', 'sales'),
                ('holiday_days', 'price'),
                ('is_lockdown', 'sales'),
                ('curfew', 'sales'),
                ('year', 'sales'),
                ('month', 'sales'),
                ('store_type', 'price'),
                ('store_type', 'sales'),
                ('week_number_trend', 'price'),
                ('week_number_trend', 'sales'),
                ('price', 'sales'),
                ('sales', 'competitor_sales'),
                ('price', 'competitor_sales'),
                ])

causal_graph_log = nx.DiGraph([
                    ('holiday_days', 'log_sales'),
                    ('holiday_days', 'log_price'),
                    ('is_lockdown', 'log_sales'),
                    ('curfew', 'log_sales'),
                    ('year', 'log_sales'),
                    ('month', 'log_sales'),
                    ('store_type', 'log_price'),
                    ('store_type', 'log_sales'),
                    ('week_number_trend', 'log_price'),
                    ('week_number_trend', 'log_sales'),
                    ('log_price', 'log_sales'),
                    ('log_sales', 'competitor_sales'),
                    ('log_price', 'competitor_sales'),
                    ])

# potentially include city_id as categorical variable as well?

plot(causal_graph)

### 3) Instantiate GCM and fit the model
# Create the structural causal model object
scm_data_generator = gcm.StructuralCausalModel(causal_graph)
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm_data_generator, df, override_models=True, quality=gcm.auto.AssignmentQuality.GOOD) # BETTER
if False:
    print(auto_assignment_summary)

gcm.fit(scm_data_generator, df)
if False: # WARNING - this takes probably more than 10 minutes!
    print(gcm.evaluate_causal_model(scm_data_generator, df, compare_mechanism_baselines=True, evaluate_invertibility_assumptions=False))

# check the causal mechanisms
print(f"Noise model for sales node: {scm_data_generator.causal_mechanism('sales').noise_model}")
print(f"Prediction model for sales node: {scm_data_generator.causal_mechanism('sales').prediction_model}")

### 5) Draw random samples
df_generated = gcm.draw_samples(scm_data_generator, num_samples=10000)
print(f"Minimum sales value: {df_generated['sales'].min()}")

# compare sales ranges between original df and generated df
plt.scatter(df_generated['price'], df_generated['sales'], color='blue', label='synthetic')
plt.scatter(df['price'], df['sales'], color='red', label='true')
plt.legend()
plt.show()

# prepare variables used in modelling
df_generated['log_price'] = np.log(df_generated['price'] + 1)  # Adding 1 to avoid log(0)
df_generated['log_sales'] = np.log(df_generated['sales'] + 1)  # Adding 1 to avoid log(0)

############# remove outliying prices - SHOULD WE? #############
df_generated = df_generated[(df_generated['price']>=df['price'].min()) & 
                            (df_generated['price']<=df['price'].max())]
print(f"N rows in df_generted after removing outliers: {len(df_generated)}")

### 6) Split the generated data into train and test sets
train_df, test_df = train_test_split(df_generated, test_size=0.2, random_state=42)

# Generate "ground truth" datasets for different price interventions
############# SHOULD PRICE BE TAKEN FROM GENERATED DATASET OR RAW DATASET? #############
dict_prices_intervention = {
    'avg': np.round(train_df['price'].mean(), 0),
    'min': np.round(train_df['price'].min(), 0),
    'max': np.round(train_df['price'].max(), 0)
}
intervention_price_avg = {'price': lambda price: dict_prices_intervention['avg']}
df_test_avg_price = generate_test_dataset(scm_data_generator, intervention_price_avg, test_df)
df_test_avg_price['intervention'] = 'avg'

intervention_price_min = {'price': lambda price: dict_prices_intervention['min']}
df_test_min_price = generate_test_dataset(scm_data_generator, intervention_price_min, test_df)
df_test_min_price['intervention'] = 'min'

intervention_price_max = {'price': lambda price: dict_prices_intervention['max']}
df_test_max_price = generate_test_dataset(scm_data_generator, intervention_price_max, test_df)
df_test_max_price['intervention'] = 'max'

df_test_interevention = pd.concat([df_test_avg_price, df_test_min_price, df_test_max_price])
df_test_interevention.rename(columns={'sales': 'ground_truth_sales'}, inplace=True)
df_test_interevention['index'] = df_test_interevention.index

### 7) prepare variable names
treatment_name='price'
outcome_name='sales'
treatment_name_log='log_price'
outcome_name_log='log_sales'
list_cols_treatment_outcome = return_regex_cols(train_df, 'price|sales|log_price|log_sales')
X_cols = [name for name in train_df.columns.tolist() if name not in list_cols_treatment_outcome]
X_cols_no_collider = [name for name in X_cols if name != 'competitor_sales']
X_cols_price = X_cols + ['price']

### 8) estimate causal model:
# tune first stage for linear DML
objective_t = create_objective_function_DML(train_df, treatment_name, X_cols_no_collider)
# Run the hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective_t, n_trials=n_trials)
best_params_t = study.best_params
print(f"Best parameters t: {best_params_t}")

objective_y = create_objective_function_DML(train_df, outcome_name, X_cols_no_collider)
# Run the hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective_y, n_trials=n_trials)
best_params_y = study.best_params
print(f"Best parameters t: {best_params_y}")

objective_t_log = create_objective_function_DML(train_df, treatment_name_log, X_cols_no_collider)
# Run the hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective_t_log, n_trials=n_trials)
best_params_t_log = study.best_params
print(f"Best parameters t: {best_params_t_log}")

objective_y_log = create_objective_function_DML(train_df, outcome_name_log, X_cols_no_collider)
# Run the hyperparameter tuning
study = optuna.create_study(direction='minimize')
study.optimize(objective_y_log, n_trials=n_trials)
best_params_y_log = study.best_params
print(f"Best parameters t: {best_params_y_log}")

# Use the best parameters for the GradientBoostingRegressor in LinearDML
model_y = lgb.LGBMRegressor(**best_params_y)
model_t = lgb.LGBMRegressor(**best_params_t)

# Use the best parameters for the GradientBoostingRegressor in LinearDML
model_y_log = lgb.LGBMRegressor(**best_params_y_log)
model_t_log = lgb.LGBMRegressor(**best_params_t_log)

# Estimate DML models
# Linear model
cusal_model_lin = LinearDML(model_y=model_y, model_t=model_t, fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df[outcome_name]
T = train_df[treatment_name]
W = train_df[X_cols_no_collider]
cusal_model_lin.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
print(f"calculated effect for linear model: {cusal_model_lin.effect(T0=0, T1=1)[0]}")

# log-log model
cusal_model_log = LinearDML(model_y=model_y_log, model_t=model_t_log, fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df[outcome_name_log]
T = train_df[treatment_name_log]
W = train_df[X_cols_no_collider]
cusal_model_log.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
print(f"calculated effect for log model: {cusal_model_log.effect(T0=0, T1=1)[0]}")

## ML model
# Define the objective function for hyperparameter tuning
best_params = tune_lgbm(train_df, X_cols_price, 'sales', n_trials)

ml_model = lgb.LGBMRegressor(**best_params)
ml_model.fit(train_df[X_cols_price], train_df['sales'])

### 8) Generate predictions for 3 interventions using all models:
# ML model:
test_df['price_orig'] = test_df['price']
list_test_dfs_pred = []
for p_name,p_value in dict_prices_intervention.items():
    df_test_tmp = test_df.copy()
    df_test_tmp['price'] = p_value
    df_test_tmp['intervention'] = p_name
    df_test_tmp['ml_pred'] = ml_model.predict(df_test_tmp[X_cols_price])
    list_test_dfs_pred.append(df_test_tmp)

df_test_preds = pd.concat(list_test_dfs_pred)

# linear DML:
df_test_preds['dml_lin_pred'] = df_test_preds['sales'] + cusal_model_lin.effect(T0=df_test_preds['price_orig'], T1=df_test_preds['price'])
# log-linear DML:
df_test_preds['log_price_intervention'] = np.log(df_test_preds['price'] + 1)
# df_test_preds['dml_lin_pred'] = df_test_preds['sales'] + cusal_model_lin.effect(T0=df_test_preds['price_orig'], T1=df_test_preds['price'])
df_test_preds['dml_log_pred_log'] = df_test_preds['log_sales'] + cusal_model_log.effect(T0=df_test_preds['log_price'], T1=df_test_preds['log_price_intervention'])
df_test_preds['dml_log_pred'] = np.exp(df_test_preds['dml_log_pred_log']) - 1

# attach true values:
df_test_preds['index'] = df_test_preds.index
df_test_preds = pd.merge(df_test_preds, df_test_interevention[['index', 'intervention', 'ground_truth_sales']], on = ['index', 'intervention'], how='inner')


# Calculate metrics:
df_metrics = calculate_grouped_metrics(df_test_preds)
print(df_metrics)

plot_metrics(df_metrics)


