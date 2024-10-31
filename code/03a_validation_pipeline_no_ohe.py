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
import optuna
import lightgbm as lgb
from sklearn.model_selection import cross_val_score



### 1) Import data
PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop/projects/202410_DSSummit/repo/causalML_DSS24/data"
rescale = False

df = pd.read_csv(f"{PATH_DATA}/df_prepared.csv")
# drop year and month columns:
list_cols_drop = return_regex_cols(df, 'year_|month_')
df.drop(columns=list_cols_drop, inplace=True)

list_cols_ohe = ['year', 'month', 'store_type']
df_merge_back = df[list_cols_ohe].copy()
# create new columns
df = pd.get_dummies(df, columns=['year', 'month', 'store_type'], drop_first=True, dtype=int)
df = pd.concat([df, df_merge_back], axis=1)

if rescale:
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

graph_simple_for_plot = nx.DiGraph([
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


df.groupby('price').size()
plt.show()
df.groupby('price').size().plot()
plt.show()

# Drop 50% of rows of each of the two highest values of 'price'
if False:
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

### 4) Instantiate GCM and fit the model
# Create the structural causal model object
scm_data_generator = gcm.StructuralCausalModel(graph_simple_for_plot)
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm_data_generator, df, override_models=True, quality=gcm.auto.AssignmentQuality.GOOD) # BETTER
if False:
    print(auto_assignment_summary)

gcm.fit(scm_data_generator, df)
if False: # WARNING - this takes probably more than 10 minutes!
    print(gcm.evaluate_causal_model(scm_data_generator, df, compare_mechanism_baselines=True, evaluate_invertibility_assumptions=False))

list_attrs_data_generator = []
for attribute in dir(scm_data_generator):
    if not attribute.startswith('__'):
        list_attrs_data_generator.append(attribute)
print(list_attrs_data_generator)
if False:
    plot(scm_data_generator.graph)

scm_data_generator.causal_mechanism('sales')
scm_data_generator.causal_mechanism('price')
scm_data_generator.causal_mechanism('holiday_days')

list_attrs_causal_mechanism = []
for attribute in dir(scm_data_generator.causal_mechanism('sales')):
    if not attribute.startswith('__'):
        list_attrs_causal_mechanism.append(attribute)
print(list_attrs_causal_mechanism)

print(f"Noise model for sales node: {scm_data_generator.causal_mechanism('sales').noise_model}")
print(f"Prediction model for sales node: {scm_data_generator.causal_mechanism('sales').prediction_model}")

# the line below dramatically increases the variance of generated data and degrades quality of the model
if False:
    scm_data_generator.set_causal_mechanism('sales', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

### 5) Draw random samples
df_generated = gcm.draw_samples(scm_data_generator, num_samples=10000)
# One-hot encode 'year', 'month', and 'store_type' columns
df_generated = pd.get_dummies(df_generated, columns=['year', 'month', 'store_type'], drop_first=True).astype(int)

list_cols_df_generated = df_generated.columns.tolist()
df_generated['sales'] = df_generated['sales'] + abs(df_generated['sales'].min())

# compare sales ranges between original df and generated df
plt.scatter(df_generated['price'], df_generated['sales'], color='blue')
plt.scatter(df['price'], df['sales'], color='red')
plt.show()
# The variance in generated data is INSANE

# prepare variables used in modelling
treatment_name='log_price'
outcome_name='log_sales'
list_cols_treatment_outcome = return_regex_cols(df_generated, 'price|sales')
X_cols = [name for name in df_generated.columns.tolist() if name not in list_cols_treatment_outcome]
X_cols_price = X_cols + ['price']
df_generated['log_price'] = np.log(df_generated['price'] + 1)  # Adding 1 to avoid log(0)
df_generated['log_sales'] = np.log(df_generated['sales'] + 1)  # Adding 1 to avoid log(0)
# scale sales to be non zero:


### 6) Estimate ATE for reference
est = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = df_generated[outcome_name]
T = df_generated[treatment_name]
W = df_generated[X_cols]
est.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
print(f"calculated price elasticity coeficient: {est.effect(T0=0, T1=1)[0]}")

def print_dml_coef(df, outcome_name=outcome_name, treatment_name=treatment_name, X_cols=X_cols):
    df_est = df.copy()
    est = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
    df_est = df_est[df_est['sales'] >= 0]
    Y = df_est[outcome_name]
    T = df_est[treatment_name]
    W = df_est[X_cols]
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
df_generated_modeling = gcm.draw_samples(scm_data_generator, num_samples=10000)

# One-hot encode 'year', 'month', and 'store_type' columns
df_generated_modeling = df_generated_modeling[df_generated_modeling['sales'] >= 0]
df_generated_modeling = df_generated_modeling[(df_generated_modeling['price'] >= df['price'].min()) & 
                                              (df_generated_modeling['price'] <= df['price'].max())]
df_merge_back_modeling = df_generated_modeling[list_cols_ohe].copy()
# create new columns
df_generated_modeling = pd.get_dummies(df_generated_modeling, columns=['year', 'month', 'store_type'], drop_first=True).astype(int)
df_generated_modeling = pd.concat([df_generated_modeling, df_merge_back_modeling], axis=1)

print(f"Number of rows in df_generated_modeling after removing sales<0 rows: {df_generated_modeling.shape[0]}")
plt.scatter(df_generated_modeling['price'], df_generated_modeling['sales'], color='blue')
plt.show()
# Split the generated data into train and test sets
train_df, test_df = train_test_split(df_generated_modeling, test_size=0.2, random_state=42)
plt.scatter(train_df['price'], train_df['sales'], color='blue')
plt.scatter(test_df['price'], test_df['sales'], color='red')
plt.show()
test_df['sales'].mean()
test_df_orig = test_df.copy()

train_df['log_price'] = np.log(df_generated_modeling['price'] + 1)  # Adding 1 to avoid log(0)
train_df['log_sales'] = np.log(df_generated_modeling['sales'] + 1)  # Adding 1 to avoid log(0)

# check treatment effect in df_train
print("Elasticity coefficient in original data:")
print_dml_coef(df)
print("Elasticity coefficient in synthetic data data:")
print_dml_coef(train_df)


avg_price = np.round(np.sum(train_df['sales'] * train_df['price']) / np.sum(train_df['sales']), 0)
avg_price = np.round(train_df['price'].mean(), 0)
intervention_price_avg = {'price': lambda price: avg_price}
df_test_price_avg = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_avg, test_df)
print(f"Average sales: {df_test_price_avg['sales'].mean()} for price: {avg_price}")
plt.scatter(df_test_price_avg['price'], df_test_price_avg['sales'], color='blue')
plt.show()

intervention_price_min = {'price': lambda price: df_generated_modeling['price'].min()}
df_test_price_min = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_min, test_df)
print(f"Average sales: {df_test_price_min['sales'].mean()} for price: {df_generated_modeling['price'].min()}")
plt.scatter(df_test_price_min['price'], df_test_price_min['sales'], color='blue')
plt.show()

intervention_price_max = {'price': lambda price: df_generated_modeling['price'].max()}
df_test_price_max = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_max, test_df)
print(f"Average sales: {df_test_price_max['sales'].mean()} for price: {df_generated_modeling['price'].max()}")
plt.scatter(df_test_price_max['price'], df_test_price_max['sales'], color='blue')
plt.show()



### 2) Fit models and predict outcomes
# 2a) predict using SCM:

################## HOW TO MAKE PREDICTIONS ON TEST DATA USING SCM? ##################
df_test_price_avg['sales_scm'] = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_avg, test_df.drop(columns=['sales']))['sales']
df_test_price_min['sales_scm'] = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_min, test_df.drop(columns=['sales']))['sales']
df_test_price_max['sales_scm'] = dowhy.gcm.whatif.interventional_samples(scm_data_generator, intervention_price_max, test_df.drop(columns=['sales']))['sales']
df_test_price_avg['resid_scm'] = df_test_price_avg['sales'] - df_test_price_avg['sales_scm']
df_test_price_min['resid_scm'] = df_test_price_min['sales'] - df_test_price_min['sales_scm']
df_test_price_max['resid_scm'] = df_test_price_max['sales'] - df_test_price_max['sales_scm']

# remove price outliers and sales<0 rows:
df_test_price_avg_fltrd = df_test_price_avg[df_test_price_avg['sales'] >= 0]
df_test_price_avg_fltrd = df_test_price_avg[(df_test_price_avg['price'] >= df['price'].min()) & 
                                            (df_test_price_avg['price'] <= df['price'].max())]
df_test_price_min_fltrd = df_test_price_min[df_test_price_min['sales'] >= 0]
df_test_price_min_fltrd = df_test_price_min[(df_test_price_min['price'] >= df['price'].min()) & 
                                            (df_test_price_min['price'] <= df['price'].max())]
df_test_price_max_fltrd = df_test_price_max[df_test_price_max['sales'] >= 0]
df_test_price_max_fltrd = df_test_price_max[(df_test_price_max['price'] >= df['price'].min()) & 
                                            (df_test_price_max['price'] <= df['price'].max())]
plt.scatter(df_test_price_avg_fltrd['price'], df_test_price_avg_fltrd['sales'], color='blue')
plt.show()
plt.scatter(df_test_price_min_fltrd['price'], df_test_price_min_fltrd['sales'], color='blue')
plt.show()
plt.scatter(df_test_price_max_fltrd['price'], df_test_price_max_fltrd['sales'], color='blue')
plt.show()


# 2b) fit and predict using ML:
# fit and predict
# fit and predict
def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'num_trees': trial.suggest_int('num_trees', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
    }
    
    lgb_train = lgb.Dataset(train_df[X_cols_price], train_df['sales'])
    cv_results = lgb.cv(param, lgb_train, nfold=3, metrics='rmse', seed=42)
    
    # Print the keys to verify the correct key
    print(f"printing cv results keys: {cv_results.keys()}")
    
    # Use the correct key for accessing the results
    return np.mean(cv_results['valid rmse-mean'])

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

if False:
    list_attrs_study = []
    for attribute in dir(study):
        if not attribute.startswith('__'):
            list_attrs_study.append(attribute)
    print(list_attrs_study)

print(study.best_value) # 13529
best_params = study.best_params
print(f"Best parameters: {best_params}")

ml_model = lgb.LGBMRegressor(**best_params)
ml_model.fit(train_df[X_cols_price], train_df['sales'])

df_test_price_avg['sales_ml'] = ml_model.predict(df_test_price_avg[X_cols_price])
df_test_price_min['sales_ml'] = ml_model.predict(df_test_price_min[X_cols_price])
df_test_price_max['sales_ml'] = ml_model.predict(df_test_price_max[X_cols_price])
df_test_price_avg['resid_ml'] = df_test_price_avg['sales'] - df_test_price_avg['sales_ml']
df_test_price_min['resid_ml'] = df_test_price_min['sales'] - df_test_price_min['sales_ml']
df_test_price_max['resid_ml'] = df_test_price_max['sales'] - df_test_price_max['sales_ml']

# plot residuals
plt.hist(df_test_price_avg['resid_scm'], bins=50, alpha=0.5, color='blue', label='SCM')
plt.hist(df_test_price_avg['resid_ml'], bins=50, alpha=0.5, color='red', label='ML')
plt.title("Residuals for avg price intervention")
plt.legend()
plt.show()
plt.hist(df_test_price_min['resid_scm'], bins=50, alpha=0.5, color='blue', label='SCM')
plt.hist(df_test_price_min['resid_ml'], bins=50, alpha=0.5, color='red', label='ML')
plt.title("Residuals for min price intervention")
plt.legend()
plt.show()
plt.hist(df_test_price_max['resid_scm'], bins=50, alpha=0.5, color='blue', label='SCM')
plt.hist(df_test_price_max['resid_ml'], bins=50, alpha=0.5, color='red', label='ML')
plt.title("Residuals for max price intervention")
plt.legend()
plt.show()

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
print(df_res.pivot_table(index='model_type', columns='df_type', values=['MAPE', 'MSE']))

######################################################################
#### Test LOG-LOG doubleML approach to correct for different prices
# estimate price elasticity model
est_modeling = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df[outcome_name]
T = train_df[treatment_name]
W = train_df[X_cols]
est_modeling.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
print(f"calculated price elasticity coeficient: {est_modeling.effect(T0=0, T1=1)[0]}")
plt.scatter(train_df['price'], train_df['sales'], color='blue')
plt.show()


# add necessary variables to test_df:
test_df_dml = test_df_orig.copy()
test_df_dml = test_df_dml[(test_df_dml['price'] >= df['price'].min()) & 
                                              (test_df_dml['price'] <= df['price'].max())]

test_df_dml['avg_price'] = avg_price
test_df_dml['min_price'] = df_generated_modeling['price'].min()
test_df_dml['max_price'] = df_generated_modeling['price'].max()

test_df_dml['log_sales'] = np.log(test_df_dml['sales'] + 1)  # Adding 1 to avoid log(0)
test_df_dml['log_price'] = np.log(test_df_dml['price'] + 1)  # Adding 1 to avoid log(0)
test_df_dml['log_avg_price'] = np.log(test_df_dml['avg_price'] + 1)  # Adding 1 to avoid log(0)
test_df_dml['log_min_price'] = np.log(test_df_dml['min_price'] + 1)  # Adding 1 to avoid log(0)
test_df_dml['log_max_price'] = np.log(test_df_dml['max_price'] + 1)  # Adding 1 to avoid log(0)

# Retrieve actual sales using logarithms and price elasticity coefficient
price_elasticity_coef = est_modeling.effect(T0=0, T1=1)[0]

# Calculate predicted log_sales for different price interventions
test_df_dml['log_sales_avg'] = test_df_dml['log_sales'] + price_elasticity_coef * (test_df_dml['log_avg_price'] - test_df_dml['log_price'])
test_df_dml['log_sales_min'] = test_df_dml['log_sales'] + price_elasticity_coef * (test_df_dml['log_min_price'] - test_df_dml['log_price'])
test_df_dml['log_sales_max'] = test_df_dml['log_sales'] + price_elasticity_coef * (test_df_dml['log_max_price'] - test_df_dml['log_price'])

# Convert log_sales back to actual sales
test_df_dml['sales_dml_avg'] = np.exp(test_df_dml['log_sales_avg']) - 1  # Subtract 1 to reverse the earlier addition
test_df_dml['sales_dml_min'] = np.exp(test_df_dml['log_sales_min']) - 1
test_df_dml['sales_dml_max'] = np.exp(test_df_dml['log_sales_max']) - 1


# add actual sales generated by SCM and sales from ML model:
test_df_dml['sales_scm_avg'] = df_test_price_avg_fltrd['sales_scm']
test_df_dml['sales_scm_min'] = df_test_price_avg_fltrd['sales_scm']
test_df_dml['sales_scm_max'] = df_test_price_avg_fltrd['sales_scm']

# inspect one row
test_df_dml[['price', 'avg_price', 'sales', 'sales_dml_avg', 'sales_scm_avg','log_sales_avg', 'log_sales', 'log_avg_price', 'log_price']].iloc[:20]

# ML model:
test_df_dml['price_orig'] = test_df_dml['price']
for sales_type in ['avg', 'min', 'max']:
    test_df_dml['price'] = test_df_dml[f'{sales_type}_price']
    test_df_dml[f'sales_ml_{sales_type}'] = ml_model.predict(test_df_dml[X_cols_price])

plt.scatter(test_df_dml['price_orig'], test_df_dml['sales'], color='blue')
plt.show()

def calculate_metrics_dml(df, sales_type):
    mape_dml = mean_absolute_percentage_error(df[f'sales_scm_{sales_type}'], df[f'sales_dml_{sales_type}'])
    mse_dml = mean_squared_error(df[f'sales_scm_{sales_type}'], df[f'sales_dml_{sales_type}'])
    mape_ml = mean_absolute_percentage_error(df[f'sales_scm_{sales_type}'], df[f'sales_ml_{sales_type}'])
    mse_ml = mean_squared_error(df[f'sales_scm_{sales_type}'], df[f'sales_ml_{sales_type}'])
    return pd.DataFrame({'model_type': ['DML', 'ML'], 
                         'MAPE': [mape_dml, mape_ml], 
                         'MSE': [mse_dml, mse_ml]})



list_df_res = []
for sales_type in ['avg', 'min', 'max']:
    df_res = calculate_metrics_dml(test_df_dml, sales_type)
    df_res['df_type'] = sales_type
    list_df_res.append(df_res)

df_res = pd.concat(list_df_res)
df_res['MSE'] = df_res['MSE'] / 1000000
print(df_res.pivot_table(index='model_type', columns='df_type', values=['MAPE', 'MSE']))


######################################################################
#### Test LINEAR doubleML approach to correct for different prices
est_modeling = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df['sales']
T = train_df['price']
W = train_df[X_cols]
est_modeling.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
print(f"calculated price-sales coeficient: {est_modeling.effect(T0=0, T1=1)[0]}")
plt.scatter(train_df['price'], train_df['sales'], color='blue')
plt.show()


# add necessary variables to test_df:
test_df_dml = test_df_orig.copy()
test_df_dml = test_df_dml[(test_df_dml['price'] >= df['price'].min()) & 
                                              (test_df_dml['price'] <= df['price'].max())]

test_df_dml['avg_price'] = avg_price
test_df_dml['min_price'] = df_generated_modeling['price'].min()
test_df_dml['max_price'] = df_generated_modeling['price'].max()


# add actual sales generated by SCM and sales from ML model:
test_df_dml['sales_scm_avg'] = df_test_price_avg['sales_scm']
test_df_dml['sales_scm_min'] = df_test_price_min['sales_scm']
test_df_dml['sales_scm_max'] = df_test_price_max['sales_scm']

# ML & DML model:
test_df_dml['price_orig'] = test_df_dml['price']
for sales_type in ['avg', 'min', 'max']:
    test_df_dml['price'] = test_df_dml[f'{sales_type}_price']
    test_df_dml[f'sales_ml_{sales_type}'] = ml_model.predict(test_df_dml[X_cols_price])
    test_df_dml[f'sales_dml_{sales_type}'] = test_df_dml['sales'] + est_modeling.effect(T0=test_df_dml['price_orig'], T1=test_df_dml['price'])

# inspect one row
test_df_dml[['price_orig', 'avg_price', 'sales', 'sales_dml_avg', 'sales_scm_avg']].iloc[:20]

plt.scatter(test_df_dml['price_orig'], test_df_dml['sales'], color='blue')
plt.show()

def calculate_metrics_dml(df, sales_type):
    mape_dml = mean_absolute_percentage_error(df[f'sales_scm_{sales_type}'], df[f'sales_dml_{sales_type}'])
    mse_dml = mean_squared_error(df[f'sales_scm_{sales_type}'], df[f'sales_dml_{sales_type}'])
    mape_ml = mean_absolute_percentage_error(df[f'sales_scm_{sales_type}'], df[f'sales_ml_{sales_type}'])
    mse_ml = mean_squared_error(df[f'sales_scm_{sales_type}'], df[f'sales_ml_{sales_type}'])
    return pd.DataFrame({'model_type': ['DML', 'ML'], 
                         'MAPE': [mape_dml, mape_ml], 
                         'MSE': [mse_dml, mse_ml]})



list_df_res = []
for sales_type in ['avg', 'min', 'max']:
    df_res = calculate_metrics_dml(test_df_dml, sales_type)
    df_res['df_type'] = sales_type
    list_df_res.append(df_res)

df_res = pd.concat(list_df_res)
df_res['MSE'] = df_res['MSE'] / 1000000
print(df_res.pivot_table(index='model_type', columns='df_type', values=['MAPE', 'MSE']))
