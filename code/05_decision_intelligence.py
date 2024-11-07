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
PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop/projects/202410_DSSummit/repo/causalML_DSS24/data"
rescale = True
manual_corrections = True

df = pd.read_csv(f"{PATH_DATA}/df_prepared.csv")
df = change_store_type_type(df)

df.groupby('store_type')['sales'].mean().plot(kind='bar')
df.groupby('store_type')['sales'].describe()

# Scale 'sales' values to have the same average for each 'store_type'
if True:
    store_type_means = df.groupby('store_type')['sales'].mean()
    overall_mean = df['sales'].mean()

    df['sales'] = df.apply(lambda row: row['sales'] * (overall_mean / store_type_means[row['store_type']]), axis=1)

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

# Rescale 'sales' column to be between 10000 and 50000 (to ensure generated data has no negative sales values)
min_sales = df['sales'].min()
max_sales = df['sales'].max()
df['sales'] = 10000 + (df['sales'] - min_sales) * (50000 - 10000) / (max_sales - min_sales)


# drop OHE year and month columns:
list_cols_drop = return_regex_cols(df, 'year_|month_')
df.drop(columns=list_cols_drop, inplace=True)

# create collider variable
df['competitor_sales'] = df['sales'] * np.random.uniform(0.4, 0.6) * (df['price'] * np.random.uniform(0.02, 0.1)) + np.random.normal(0, 0.1, df.shape[0]) * 100

if manual_corrections:
    # Perform manual corrections on column price
    df.loc[df['price'] == 32, 'price'] = 31
    df.loc[df['price'] > 32, 'price'] -= 3

    # remove outliers:
    df = df[df['sales'] != df['sales'].max()]
    # Remove observations with top 3 highest sales for rows where price is 27, 28, or 29
    for price in [27, 28, 29]:
        df_price = df[df['price'] == price]
        top_3_sales_indices = df_price.nlargest(3, 'sales').index
        df = df.drop(top_3_sales_indices)

plt.scatter(df['price'], df['sales'])
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
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm_data_generator, df, override_models=True, quality=gcm.auto.AssignmentQuality.BETTER) # BETTER
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
causal_model_lin = LinearDML(model_y=model_y, model_t=model_t, fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df[outcome_name]
T = train_df[treatment_name]
# exclude year from W at use it as source of heterogeneity
X_cols_DML = [col for col in X_cols_no_collider if col not in 'year']
W = train_df[X_cols_no_collider]
X = train_df[['year']]
causal_model_lin.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
# causal_model_lin.effect(T0=0, T1=1, X=train_df[['year']].drop_duplicates())
print(f"calculated effect for linear model: {causal_model_lin.effect(T0=0, T1=1)[0]}")
help(causal_model_lin.effect)
# log-log model
causal_model_log = LinearDML(model_y=model_y_log, model_t=model_t_log, fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df[outcome_name_log]
T = train_df[treatment_name_log]
causal_model_log.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
# causal_model_log.effect(T0=0, T1=1, X=train_df[['year']].drop_duplicates())
print(f"calculated effect for log model: {causal_model_log.effect(T0=0, T1=1)[0]}")

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
df_test_preds['index'] = df_test_preds.index

# linear DML:
df_test_preds['dml_lin_pred'] = df_test_preds['sales'] + causal_model_lin.effect(T0=df_test_preds['price_orig'], T1=df_test_preds['price']) # , X=df_test_preds[['year']]
if False: # use ML model prediction as baseline for effect application
    df_test_preds['dml_lin_pred'] = ml_model.predict(df_test_preds[X_cols_price]) + causal_model_lin.effect(T0=df_test_preds['price_orig'], T1=df_test_preds['price'])
# log-linear DML:
df_test_preds['log_price_intervention'] = np.log(df_test_preds['price'] + 1)
df_test_preds['dml_log_pred_log'] = df_test_preds['log_sales'] + causal_model_log.effect(T0=df_test_preds['log_price'], T1=df_test_preds['log_price_intervention']) #, X=df_test_preds[['year']]
if False: # use ML model prediction as baseline for effect application
    df_test_preds['dml_log_pred_log'] = np.log(ml_model.predict(df_test_preds[X_cols_price]) + 1) + causal_model_log.effect(T0=df_test_preds['log_price'], T1=df_test_preds['log_price_intervention']) # , X=df_test_preds[['year']]
df_test_preds['dml_log_pred'] = np.exp(df_test_preds['dml_log_pred_log']) - 1

# attach true values:
df_test_preds_for_metrics = pd.merge(df_test_preds, df_test_interevention[['index', 'intervention', 'ground_truth_sales']], on = ['index', 'intervention'], how='inner')

for intervention in df_test_preds_for_metrics['intervention'].unique():
    df_tmp = df_test_preds_for_metrics[df_test_preds_for_metrics['intervention']==intervention]
    print(f"Intervention type: price {intervention}")
    plot_residuals(
        df_tmp,
        ground_truth_col='ground_truth_sales',
        pred_cols=['ml_pred', 'dml_lin_pred', 'dml_log_pred'],
        titles=['Residuals for ML model', 'Residuals for Linear DML model', 'Residuals for Log-Log DML model']
    )


# Calculate metrics:
df_metrics = calculate_grouped_metrics(df_test_preds_for_metrics)
print(df_metrics)

plot_metrics(df_metrics)

######################################################
# Decision Intelligence:
# Podzielenie df_test na pół
# Na jednej połowie wyznaczenie optymalnej strategii:
# Wyznaczenie kosztu produktu jako 70-80% ceny minimalnej (marża = 20-30% + róznica względem minimalnej)
# Zrobienie predykcji dla każdej wartości ceny z przedziału <min; max>
# Wybranie ceny maksymalizującej marżę (zakładamy, że znamy realną sprzedaż dla tej części df_test)
# Zaaplikowanie ceny maksymalizującej marżę na drugiej połowie df_test
# Wygenerowanie "ground truth" dla wyznaczonej ceny w punkcie c.
# Policzenie masy marży

# Divide df_test into two halves
df_test_for_price_optimization = test_df.sample(frac=0.5, random_state=42)
df_test_for_optimal_price_application = test_df.drop(df_test_for_price_optimization.index)

def get_optimal_price_ml(df, ml_model, X_cols_price, sku_unit_cost):
    df_tmp = df.copy()
    dict_margins = {}
    for price_point in np.sort(df_tmp['price'].unique()):
        df_tmp['price'] = price_point
        df_tmp['unit_cost'] = sku_unit_cost
        df_tmp['predicted_sales'] = ml_model.predict(df_tmp[X_cols_price])
        df_tmp['unit_margin'] = (df_tmp['price'] - df_tmp['unit_cost']) * df_tmp['predicted_sales']
        dict_margins[price_point] = df_tmp['unit_margin'].sum()
    optimal_price = max(dict_margins, key=dict_margins.get)
    return optimal_price

def get_optimal_price_log_dml(df, causal_model_log, sku_unit_cost):
    df_tmp = df.copy()
    dict_margins = {}
    for price_point in np.sort(df_tmp['price'].unique()):
        df_tmp['price'] = price_point
        df_tmp['unit_cost'] = sku_unit_cost
        df_tmp['log_price_intervention'] = np.log(df_tmp['price'] + 1)
        df_tmp['dml_log_pred_log'] = df_tmp['log_sales'] + causal_model_log.effect(T0=df_tmp['log_price'], T1=df_tmp['log_price_intervention'])
        df_tmp['predicted_sales'] = np.exp(df_tmp['dml_log_pred_log']) - 1
        df_tmp['unit_margin'] = (df_tmp['price'] - df_tmp['unit_cost']) * df_tmp['predicted_sales']
        dict_margins[price_point] = df_tmp['unit_margin'].sum()
    optimal_price = max(dict_margins, key=dict_margins.get)
    return optimal_price

def apply_optimal_price(df, optimal_price, sku_unit_cost, scm_data_generator):
    df_tmp = df.copy()
    df_tmp['price'] = optimal_price
    intervention_price_optimal = {'price': lambda price: optimal_price}
    df_tmp['sales'] = generate_test_dataset(scm_data_generator, intervention_price_optimal, df_tmp, verbose=False)['sales']
    df_tmp['unit_cost'] = sku_unit_cost
    df_tmp['unit_margin'] = (df_tmp['price'] - df_tmp['unit_cost']) * df_tmp['sales']
    margin_sum = df_tmp['unit_margin'].sum()
    sales_sum = df_tmp['sales'].sum()
    return margin_sum, sales_sum

list_prices_ml = []
list_margins_ml = []
list_sales_ml = []
list_prices_log_dml = []
list_margins_log_dml = []
list_sales_log_dml = []
i = 1
while i < 50:
    # select unit cost randomly
    df_test_for_price_optimization = test_df.sample(frac=0.5)
    df_test_for_optimal_price_application = test_df.drop(df_test_for_price_optimization.index)  
    sku_unit_cost = np.round(test_df['price'].min() * 0.7, 2) # np.random.uniform(0.3, 0.7)
    # ML model:
    optimal_price = get_optimal_price_ml(df_test_for_price_optimization, ml_model, X_cols_price, sku_unit_cost)
    margin_sum, sales_sum = apply_optimal_price(df_test_for_optimal_price_application, optimal_price, sku_unit_cost, scm_data_generator)
    list_prices_ml.append(optimal_price)
    list_margins_ml.append(margin_sum)
    list_sales_ml.append(sales_sum)
    # log DML model:
    optimal_price = get_optimal_price_log_dml(df_test_for_price_optimization, causal_model_log, sku_unit_cost)
    margin_sum, sales_sum = apply_optimal_price(df_test_for_optimal_price_application, optimal_price, sku_unit_cost, scm_data_generator)
    list_prices_log_dml.append(optimal_price)
    list_margins_log_dml.append(margin_sum)
    list_sales_log_dml.append(sales_sum)
    i+=1

df_results_margin = pd.DataFrame({'optimal_price': list_prices_ml, 
                                  'margin_sum': list_margins_ml,
                                  'sales_sum': list_sales_ml,
                                  'optimal_price_log_dml': list_prices_log_dml, 
                                  'margin_sum_log_dml': list_margins_log_dml,
                                  'sales_sum_log_dml': list_sales_log_dml})

df_results_margin[['margin_sum', 'margin_sum_log_dml']] = (df_results_margin[['margin_sum', 'margin_sum_log_dml']] / 10000).round().astype(int)
df_results_margin[['sales_sum', 'sales_sum_log_dml']] = (df_results_margin[['sales_sum', 'sales_sum_log_dml']] / 1000).round().astype(int)

plt.hist(df_results_margin['margin_sum'], bins=30, alpha=0.5, label='ML Model')
plt.hist(df_results_margin['margin_sum_log_dml'], bins=30, alpha=0.5, label='Log DML Model')
plt.xlabel('Margin Sum')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Overlying Histogram of Margin Sum for ML and Log DML Models')
plt.show()


# Select rows with minimum and maximum values for both margin_sum and margin_sum_log_dml
df_min_max_ml = pd.DataFrame({
    'optimal_price': [df_results_margin.loc[df_results_margin['margin_sum'].idxmin(), 'optimal_price'],
                      df_results_margin.loc[df_results_margin['margin_sum'].idxmax(), 'optimal_price']],
    'margin_sum': [df_results_margin['margin_sum'].min(), df_results_margin['margin_sum'].max()],
    'sales_sum': [df_results_margin.loc[df_results_margin['margin_sum'].idxmin(), 'sales_sum'],
                  df_results_margin.loc[df_results_margin['margin_sum'].idxmax(), 'sales_sum']],
    'model': ['ML', 'ML'],
    'value': ['min', 'max']
})

df_min_max_dml = pd.DataFrame({
    'optimal_price': [df_results_margin.loc[df_results_margin['margin_sum_log_dml'].idxmin(), 'optimal_price_log_dml'],
                      df_results_margin.loc[df_results_margin['margin_sum_log_dml'].idxmax(), 'optimal_price_log_dml']],
    'margin_sum': [df_results_margin['margin_sum_log_dml'].min(), df_results_margin['margin_sum_log_dml'].max()],
    'sales_sum': [df_results_margin.loc[df_results_margin['margin_sum_log_dml'].idxmin(), 'sales_sum_log_dml'],
                  df_results_margin.loc[df_results_margin['margin_sum_log_dml'].idxmax(), 'sales_sum_log_dml']],

    'model': ['DML', 'DML'],
    'value': ['min', 'max']
})
df_results_margin_edges = pd.concat([df_min_max_ml, df_min_max_dml]).reset_index(drop=True)
print(df_results_margin_edges)


