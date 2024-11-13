import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from econml.dml import LinearDML
import optuna
import lightgbm as lgb
import networkx as nx

from utils_decision_intelligence import *
import seaborn as sns
from datetime import datetime

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

### 4) Draw random samples
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

### 5) Split the generated data into train and test sets
train_df, test_df = train_test_split(df_generated, test_size=0.2, random_state=42)

# Generate "ground truth" datasets for different price interventions
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

### 6) prepare variable names
treatment_name='price'
outcome_name='sales'
treatment_name_log='log_price'
outcome_name_log='log_sales'
list_cols_treatment_outcome = return_regex_cols(train_df, 'price|sales|log_price|log_sales')
X_cols = [name for name in train_df.columns.tolist() if name not in list_cols_treatment_outcome]
X_cols_no_collider = [name for name in X_cols if name != 'competitor_sales']
X_cols_price = X_cols + ['price']

### 7) estimate causal model:
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
model_y_log = lgb.LGBMRegressor(**best_params_y_log)
model_t_log = lgb.LGBMRegressor(**best_params_t_log)

# Estimate DML models
# log-log model
causal_model_log = LinearDML(model_y=model_y_log, model_t=model_t_log, fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = train_df[outcome_name_log]
T = train_df[treatment_name_log]
W = train_df[X_cols_no_collider]
causal_model_log.fit(Y=Y, T=T, X=None, W=W, cache_values=True)
# causal_model_log.effect(T0=0, T1=1, X=train_df[['year']].drop_duplicates())
print(f"calculated effect for log model: {causal_model_log.effect(T0=0, T1=1)[0]}")

## ML model
# Define the objective function for hyperparameter tuning
best_params = tune_lgbm(train_df, X_cols_price, outcome_name, n_trials)

ml_model = lgb.LGBMRegressor(**best_params)
ml_model.fit(train_df[X_cols_price], train_df[outcome_name])

### 8) Generate predictions for test dataset
# ML MODEL
test_df['ml_pred'] = ml_model.predict(test_df[X_cols_price])
# DML
test_df['dml_pred'] = predict_log_dml(causal_model_log, test_df, X_cols_no_collider)


#################################################################
##### VALIDATION 1: PERFORMANCE METRICS ON TEST DATASET
#################################################################
# Calculate mean and MAPE for ml_pred and dml_pred
mse_ml_pred = np.round(mean_squared_error(test_df['sales'], test_df['ml_pred']), 0)
mse_dml_pred = np.round(mean_squared_error(test_df['sales'], test_df['dml_pred']), 0)

mape_ml_pred = np.round(mean_absolute_percentage_error(test_df['sales'], test_df['ml_pred']), 3)
mape_dml_pred = np.round(mean_absolute_percentage_error(test_df['sales'], test_df['dml_pred']), 3)

print(f"MSE of ml_pred: {mse_ml_pred}")
print(f"MSE of dml_pred: {mse_dml_pred}")
print(f"MAPE of ml_pred: {mape_ml_pred}")
print(f"MAPE of dml_pred: {mape_dml_pred}")

plt.figure(figsize=(10, 6))
plt.scatter(test_df['sales'], test_df['ml_pred'], alpha=0.3, color='blue', label=f'ML model MSE: {mse_ml_pred}, MAPE: {mape_ml_pred}', s=15)
plt.scatter(test_df['sales'], test_df['dml_pred'], alpha=0.3, color='green', label=f'DML model MSE: {mse_dml_pred}, MAPE: {mape_dml_pred}', s=15)
plt.plot(test_df['sales'].values, test_df['sales'].values, label='perfect prediction', color='red', linestyle='--')
plt.title("Predictions vs. true values")
plt.legend()
plt.show()


### 9) Generate predictions for 3 interventions using all models:
# ML model:
test_df['price_orig'] = test_df['price']
list_test_dfs_pred = []
for price_name,price_value in dict_prices_intervention.items():
    df_test_tmp = test_df.copy()
    df_test_tmp['price'] = price_value
    df_test_tmp['intervention'] = price_name
    df_test_tmp['ml_pred'] = ml_model.predict(df_test_tmp[X_cols_price])
    list_test_dfs_pred.append(df_test_tmp)

df_test_preds = pd.concat(list_test_dfs_pred)
df_test_preds['index'] = df_test_preds.index

# DML:
df_test_preds['log_price_intervention'] = np.log(df_test_preds['price'] + 1)
df_test_preds['dml_pred'] = predict_log_dml(causal_model_log, df_test_preds, X_cols_no_collider, 'log_price_intervention')

df_test_preds[['price', 'price_orig', 'log_price', 'log_price_intervention']].head()

#################################################################
##### VALIDATION 2: PERFORMANCE METRICS ON INTERVENTIONS DATASET
#################################################################
df_test_preds_for_metrics = pd.merge(df_test_preds, df_test_interevention[['index', 'intervention', 'ground_truth_sales']], on = ['index', 'intervention'], how='inner')

for intervention in df_test_preds_for_metrics['intervention'].unique():
    df_tmp = df_test_preds_for_metrics[df_test_preds_for_metrics['intervention']==intervention]
    print(f"Intervention type: price {intervention}")
    plot_residuals(
        df_tmp,
        ground_truth_col='ground_truth_sales',
        pred_cols=['ml_pred', 'dml_pred'],
        titles=['Residuals for ML model', 'Residuals for DML model']
    )

# Calculate metrics:
df_metrics = calculate_grouped_metrics_ml_dml(df_test_preds_for_metrics)
print(df_metrics)

plot_metrics_ml_dml(df_metrics)

df_metrics['ml_test_mse'] = mse_ml_pred
df_metrics['dml_test_mse'] = mse_dml_pred
df_metrics['ml_test_mape'] = mape_ml_pred
df_metrics['dml_test_mape'] = mape_dml_pred

plot_metrics_ml_dml(df_metrics, test_data=True)

######################################################
############ Decision intelligence module ############
######################################################


n_trials = 30
list_sku_min_price_cost_share = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8] # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
list_prices = sorted(test_df['price'].unique().tolist())
n_iters = 30

experiment_parameters = f"n_{n_trials}_prices_{'_'.join(map(str, list_sku_min_price_cost_share))}_iters_{n_iters}"
print(experiment_parameters)


list_decision_intelligence_results = []
for sku_min_price_cost_share in list_sku_min_price_cost_share:
    list_prices_ml = []
    list_margins_ml = []
    list_sales_ml = []
    list_prices_log_dml = []
    list_margins_log_dml = []
    list_sales_log_dml = []

    print(f"""#################################/n
          #################################/n
          #################################/n
          price cost share: {sku_min_price_cost_share}/n
          #################################/n
          #################################/n
          #################################/n""")
    i = 1
    while i < (n_iters + 1):
        # generate data
        df_synth = gcm.draw_samples(scm_data_generator, num_samples=5000)
        train_df, test_df = train_test_split(df_generated, test_size=0.2)

        # Fit models
        # DML
        objective_t_log = create_objective_function_DML(train_df, treatment_name_log, X_cols_no_collider)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_t_log, n_trials=n_trials)
        best_params_t_log = study.best_params

        objective_y_log = create_objective_function_DML(train_df, outcome_name_log, X_cols_no_collider)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_y_log, n_trials=n_trials)
        best_params_y_log = study.best_params

        model_y_log = lgb.LGBMRegressor(**best_params_y_log)
        model_t_log = lgb.LGBMRegressor(**best_params_t_log)

        causal_model_log = LinearDML(model_y=model_y_log, model_t=model_t_log, fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
        Y = train_df[outcome_name_log]
        T = train_df[treatment_name_log]
        W = train_df[X_cols_no_collider]
        causal_model_log.fit(Y=Y, T=T, X=None, W=W, cache_values=False)

        # ML
        best_params = tune_lgbm(train_df, X_cols_price, outcome_name, n_trials)

        ml_model = lgb.LGBMRegressor(**best_params)
        ml_model.fit(train_df[X_cols_price], train_df[outcome_name])

        # split test set for policy optimization and validation
        df_test_for_price_optimization = test_df.sample(frac=0.5)
        df_test_for_optimal_price_application = test_df.drop(df_test_for_price_optimization.index)  
        sku_unit_cost = np.round((np.min(list_prices) - 6) * sku_min_price_cost_share, 2) # consultant patch
        # ML model:
        optimal_price = get_optimal_price_ml(df_test_for_price_optimization, ml_model, X_cols_price, sku_unit_cost, list_prices)
        margin_sum, sales_sum = apply_optimal_price(df_test_for_optimal_price_application, optimal_price, sku_unit_cost, scm_data_generator)
        list_prices_ml.append(optimal_price)
        list_margins_ml.append(margin_sum)
        list_sales_ml.append(sales_sum)
        # log DML model:
        optimal_price = get_optimal_price_log_dml(df_test_for_price_optimization, causal_model_log, sku_unit_cost, X_cols_no_collider, list_prices)
        margin_sum, sales_sum = apply_optimal_price(df_test_for_optimal_price_application, optimal_price, sku_unit_cost, scm_data_generator)
        list_prices_log_dml.append(optimal_price)
        list_margins_log_dml.append(margin_sum)
        list_sales_log_dml.append(sales_sum)
        i+=1

    df_results = pd.DataFrame({
        'sku_min_price_cost_share': [sku_min_price_cost_share] * len(list_prices_ml),
        'optimal_price': list_prices_ml, 
        'margin_sum': list_margins_ml,
        'sales_sum': list_sales_ml,
        'optimal_price_log_dml': list_prices_log_dml, 
        'margin_sum_log_dml': list_margins_log_dml,
        'sales_sum_log_dml': list_sales_log_dml})
    
    list_decision_intelligence_results.append(df_results)

df_results_all = pd.concat(list_decision_intelligence_results)

#################################################################
##### VALIDATION 3: BUSINESS PERFORMANCE
#################################################################

df_results_all.groupby('sku_min_price_cost_share').size()

for val in df_results_all['sku_min_price_cost_share'].unique():
    df_tmp = df_results_all[df_results_all['sku_min_price_cost_share'] == val]
    print("Results for sku_min_price_cost_share: ", val)
    print_decision_intelligence_results(df_tmp, n_bins=n_iters)

# Calculate minimum and maximum optimal_price_log_dml and optimal_price for every value of 'sku_min_price_cost_share'
min_max_prices = df_results_all.groupby('sku_min_price_cost_share').agg({
    'optimal_price_log_dml': ['min', 'max'],
    'optimal_price': ['min', 'max']
}).reset_index()

min_max_prices.columns = ['sku_min_price_cost_share', 'min_optimal_price_log_dml', 'max_optimal_price_log_dml', 'min_optimal_price', 'max_optimal_price']
# print(min_max_prices)

df_margin_comparison_results = np.round(df_results_all.groupby('sku_min_price_cost_share')[['margin_sum', 'margin_sum_log_dml']].mean() / 10000, 0)
df_margin_comparison_results['prct_diff'] = np.round((df_margin_comparison_results['margin_sum_log_dml'] - df_margin_comparison_results['margin_sum']) / df_margin_comparison_results['margin_sum'] * 100, 2)
df_margin_comparison_results

if True:
    df_results_all.to_csv(f"{PATH_DATA}/df_results_all_{experiment_parameters}.csv", index=False)
    df_margin_comparison_results.to_csv(f"{PATH_DATA}/df_margin_comparison_results_{experiment_parameters}.csv", index=False)



#################################################################
##### DEBUGGING DECISION INTELLIGENCE
#################################################################
if False:
    val = 0.3
    df_tmp = df_results_all[df_results_all['sku_min_price_cost_share'] == val]
    df_tmp[['margin_sum', 'margin_sum_log_dml']] = (df_tmp[['margin_sum', 'margin_sum_log_dml']] / 10000).round().astype(int)
    df_tmp[['sales_sum', 'sales_sum_log_dml']] = (df_tmp[['sales_sum', 'sales_sum_log_dml']] / 1000).round().astype(int)
    df_tmp['margin_sum'] = (df_tmp['margin_sum'] / 100).round() * 100
    df_tmp['margin_sum_log_dml'] = (df_tmp['margin_sum_log_dml'] / 100).round() * 100

    print("Results for sku_min_price_cost_share: ", val)
    print_decision_intelligence_results(df_tmp)
    2+2
    plt.hist(df_tmp['margin_sum'], bins=30, alpha=0.5, label='ML model')
    plt.hist(df_tmp['margin_sum_log_dml'], bins=30, alpha=0.5, label='ML model')
    plt.show()

    # Generate overlying histograms for 'margin_sum' and 'margin_sum_log_dml'
    sns.histplot(df_tmp['margin_sum'], kde=False, color='blue', label='ML model', alpha=0.5)
    sns.histplot(df_tmp['margin_sum_log_dml'], kde=False, color='green', label='DML model', alpha=0.5)
    plt.legend()
    plt.title('Overlying Histograms for Margin Sum')
    plt.xlabel('Margin Sum')
    plt.ylabel('Frequency')
    plt.show()


# Generate histograms of true margin mass for SCM data for different prices
if False:
    cost_price_share = 0.8
    sku_unit_cost = np.round((np.min(list_prices) - 6) * cost_price_share, 2)
    list_prices_tmp = []
    list_margins_tmp = []
    list_sales_tmp = []
    i = 0
    while i < 31:
        for price_tmp in list_prices:
            intervention_price_tmp = {'price': lambda price: price_tmp}
            test_df = gcm.draw_samples(scm_data_generator, num_samples=2000)
            df_price_opt = test_df.sample(frac=0.5)
            df_price_application = test_df.drop(df_price_opt.index)
            df_price_application['price'] = price_tmp
            df_price_application['unit_cost'] = sku_unit_cost
            df_price_application['sales'] = generate_test_dataset(scm_data_generator, intervention_price_tmp, df_test_for_optimal_price_application, verbose=False)['sales']
            df_price_application['unit_margin'] = (df_price_application['price'] - df_price_application['unit_cost']) * df_price_application['sales']
            list_prices_tmp.append(price_tmp)
            list_margins_tmp.append((df_price_application['unit_margin'].sum()))
            list_sales_tmp.append((df_price_application['sales'].sum()))
        i+=1

    test_df[['price', 'sales']].describe()

    df_res_experiment = pd.DataFrame({'price': list_prices_tmp, 
                                    'margin_sum': list_margins_tmp,
                                    'sales_sum': list_sales_tmp})
    df_res_experiment.groupby('price')['margin_sum'].mean().plot(kind='bar')
    df_res_experiment.groupby('price')['sales_sum'].mean().plot(kind='bar')

    for price_point in list_prices:
        df_tmp = df_res_experiment[df_res_experiment['price'] == price_point]
        plt.hist(df_tmp['margin_sum'], bins=30, alpha=0.5, label=f'price: {price_point}')
        plt.legend()
        plt.show()

    for price_point in list_prices:
        df_tmp = df_res_experiment[df_res_experiment['price'] == price_point]
        plt.hist(df_tmp['sales_sum'], bins=30, alpha=0.5, label=f'price: {price_point}')
        plt.legend()
        plt.show()

