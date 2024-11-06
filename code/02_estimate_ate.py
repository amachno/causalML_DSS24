## Generate syntethic data based on actual data from Kaggle
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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
scm_data_generator.set_causal_mechanism('price', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

# Automatically assign generative models to each node based on the given data
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
    
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm_data_generator, df, override_models=True, quality=gcm.auto.AssignmentQuality.BETTER)
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

### 6) Estimate price elasticity om generated data using various methods and APIs:
causal_graph_logs = nx.DiGraph([('holiday_days', 'log_sales'),
                           ('holiday_days', 'log_price'),
                           ('is_lockdown', 'log_sales'),
                           ('curfew', 'log_sales'),
                           ('year_2021', 'log_sales'),
                           ('year_2022', 'log_sales'),
                           ('year_2023', 'log_sales'),
                           ('month_2', 'log_sales'),
                           ('month_3', 'log_sales'),
                           ('month_4', 'log_sales'),
                           ('month_5', 'log_sales'),
                           ('month_6', 'log_sales'),
                           ('month_7', 'log_sales'),
                           ('month_8', 'log_sales'),
                           ('month_9', 'log_sales'),
                           ('month_10', 'log_sales'),
                           ('month_11', 'log_sales'),
                           ('month_12', 'log_sales'),
                           ('store_type', 'log_price'),
                           ('store_type', 'log_sales'),
                           ('week_number_trend', 'log_price'),
                           ('week_number_trend', 'log_sales'),
                           ('log_price', 'log_sales')
                           ])

### 6a) Try model specification using graph
model = CausalModel(
    data=df_generated,
    treatment='log_price',
    outcome='log_sales',
    graph=causal_graph_logs
)
list_model_attributes = []
for attribute in dir(model):
    if not attribute.startswith('__'):
        list_model_attributes.append(attribute)
list_model_attributes
model.get_common_causes()
model.get_effect_modifiers()
model.summary()
model.view_model()
# this means that heterogenous treatment effect will be calculated for all months and years separately
# this means that within graph every variable that is linked only with outcome will be treated as effect modifier
# that is bad


# Identify the causal effect
identified_estimand = model.identify_effect()
print(identified_estimand)
# List all attributes of identified_estimand
list_estimand_attributes = []
for attribute in dir(identified_estimand):
    if not attribute.startswith('__'):
        list_estimand_attributes.append(attribute)
list_estimand_attributes

# Estimate the causal effect using Double ML
estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.econml.dml.LinearDML",
                                 control_value = 0,
                                 treatment_value = 1,
                                 confidence_intervals=False,
                                 method_params={
                                    "init_params":{'model_y': LinearRegression(), # GradientBoostingRegressor
                                                   'model_t': LinearRegression() # GradientBoostingRegressor
                                    }})
print(estimate)

estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.econml.dml.LinearDML",
                                 control_value = 0,
                                 treatment_value = 1,
                                 confidence_intervals=False,
                                 method_params={
                                    "init_params":{'model_y': GradientBoostingRegressor(), 
                                                   'model_t': GradientBoostingRegressor() 
                                    }})
print(estimate)
# very low price elasticity coefficient

### 6b) Try manual specification
model = CausalModel(
    data=df_generated,
    treatment=treatment_name,
    outcome=outcome_name,
    common_causes=X_cols,
)
identified_estimand = model.identify_effect()
print(identified_estimand)
estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.econml.dml.LinearDML",
                                 control_value = 0,
                                 treatment_value = 1,
                                 confidence_intervals=False,
                                 method_params={
                                    "init_params":{'model_y': LinearRegression(), 
                                                   'model_t': LinearRegression() 
                                    }})
print(estimate)

estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.econml.dml.LinearDML",
                                 control_value = 0,
                                 treatment_value = 1,
                                 confidence_intervals=False,
                                 method_params={
                                    "init_params":{'model_y': GradientBoostingRegressor(), 
                                                   'model_t': GradientBoostingRegressor() 
                                    }})
print(estimate)
# very low price elasticity coefficient

### 6b) Try manual base econml
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

### 6c) Try manual base econml on original data
est = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), fit_cate_intercept=True, linear_first_stages=False, featurizer=None, cv=3, random_state=123)
Y = df[outcome_name]
T = df[treatment_name]
df['store_type_numeric'] = df['store_type'].astype('category').cat.codes + 1
X_cols_store_type_numeric = X_cols + ['store_type_numeric']
X_cols_store_type_numeric.remove('store_type')
W = df[X_cols_store_type_numeric]
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

2+2

### Conclusion: it seems there are significant differences between original data and generated data
# when it comes to calculating treatment effect