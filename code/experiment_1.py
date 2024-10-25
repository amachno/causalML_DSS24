## Generate syntethic data based on actual data from Kaggle
import numpy as np
import pandas as pd

import dowhy
from dowhy import CausalModel
from dowhy.utils import plot
from dowhy import gcm

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import networkx as nx

from utils import *

### 1) Import data
PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop/projects/202410_DSSummit/repo/causalML_DSS24/data"

df_week = pd.read_csv(f"{PATH_DATA}/df_week_top_sku.csv")


### 7) Create causal graph
df_week.columns

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
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(scm, df_week, override_models=True, quality=gcm.auto.AssignmentQuality.GOOD)
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

########################################################
### Experiment with other types of interventions:
var_intervened = 'store_size'
intervention = {var_intervened: lambda var_intervened: 0} # try different values. Low variance estimator
df_intervention = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[[var_intervened, 'sales']].mean(), df_intervention[[var_intervened, 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})

# check for seasonality
df_week.groupby(['month'])['sales'].mean().plot()
df_week[['month', 'sin_month', 'cos_month']].drop_duplicates().round(2)
df_generated.groupby(['sin_month'])['sales'].mean().plot()
var_intervened = 'sin_month'
intervention = {var_intervened: lambda var_intervened: 0.5}
df_intervention = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[[var_intervened, 'sales']].mean(), df_intervention[[var_intervened, 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})
# Seasonality works pretty well

var_intervened = 'n_days_sales'
intervention = {var_intervened: lambda var_intervened: 0} 
df_intervention = dowhy.gcm.whatif.interventional_samples(scm, intervention, df_generated)
# compare intervention with generated data
pd.concat([df_generated[[var_intervened, 'sales']].mean(), df_intervention[[var_intervened, 'sales']].mean()], axis=1).rename(columns={0: 'generated', 1: 'intervention'})


# Results look nice.
# Note that if line below is used:
# scm.set_causal_mechanism('sales', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
# then model can extrapolate well
# If it is not used, then 'Discrete AdditiveNoiseModel using HistGradientBoostingRegressor' is assigned to sales node.
# Then, model cannot extrapolate.



########################################################
### Experiment with causal model:
model = CausalModel(
    data=df_generated,
    treatment='price',
    outcome='sales',
    # common_causes=df_generated['common_causes_names'],
    # instruments=df_generated['instrument_names']
    graph=causal_graph
)

# Identify the causal effect
identified_estimand = model.identify_effect()
print(identified_estimand)

# Estimate the causal effect using Double ML
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

len(estimate.params['cate_estimates'])

df_generated['cate'] = estimate.params['cate_estimates']
# it calculates heterogeneous treatment effects for different values of n_days_sales (???)

df_generated.iloc[:20]

# estimate.control_value
# estimate.treatment_value
# estimate.realized_estimand_expr
# estimate.target_estimand
# estimate.effect_strength
# estimate.value

########################################################
### Experiment with log-log causal model:
df_generated_logs = df_generated.copy()
df_generated_logs['sales'] = df_generated_logs['sales'].apply(lambda x: max(x, 0))
df_generated_logs['log_sales'] = np.log(df_generated_logs['sales']+1)
df_generated_logs['log_price'] = np.log(df_generated_logs['price']+1)
df_generated_logs.describe()

causal_graph_logs = nx.DiGraph([('week_num', 'log_sales'),
                                ('week_num', 'log_price'),
                                ('log_price', 'log_sales'),
                                ('n_cities', 'log_sales'),
                                ('n_cities', 'log_price'),
                                ('store_size', 'log_sales'),
                                ('store_size', 'log_price'),
                                ('storetype_id', 'log_sales'),
                                ('storetype_id', 'log_price'),
                                ('sin_month', 'log_price'),
                                ('sin_month', 'log_sales'),
                                ('cos_month', 'log_price'),
                                ('cos_month', 'log_sales'),
                                ('n_days_sales', 'log_sales'),
                                ])

model = CausalModel(
    data=df_generated_logs,
    treatment='log_price',
    outcome='log_sales',
    # common_causes=df_generated_logs['common_causes_names'],
    # instruments=df_generated_logs['instrument_names']
    graph=causal_graph_logs
)

# Identify the causal effect
identified_estimand = model.identify_effect()
print(identified_estimand)

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
# there is HUGE variance of ATE estimate when using GradientBoostingRegressor

len(estimate.params['cate_estimates'])

df_generated['cate'] = estimate.params['cate_estimates']
# it calculates heterogeneous treatment effects for different values of n_days_sales (???)

df_generated.iloc[:20]