import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import dowhy
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

# PATH_DATA = "C:/Users/lukasz.frydrych/OneDrive - Lingaro Sp. z o. o/Desktop\projects/202410_DSSummit/data"

def return_regex_cols(df, rgx):
    """returns list of columns"""
    comp = re.compile(rgx)
    cols = list(filter(comp.match, df.columns))
    return cols


def generate_test_dataset(scm_data_gen, intervention_dict, df_test, verbose=True):
    df_test_c = dowhy.gcm.whatif.interventional_samples(scm_data_gen, intervention_dict, df_test)
    if verbose:
        print(f"Average sales: {df_test_c['sales'].mean()} for price: {df_test_c['price'].mean()}")
    df_test_c['log_price'] = np.log(df_test_c['price'] + 1)  # Adding 1 to avoid log(0)
    df_test_c['log_sales'] = np.log(df_test_c['sales'] + 1)  # Adding 1 to avoid log(0)
    if verbose:
        plt.scatter(df_test_c['price'], df_test_c['sales'], color='blue')
        plt.show()
    return df_test_c


def change_store_type_type(df):
    df_c = df.copy()
    df_c['store_type'] = df_c['store_type'].astype('category').cat.codes + 1
    df_c['store_type'] = df_c['store_type'].astype(int)
    return df_c


def tune_lgbm(df_train, x_cols, y_col, n_trials=15):
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
            'num_boost_round': trial.suggest_int('num_boost_round', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
        }
        
        lgb_train = lgb.Dataset(df_train[x_cols], df_train[y_col])
        cv_results = lgb.cv(param, lgb_train, nfold=3, metrics='rmse', seed=42, stratified=False)
        
        return np.mean(cv_results['valid rmse-mean'])
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    return best_params


def calculate_grouped_metrics_ml_dml(df):
    metrics = df.groupby(['intervention', 'price']).apply(lambda group: pd.Series({
        'ml_pred_mape': mean_absolute_percentage_error(group['ground_truth_sales'], group['ml_pred']),
        'ml_pred_mse': mean_squared_error(group['ground_truth_sales'], group['ml_pred']),
        'dml_pred_mape': mean_absolute_percentage_error(group['ground_truth_sales'], group['dml_pred']),
        'dml_pred_mse': mean_squared_error(group['ground_truth_sales'], group['dml_pred'])
    }))
    metrics.reset_index(inplace=True)
    metrics['price_intervention'] = metrics['price'].astype(str) + ": " + metrics['intervention']
    metrics.drop(columns=['intervention', 'price'], inplace=True)
    metrics.sort_values(by='price_intervention', inplace=True)
    return metrics


def create_objective_function_DML(dataset, y_col, X_cols):
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
            'num_boost_round': trial.suggest_int('num_boost_round', 50, 500),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
        }
        
        lgb_train = lgb.Dataset(dataset[X_cols], dataset[y_col])
        cv_results = lgb.cv(param, lgb_train, nfold=3, metrics='rmse', seed=42, stratified=False)
        
        return np.mean(cv_results['valid rmse-mean'])
    return objective


def plot_metrics_ml_dml(df_metrics, test_data=False):
    def plot_metric(metric):
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['price_intervention'].values, df_metrics[f'ml_pred_{metric}'].values, color='blue', label=f'ML Model {str.upper(metric)}')
        plt.plot(df_metrics['price_intervention'].values, df_metrics[f'dml_pred_{metric}'].values, color='green', label=f'DML Model {str.upper(metric)}')
        if test_data:
            plt.plot(df_metrics['price_intervention'].values, df_metrics[f'ml_test_{metric}'].values, color='blue', label=f'ML Model on Test {str.upper(metric)}', linestyle='--')
            plt.plot(df_metrics['price_intervention'].values, df_metrics[f'dml_test_{metric}'].values, color='green', label=f'DML Model on Test {str.upper(metric)}', linestyle='--')
        plt.xlabel('Price Intervention')
        plt.ylabel(str.upper(metric))
        plt.title(f'{str.upper(metric)} for Different Models by Price Intervention')
        plt.legend()
        plt.grid(True)
        plt.show()
    plot_metric('mse')
    plot_metric('mape')


def plot_residuals(df, ground_truth_col, pred_cols, titles):
    fig, axes = plt.subplots(1, len(pred_cols), figsize=(18, 6))
    
    for i, pred_col in enumerate(pred_cols):
        axes[i].scatter(df[ground_truth_col], df[ground_truth_col] - df[pred_col])
        axes[i].axhline(y=0, color='r', linestyle='--')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Ground Truth Sales')
        axes[i].set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()

    
def predict_log_dml(model, df_pred, X_cols, log_price_baseline_col='log_price'):
    df_tmp = df_pred.copy()
    df_tmp['dml_log_price_pred'] = np.mean([model_t.predict(df_tmp[X_cols]) for model_t in model.models_t[0]], axis=0)
    df_tmp['dml_log_sales_pred'] = np.mean([model_y.predict(df_tmp[X_cols]) for model_y in model.models_y[0]], axis=0)
    df_tmp['dml_pred_log'] = df_tmp['dml_log_sales_pred'] + model.effect(T0=df_tmp['dml_log_price_pred'], T1=df_tmp[log_price_baseline_col])
    return np.exp(df_tmp['dml_pred_log']) - 1


def get_optimal_price_ml(df, ml_model, X_cols_price, sku_unit_cost, list_prices):
    df_tmp = df.copy()
    dict_margins = {}
    for price_point in list_prices:
        df_tmp['price'] = price_point
        df_tmp['unit_cost'] = sku_unit_cost
        df_tmp['predicted_sales'] = ml_model.predict(df_tmp[X_cols_price])
        df_tmp['unit_margin'] = (df_tmp['price'] - df_tmp['unit_cost']) * df_tmp['predicted_sales']
        dict_margins[price_point] = df_tmp['unit_margin'].sum()
    optimal_price = max(dict_margins, key=dict_margins.get)
    return optimal_price


def get_optimal_price_log_dml(df, causal_model_log, sku_unit_cost, X_cols, list_prices):
    df_tmp = df.copy()
    dict_margins = {}
    for price_point in list_prices:
        df_tmp['price'] = price_point
        df_tmp['unit_cost'] = sku_unit_cost
        df_tmp['log_price_intervention'] = np.log(df_tmp['price'] + 1)
        df_tmp['dml_log_pred_log'] = df_tmp['log_sales'] + causal_model_log.effect(T0=df_tmp['log_price'], T1=df_tmp['log_price_intervention'])
        df_tmp['predicted_sales'] = predict_log_dml(causal_model_log, df_tmp, X_cols, 'log_price_intervention')
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


def print_decision_intelligence_results(df, n_bins=30):
    df_results_margin = df.copy()

    df_results_margin[['margin_sum', 'margin_sum_log_dml']] = (df_results_margin[['margin_sum', 'margin_sum_log_dml']] / 10000).round().astype(int)
    df_results_margin[['sales_sum', 'sales_sum_log_dml']] = (df_results_margin[['sales_sum', 'sales_sum_log_dml']] / 1000).round().astype(int)

    plt.hist(df_results_margin['margin_sum'], bins=n_bins, alpha=0.5, label='ML Model')
    plt.hist(df_results_margin['margin_sum_log_dml'], bins=n_bins, alpha=0.5, label='Log DML Model')
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
