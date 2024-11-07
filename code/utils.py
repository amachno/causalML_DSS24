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

def plot_sales_timeseries(df, store_id, product_id):
    # Filter the dataframe for the given store_id and product_id
    df_filtered = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)]
    
    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print(f"No data available for store_id {store_id} and product_id {product_id}")
        return
    
    # Plot the timeseries
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered['date'].values, df_filtered['sales'].values, marker='o', linestyle='-')
    plt.title(f'Sales Time Series for Store {store_id} and Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.show()

#### check if prices vary for the same SKU vary at the same time (across different stores)
def plot_min_max_price_over_time(df, product_id, price_col='price'):
    # Filter the dataframe for the given product_id
    df_filtered = df[df['product_id'] == product_id]
    
    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print(f"No data available for product_id {product_id}")
        return

    # Calculate minimum and maximum price for every date   
    min_max_price_per_date = df_filtered.groupby('date')[price_col].agg(['min', 'max']).reset_index().sort_values('date')

    plt.scatter(min_max_price_per_date['date'].values, min_max_price_per_date['min'].values, color='green', label='Min Price', alpha=0.1, s=5)
    plt.scatter(min_max_price_per_date['date'].values, min_max_price_per_date['max'].values, color='red', label='Max Price', alpha=0.1, s=5)
    
    plt.title(f'Min and Max Price Over Time for Product {product_id}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def calc_prc_rows_by_combination(df, list_cols):
    combination_counts = df.groupby(list_cols, dropna=False).size()
    total_rows = len(df)
    percentage_combination = (combination_counts / total_rows) * 100
    percentage_combination = percentage_combination.reset_index(name='percentage')
    return percentage_combination


def plot_sales_and_price_timeseries(df, store_id, product_id, price_col='price'):
    # Filter the dataframe for the given store_id and product_id
    df_filtered = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)]
    
    # Check if the filtered dataframe is empty
    if df_filtered.empty:
        print(f"No data available for store_id {store_id} and product_id {product_id}")
        return
    
    # Plot the timeseries
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.scatter(df_filtered['date'].values, df_filtered['sales'].values, marker='o', linestyle='-', label='Sales')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales')
    ax1.set_title(f'Sales and Price Time Series for Store {store_id} and Product {product_id}')

    # Create a secondary y-axis for the price
    ax2 = ax1.twinx()
    ax2.plot(df_filtered['date'].values, df_filtered[price_col].values, color='red', label=price_col)
    ax2.set_ylabel(price_col)

    # Set date format on x-axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

def remove_price_outliers_groupby(df, groupby_cols, price_col, threshold=1.5):
    """
    Remove price outliers from a DataFrame by grouping based on specified columns.
    This function removes outliers from the price column of a DataFrame by grouping
    the data based on the specified columns and applying the IQR (Interquartile Range) method.
    Outliers are defined as values that lie outside the range [Q1 - threshold * IQR, Q3 + threshold * IQR].
    """
    def remove_outliers(group):
        Q1 = group[price_col].quantile(0.25)
        Q3 = group[price_col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR < group[price_col].mean() * 0.1:
            return group
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return group[(group[price_col] >= lower_bound) & (group[price_col] <= upper_bound)]
    
    df_filtered = df.groupby(groupby_cols, group_keys=False).apply(remove_outliers)
    print(f"Removed {len(df) - len(df_filtered)} price outliers")
    return df_filtered


def remove_sales_outliers_groupby(df, groupby_cols, sales_col='sales', threshold=1.5):
    """
    Removes sales outliers from a DataFrame based on a specified threshold and grouping columns.
    This function groups the DataFrame by the specified columns and removes sales outliers
    within each group. Outliers are defined as sales values greater than the 95th percentile
    plus a multiple of the interquartile range (IQR), excluding rows where sales_col=0.
    """
    def remove_outliers(group):
        group_stats_calc = group[group[sales_col] > 0]
        Q95 = group_stats_calc[sales_col].quantile(0.95)
        Q1 = group_stats_calc[sales_col].quantile(0.25)
        Q3 = group_stats_calc[sales_col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q95 + threshold * IQR
        return group[group[sales_col] <= upper_bound]
    
    df_filtered = df.groupby(groupby_cols, group_keys=False).apply(remove_outliers)
    print(f"Removed {len(df) - len(df_filtered)} sales outliers")
    return df_filtered

def summarize_and_rank(df, grp_cols, get_elast_coefs=False):

    df_summary = df.groupby(grp_cols).agg(
        avg_sales=('sales', 'mean'),
        std_sales=('sales', 'std'),
        std_unit_price=('unit_price', 'std')
    ).reset_index()

    df_non_zero_sales = df[df['sales'] > 0].groupby(grp_cols).size().reset_index(name='non_zero_sales_count')
    df_summary = pd.merge(df_summary, df_non_zero_sales, on=grp_cols, how='left')

    if get_elast_coefs:
        coefficients = {}
        n_skus = len(df['product_id'].unique())
        for i, product_id in enumerate(df['product_id'].unique()):
            if i % (n_skus // 10) == 0:
                print(f"Processing {i}/{n_skus} SKUs ({(i / n_skus) * 100:.1f}%)")
            subset = df[df['product_id'] == product_id]
            if not subset.empty:
                model = LinearRegression().fit(subset[['log_price', 'month', 'year', 'store_size', 'trend']], subset['log_sales'])
                coefficients[product_id] = model.coef_[0]

        df_coefs = pd.DataFrame(list(coefficients.items()), columns=['product_id', 'coefficient']).sort_values(by='coefficient', ascending=True)
        df_coefs['coefficient'] = -df_coefs['coefficient']

        df_summary = pd.merge(df_summary, df_coefs, on=grp_cols, how='left')
        # Standardize all values in df_summary to have values between 0 and 1
        scaler = MinMaxScaler()
        df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count', 'coefficient']] = scaler.fit_transform(
            df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count', 'coefficient']]
        )
        df_summary['ranking'] = df_summary['avg_sales'] * df_summary['std_sales'] * df_summary['std_unit_price'] * df_summary['non_zero_sales_count'] * df_summary['coefficient']
        df_summary.sort_values(by=['ranking'], ascending=False, inplace=True)

        print("Returning dataframe with coefficients")
        return df_summary

    # Standardize all values in df_summary to have values between 0 and 1
    scaler = MinMaxScaler()
    df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count']] = scaler.fit_transform(
        df_summary[['avg_sales', 'std_sales', 'std_unit_price', 'non_zero_sales_count']]
    )
    df_summary['ranking'] = df_summary['avg_sales'] * df_summary['std_sales'] * df_summary['std_unit_price'] * df_summary['non_zero_sales_count']
    df_summary.sort_values(by=['ranking'], ascending=False, inplace=True)
    
    print("Returning dataframe without coefficients")
    return df_summary


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


def calculate_grouped_metrics(df):
    metrics = df.groupby(['intervention', 'price']).apply(lambda group: pd.Series({
        'ml_pred_mape': mean_absolute_percentage_error(group['ground_truth_sales'], group['ml_pred']),
        'ml_pred_mse': mean_squared_error(group['ground_truth_sales'], group['ml_pred']),
        'dml_lin_pred_mape': mean_absolute_percentage_error(group['ground_truth_sales'], group['dml_lin_pred']),
        'dml_lin_pred_mse': mean_squared_error(group['ground_truth_sales'], group['dml_lin_pred']),
        'dml_log_pred_mape': mean_absolute_percentage_error(group['ground_truth_sales'], group['dml_log_pred']),
        'dml_log_pred_mse': mean_squared_error(group['ground_truth_sales'], group['dml_log_pred'])
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

def plot_metrics(df_metrics):
    def plot_metric(metric):
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics['price_intervention'].values, df_metrics[f'ml_pred_{metric}'].values, label=f'ML Model {str.upper(metric)}')
        plt.plot(df_metrics['price_intervention'].values, df_metrics[f'dml_lin_pred_{metric}'].values, label=f'Linear DML {str.upper(metric)}')
        plt.plot(df_metrics['price_intervention'].values, df_metrics[f'dml_log_pred_{metric}'].values, label=f'Log-Log DML {str.upper(metric)}')
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

    
