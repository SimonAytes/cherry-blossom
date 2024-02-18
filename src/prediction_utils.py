import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor

scaler = StandardScaler()

# Path to the model to load
model_dir_path = "./models/GBR-20240218-01/"

# Load the trained model
model = joblib.load(f'{model_dir_path}model.pkl')

# Load the normalization parameters
normalization_params = joblib.load(f'{model_dir_path}normalization_params.pkl')

# Load the feature names
feature_list = joblib.load(f'{model_dir_path}feature_list.pkl')

# Best parameters chosen via Grid-Search HPO
best_params = {
    'alpha': 0.9,
    'ccp_alpha': 0.0,
    'criterion': 'friedman_mse',
    'init': None,
    'learning_rate': 0.2,
    'loss': 'squared_error',
    'max_depth': 5,
    'max_features': 'sqrt',
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'min_weight_fraction_leaf': 0.0,
    'n_estimators': 50,
    'n_iter_no_change': None,
    'random_state': 42,
    'subsample': 0.8,
    'tol': 0.0001,
    'validation_fraction': 0.1,
    'verbose': 0,
    'warm_start': False
}

# Chosen via a 90% threshold analysis of feature importance
selected_features = ['day_length_07', 'day_length_10', 'day_length_05', 'day_length_11',
       'tempmax_05', 'windspeedmin_07', 'windspeedmin_09', 'humidity_07',
       'day_length_06', 'day_length_02', 'tempmax_06', 'day_length_09',
       'tempmax_07', 'tempmax_02', 'windspeed_04', 'snowdepth_01', 'dew_07',
       'day_length_04', 'tempmax_08', 'day_length_12', 'humidity_05',
       'windspeedmin_10', 'cloudcover_08', 'windspeed_09', 'windspeedmean_06',
       'cloudcover_11', 'solarenergy_02', 'windspeedmax_11', 'uvindex_05',
       'day_length_01', 'solarenergy_03', 'cloudcover_09', 'dew_04',
       'humidity_09']

def normalize_new_data(new_data, normalization_params):
    # Ensure the input data is a numpy array
    if isinstance(new_data, np.ndarray):
        # Normalize each feature using the stored mean and standard deviation
        normalized_data = (new_data - normalization_params['mean']) / normalization_params['std']
        return normalized_data
    elif isinstance(new_data, pd.DataFrame):
        # Normalize each column (feature) using the stored mean and standard deviation
        normalized_data = (new_data - normalization_params['mean']) / normalization_params['std']
        return normalized_data
    else:
        raise ValueError("Input data must be a numpy array or pandas DataFrame")


def prediction_interval(x, model, alpha=0.05):
    # Get the mean prediction
    y_pred = model.predict(x)

    # Calculate the standard deviation of predictions
    std_dev = np.std(y_pred)

    # Calculate the z-score for the given significance level
    z_score = np.abs(norm.ppf(alpha / 2))

    # Calculate margin of error
    margin_of_error = z_score * std_dev

    # Calculate prediction interval
    lower_bound = y_pred - margin_of_error
    upper_bound = y_pred + margin_of_error

    # Convert predictions to integers
    lower_bound = np.round(lower_bound).astype(int)
    upper_bound = np.round(upper_bound).astype(int)

    return lower_bound, upper_bound


def get_final_predictions(x, model):
    # Make predictions
    y_pred = model.predict(x)

    # Convert predictions to integers
    predictions = np.round(y_pred).astype(int)

    lower_bound, upper_bound = prediction_interval(x, model, alpha=0.75)

    return list(predictions), list(lower_bound), list(upper_bound)

def get_interval(y_pred, alpha=0.99):
    # Calculate prediction intervals using quantiles
    quantile_lower = (1 - alpha) / 2
    quantile_upper = 1 - quantile_lower

    # Assuming y_pred is a 1-dimensional array
    sorted_indices = np.argsort(y_pred)
    n = len(y_pred)

    # Calculate lower and upper bounds for prediction intervals
    lower_index = int(np.floor(quantile_lower * (n - 1)))
    upper_index = int(np.ceil(quantile_upper * (n - 1)))

    lower_bound = y_pred[sorted_indices[lower_index]]
    upper_bound = y_pred[sorted_indices[upper_index]]

    # Output the results in the desired format
    return pd.DataFrame({'Prediction': y_pred, 'Lower': lower_bound, 'Upper': upper_bound})


def evaluate_model(y_test, y_pred_int):
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred_int)
    r2 = r2_score(y_test, y_pred_int)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


def get_predictions(x, model):
    # Make predictions
    y_pred = model.predict(x)

    # Convert predictions to integers
    y_pred_int = np.round(y_pred).astype(int)

    return y_pred_int


def compute_normalization_params(training_data):
    """
    Compute normalization parameters (mean and standard deviation) from training data.

    Parameters:
        training_data (numpy array or pandas DataFrame): Training data used for computing normalization parameters.

    Returns:
        dict: Dictionary containing normalization parameters (e.g., 'mean' and 'std' for each feature).
    """
    # Compute mean and standard deviation for each feature
    mean = np.mean(training_data, axis=0)
    std = np.std(training_data, axis=0)

    # Store the normalization parameters in a dictionary
    normalization_params = {'mean': mean, 'std': std}

    return normalization_params