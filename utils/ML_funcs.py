'''
File: ML_funcs.py
File Created: 2021-11-03
Author: Adrian G. Zucco (adrian.gabriel.zucco@regionh.dk)
-----
Last Modified: 2021-11-03 10:57:04 pm
Modified By: Adrian G. Zucco (adrian.gabriel.zucco@regionh.dk>)
'''


import pandas as pd
import numpy as np
import numbers
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import expit



def logitise(x):
    return np.log(x/(1-x))


def survival_df(predicted_probs, sample_size, time_cuts, epsilon=0.0):

    if isinstance(time_cuts, numbers.Integral):
        time_cuts = np.arange(start=0, stop=time_cuts + 1, step=1)

    # From pycox surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
    # Numpy computes cumulative products directly
    preds_df = np.stack(np.array_split(predicted_probs, sample_size))
    # Calculate the survival function
    # S(t|x) = cumprod(1 - h(t|x)) with an added value to avoid 0 values
    surv_mat = np.cumprod(1 - preds_df + epsilon, axis=1)
    surv_mat = np.insert(surv_mat, 0, np.repeat(1, sample_size), axis=1)

    surv_df = pd.DataFrame(surv_mat.T, time_cuts)
    return(surv_df)



def hazard_df(predicted_probs, sample_size, time_cuts):
    if isinstance(time_cuts, numbers.Integral):
        time_cuts = np.arange(start=0, stop=time_cuts + 1, step=1)

    haz_mat = np.stack(np.array_split(predicted_probs, sample_size))
    haz_mat = np.insert(haz_mat, 0, np.repeat(0, sample_size), axis=1)
    haz_df = pd.DataFrame(haz_mat.T, time_cuts)

    return haz_df


def to_predict_DT(to_predict_data, time_cuts, time_column_name):

    if isinstance(time_cuts, numbers.Integral):
        time_cuts = np.arange(start=1, stop=time_cuts + 1, step=1)

    # Create dataframe of the time cuts
    t_df = pd.DataFrame({time_column_name: time_cuts})
    # Expand the data by duplicating rows according to the ammount of time cuts
    dup_pred = to_predict_data.iloc[np.arange(
        len(to_predict_data)).repeat(t_df.shape[0])]
    # Match length of both dataframes
    t_df = pd.concat([t_df] * to_predict_data.shape[0], ignore_index=True)
    # Concatenante dataframes
    to_pred_with_time = pd.concat(
        [dup_pred.reset_index(drop=True), t_df], axis=1)

    return(to_pred_with_time)


def make_range(x): return tuple(np.arange(start=1, stop=x + 1, step=1))


def scale_shap_values(shap_values, expected_value, model_predictions):
    
    """
    Scale SHAP values to map probability space. Exact SHAP values for a first order taylor approximation 
    of the logit in the transformed function
    """

    assert shap_values.shape[0] == model_predictions.shape[0]

    # Adapated from https://github.com/slundberg/shap/issues/29 to run in multiple observations

    #Compute the transformed base value byn applying the logit function to the base value
    expected_value_transformed = expit(expected_value)

    #Computing the original_explanation_distance to construct the distance_coefficient later on
    original_explanation_distance = np.sum(shap_values, axis = 1)

    #Computing the distance between the model_prediction and the transformed base_value
    distance_to_explain = model_predictions - expected_value_transformed

    # The distance_coefficient is the ratio between both distances which will be used later on
    # Be carefult that this value is not 0
    # Explained here https://github.com/slundberg/shap/issues/29#issuecomment-408467993
    # Small factor added to avoid divisions by zero

    distance_coefficient = np.divide(original_explanation_distance, distance_to_explain) + 1e-6

    #Transforming the original shapley values to the new scale
    shap_values_transformed = shap_values / distance_coefficient[:,None]

    return shap_values_transformed, expected_value_transformed
