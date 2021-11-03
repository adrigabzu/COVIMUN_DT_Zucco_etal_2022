'''
File: run_predictions.py
File Created: 2021-11-03
Author: Adrian G. Zucco (adrian.gabriel.zucco@regionh.dk)
-----
Last Modified: 2021-11-03 1:50:41 pm
Modified By: Adrian G. Zucco (adrian.gabriel.zucco@regionh.dk>)
'''

# %% LOAD LIBRARIES
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import sklearn
import shap
import os
import glob
import time
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import argparse

from textwrap import wrap
from scipy import stats, cluster
from scipy.special import expit
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom Functions
from utils.ML_funcs import *
from utils.visualizations import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict survival probabilites withing 12 weeks from a first SARS-CoV-2 positive test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_data', type=str, default="./data_to_predict.csv",
                        help='Data frame with feature names as header and one patient per row')
    parser.add_argument('--models_folder', type=str, default="./trained_models",
                        help='Path to the folder with the trained models')
    parser.add_argument('--output_folder', type=str, default="./results",
                        help='Path to the folder in which results will be saved')
    parser.add_argument('--col_info', type=str, default="./metadata/column_details.csv",
                        help='Path to the file with the columns metadata')
    parser.add_argument('--plot_curves', default='True',
                        choices=('True', 'False'),
                        help='Generate cumulative incidence and instant hazard plots for each patient')
    parser.add_argument('--explain', default='True',
                        choices=('True', 'False'),
                        help='Generate individual explanations for each patient')

    args = parser.parse_args()
    parser.print_help()
    return args


def check_output_folder(output_folder, plot_curves, explain):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if plot_curves == "True":
        if not os.path.exists(output_folder + "/surv_haz_curves"):
            os.makedirs(output_folder + "/surv_haz_curves")

    if explain == "True":
        if not os.path.exists(output_folder + "/explanations"):
            os.makedirs(output_folder + "/explanations")


def load_data(input_data):
    print("\n### LOADING DATA")

    if input_data.endswith("xlsx"):
        data = pd.read_excel(input_data)
    elif input_data.endswith("csv"):
        data = pd.read_csv(input_data)
    print(data.info())

    return data


def load_col_info(col_info_data):
    print("\n### LOADING COLUMN INFORMATION")

    if col_info_data.endswith("xlsx"):
        col_info_data = pd.read_excel(col_info_data)
    elif col_info_data.endswith("csv"):
        col_info_data = pd.read_csv(col_info_data)

    return col_info_data


def load_models(folder_name):
    print("\n### LOADING MODELS")

    models = {}
    for filename in glob.glob("{folder}/*".format(folder=folder_name)):
        model_type = filename.split("-")[-1][:-4]

        if 'lgb' in model_type:
            models[model_type] = lgb.Booster(model_file=filename)
            print("## Model ", model_type, " loaded.")

    return models


def run_models(data, tomodel, models):
    print("\n### Running predictions")
    predicted_probs = {}
    predicted_logodds = {}

    X_topred = tomodel.drop("index", axis=1)
    sample_size = data.shape[0]

    for model_type, model in models.items():
        print("## Predicting using model: ", model_type, "...    ", end =" ")

        y_topred_prob = model.predict(X_topred)

        if 'lgb' in model_type:
            model.params["objective"] = "binary"
            y_topred_logodd = model.predict(X_topred, raw_score=True)

        if y_topred_prob.ndim == 2:
            y_topred_prob = y_topred_prob[:, 1]

        predicted_probs[model_type] = y_topred_prob
        predicted_logodds[model_type] = y_topred_logodd
        print(" DONE!")

        # %% Combine predictions
        pred_log_odds_df = pd.concat(
            [tomodel[["index", "range_time"]], pd.DataFrame(predicted_logodds)], axis=1)
        pred_log_odds_df["predicted_log_odds_mean"] = pred_log_odds_df.iloc[:, 2:].mean(
            axis=1)

        # Probabilities
        pred_probs_df = pd.concat(
            [tomodel[["index", "range_time"]], pd.DataFrame(predicted_probs)], axis=1)
        # Calculate the mean in log-odds and then expit(x) = 1/(1+exp(-x)).
        pred_probs_df["predicted_probs_mean"] = expit(
            pred_log_odds_df["predicted_log_odds_mean"])

    return pred_log_odds_df, pred_probs_df


def plot_curves(out, time_steps, max_time, surv_df, haz_df):

    out = out + "/surv_haz_curves"

    for PID in surv_df.index:
        fig, ax = discrete_time_plot(time_steps,
                                     1 - surv_df.loc[PID],
                                     max_time,
                                     vertical_line_at=None)
        plt.ylabel('Predicted cumulative incidence F(t)')
        plt.xlabel('Weeks')

        plt.savefig(out + "/pred_CI_{}.png".format(str(PID)),
                    bbox_inches="tight")
        plt.clf()

        fig, ax = hazard_plot(time_steps,
                              haz_df.loc[PID],
                              max_time,
                              vertical_line_at=None)
        plt.ylabel('Predicted instant hazard h(t)')
        plt.xlabel('Weeks')

        plt.savefig(out + "/pred_hazard_{}.png".format(str(PID)),
                    bbox_inches="tight")
        plt.clf()


def compute_SHAP(models, X_topred, pred_probs):
    print("# Calculating SHAP values")
    start_time = time.time()

    shap_val_list = []
    expected_values = []

    for model_name, model in models.items():
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(
            X_topred, check_additivity=True)
        shap_val_list.append(shap_values)
        expected_values.append(explainer.expected_value)

    shap_val_list = [shaps[1] for shaps in shap_val_list]
    expected_values = [expected[1] for expected in expected_values]

    # Mean of shap values and expected value
    shap_values = np.array(shap_val_list).mean(axis=0)
    expected_value = np.mean(expected_values)

    shap_values, expected_value = scale_shap_values(
        shap_values, expected_value, pred_probs)

    return shap_values, expected_value


def cleaner_feature_names(col_info, X_topred):
    columns_info = col_info.fillna('')

    new_col_names = columns_info["short_name"] + " (" + columns_info["short_code"] + \
        "), " + columns_info["source"] + ", " + columns_info["value"]

    shap_data = X_topred.copy()
    shap_data.columns = [x.rstrip(' (), , ') for x in new_col_names]

    shap_data_cat = X_topred.copy()
    shap_data_cat["sex_at_birth"] = shap_data_cat["sex_at_birth"].astype(
        "category").cat.rename_categories(["Female", "Male"])
    shap_data_cat["in_hosp"] = shap_data_cat["in_hosp"].astype(
        "category").cat.rename_categories(["No", "Yes"])
    shap_data_cat.columns = [x.rstrip(' (), , ') for x in new_col_names]

    return shap_data, shap_data_cat


def patient_explained(idx, shap_data, X_topred_patient, toplot_patient_shap, haz_probs, output_folder):

    feat_values = pd.DataFrame(
        {"variable": shap_data.columns, "Value": X_topred_patient[0]})
    feat_values["variable"]
    feat_values["name_value"] = np.repeat("( ", feat_values.shape[0]) + feat_values.Value.round(
        3).astype(str) + np.repeat(" )", feat_values.shape[0]) + np.repeat("  ", feat_values.shape[0]) + feat_values.variable

    to_plot_filt = toplot_patient_shap.set_index(
        np.arange(1, 13)).T.set_index(feat_values["name_value"])

    cg = sns.clustermap(to_plot_filt,
                        center=0,
                        cmap="coolwarm",
                        col_cluster=False,
                        cbar_kws={"orientation": "horizontal",
                                  "label": "SHAP values"},
                        linewidths=.5)  # ,
    # row_linkage = clust)

    cg.fig.set_size_inches(6, 6)
    cg.ax_heatmap.set_xlabel('Week')
    cg.ax_heatmap.set_ylabel('')

    # divide existing axes
    divider = make_axes_locatable(cg.ax_heatmap)
    divider2 = make_axes_locatable(cg.ax_col_dendrogram)

    # create new axes for bar plot
    ax = divider.append_axes("top", size="20%", pad=0.15)

    # create empty space of same size as bar plot axes (don't use this space)
    nax = divider2.new_horizontal(size="20%", pad=1)
    # Sort the values for the bar plot to have the same order as clusters
    target = [t.get_text() for t in np.array(cg.ax_heatmap.get_xticklabels())]

    # Instant hazard plot
    ax.plot(target, haz_probs[1:], color="black")
    ax.set_ylim([0.0, 1.0])
    ax.set_ylabel('h(t|x)')
    plt.setp(ax.get_xticklabels(), visible=False)

    cg.cax.set_position([0.5, 0.75, 0.33, 0.03])
    cg.ax_row_dendrogram.set_visible(False)

    # Save plots
    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
    plt.savefig(output_folder + "/individual_ " + str(idx) +
                "_shap_heatmap.png", bbox_inches="tight")
    plt.clf()


def main():
    argv = parse_args()

    print("\n ###################### START ######################")
    start_time = time.time()
    check_output_folder(argv.output_folder, argv.plot_curves, argv.explain)
    max_time = 12

    models = load_models(argv.models_folder)
    data = load_data(argv.input_data)
    col_info = load_col_info(argv.col_info)

    sample_size = data.shape[0]
    # Augment data into discrete time format
    tomodel = to_predict_DT(data, time_cuts=max_time,
                            time_column_name="range_time")
    X_no_idx = tomodel.drop("index", axis=1)

    pred_log_odds_df, pred_probs_df = run_models(data, tomodel, models)

    pred_probs_df.to_csv(argv.output_folder + "/predicted_probabilites.csv")
    pred_log_odds_df.to_csv(argv.output_folder + "/predicted_log_odds.csv")

    surv_df = survival_df(
        pred_probs_df["predicted_probs_mean"], sample_size, max_time)
    surv_df = surv_df.T
    surv_df.index = data["index"]

    haz_df = hazard_df(
        pred_probs_df["predicted_probs_mean"], sample_size, max_time)
    haz_df = haz_df.T
    haz_df.index = data["index"]

    surv_df.to_csv(argv.output_folder + "/predicted_survival_probabilities.csv")
    haz_df.to_csv(argv.output_folder + "/predicted_instant_hazards.csv")

    time_steps = np.arange(max_time + 1)

    # Generate cumulative incidence and hazard plots
    if argv.plot_curves == "True":
        print("\n### Generating cumulative incidence and instant hazard plots")
        plot_curves(argv.output_folder, time_steps, max_time, surv_df, haz_df)
        print("# DONE, see ", argv.output_folder + "/surv_haz_curves")

    if argv.explain == "True":
        print("\n### Generating explanations")
        # Compute SHAP values
        shap_values, expected_value = compute_SHAP(
            models, X_no_idx, pred_probs_df["predicted_probs_mean"])
        shap_data, shap_data_cat = cleaner_feature_names(col_info, X_no_idx)

        shap_values_pt = np.stack(np.array_split(shap_values, sample_size))
        X_3d = np.stack(np.array_split(X_no_idx.values, sample_size))

        for PID in np.arange(sample_size):
            idx = data["index"][PID]
            patient_shap = shap_values_pt[PID, :, :]
            toplot_patient_shap = pd.DataFrame(
                patient_shap, columns=shap_data.columns)
            X_topred_patient = X_3d[PID, :, :]
            haz_probs = haz_df.loc[idx]
            patient_explained(idx, shap_data, X_topred_patient,
                              toplot_patient_shap, haz_probs,
                              argv.output_folder + "/explanations")

        if sample_size > 2:
            sumplot = shap.summary_plot(
                shap_values[:], shap_data[:], max_display=25, show=False)
            plt.savefig(argv.output_folder +
                        "/explanations/global_shap.png", bbox_inches="tight")
            plt.clf()

        print("# DONE, see ", argv.output_folder + "/explanations", "\n")
    
    print("###################### END ######################")
    print("Analysis completed in ", (time.time() - start_time), " seconds")


if __name__ == "__main__":
    main()
