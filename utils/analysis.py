import os
import pandas as pd
import xml.etree.cElementTree as ET
from dataloader import clean_profile_data, load_csv
from xml_maker import age_list, gender_list
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from math import ceil


def _add_line_from_xml(df, xml):
    xml_tree = ET.parse(xml)

    predictions = xml_tree.getroot().attrib

    df = df.append({'userid':predictions['id'],
                    'age':predictions['age_group'],
                    'gender':predictions['gender'],
                    'ext':pd.to_numeric(predictions['extrovert']),
                    'neu':pd.to_numeric(predictions['neurotic']),
                    'agr':pd.to_numeric(predictions['agreeable']),
                    'con':pd.to_numeric(predictions['conscientious']),
                    'ope':pd.to_numeric(predictions['open'])},
                    ignore_index=True)

    return df

def load_xmls(results_dir):
    df = pd.DataFrame({'userid':[],
                       'age':[],
                       'gender':[],
                       'ext':[],
                       'neu':[],
                       'agr':[],
                       'con':[],
                       'ope':[]})

    files = [f for f in os.listdir(results_dir) if ".xml" in f]

    for f in files:
        df = _add_line_from_xml(df, os.path.join(results_dir, f))

    return df

def make_y_dataframe(data_dir, results_dir):
    predictions = load_xmls(results_dir)

    ground_truth = load_csv(data_dir, 'Profile', 'Profile.csv')

    AGE_BINS = [0, 24, 34, 49, 1000]
    ground_truth = clean_profile_data(ground_truth, AGE_BINS)

    ground_truth['age'] = ground_truth['age'].apply(lambda x: age_list[int(x)])
    ground_truth['gender'] = ground_truth['gender'].apply(lambda x: gender_list[int(x)])

    y_merged = pd.merge(ground_truth, predictions,
                        on=['userid'], how="left", suffixes=("_gt", "_pred"))

    y_merged = y_merged.dropna()

    y_merged = y_merged.replace("xx-24", "00-24")

    return y_merged


def confusion_mat(df_col_gt, df_col_pred, title, labels, save_path):
    confusion = confusion_matrix(df_col_gt, df_col_pred)

    plt.figure()
    
    ax = sns.heatmap(confusion,
                     cmap=sns.dark_palette("purple"),
                     annot=True,
                     fmt="d",
                     xticklabels=labels,
                     yticklabels=labels)

    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    plt.savefig(save_path)

    return confusion

def residual_error(df_col_gt, df_col_pred, baseline, title, save_path):
    df_residuals = df_col_pred - df_col_gt
    x_lim = max([abs(df_residuals.max()), abs(df_residuals.min())])
    x_lim = ceil(x_lim * 10) / 10

    plt.figure()

    plt.axvline(x=-baseline, color='k')
    plt.axvline(x=baseline, color='k')
    plt.axvline(x=0, color='k', linestyle=":")

    plt.legend(["Baseline = " + str(baseline)])
    
    bins = [i/10 for i in range(-40, 40)]
    hist = df_residuals.hist(bins=bins, color='purple', edgecolor='indigo')

    hist.set(xlim=(-x_lim, x_lim))
    hist.locator_params(axis='y', integer=True)
    hist.grid(False)

    hist.set_title(title)
    hist.set_xlabel("Prediction residuals")
    hist.set_ylabel("Count")

    plt.savefig(save_path)

    return df_residuals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-r', '--results_dir', help="path to the results data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/results/')
    parser.add_argument('-s', '--save_dir', help="path to the results data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/analysis/')               
    args = parser.parse_args()

    y_merged = make_y_dataframe(args.data_dir, args.results_dir)

    ope_residuals = residual_error(
        y_merged['ope_gt'],
        y_merged['ope_pred'],
        baseline=0.632,
        title="Openess prediction residuals",
        save_path=os.path.join(args.save_dir, "ope_residuals.png"))

    neu_residuals = residual_error(
        y_merged['neu_gt'],
        y_merged['neu_pred'],
        baseline=0.793,
        title="Neurotic prediction residuals",
        save_path=os.path.join(args.save_dir, "neu_residuals.png"))

    ext_residuals = residual_error(
        y_merged['ext_gt'],
        y_merged['ext_pred'],
        baseline=0.801,
        title="Extrovert prediction residuals",
        save_path=os.path.join(args.save_dir, "ext_residuals.png"))

    agr_residuals = residual_error(
        y_merged['agr_gt'],
        y_merged['agr_pred'],
        baseline=0.661,
        title="Agreeable prediction residuals",
        save_path=os.path.join(args.save_dir, "agr_residuals.png"))

    con_residuals = residual_error(
        y_merged['con_gt'],
        y_merged['con_pred'],
        baseline=0.720,
        title="Conscientious prediction residuals",
        save_path=os.path.join(args.save_dir, "con_residuals.png"))

    gender_confusion = confusion_mat(
        y_merged['gender_gt'],
        y_merged['gender_pred'],
        title="Gender confusion matrix",
        labels=["female", "male"],
        save_path=os.path.join(args.save_dir, "gender_confusion.png"))

    age_confusion = confusion_mat(
        y_merged['age_gt'],
        y_merged['age_pred'],
        title="Age confusion matrix",
        labels=["xx-24", "25-34", "35-49", "50-xx"],
        save_path=os.path.join(args.save_dir, "age_confusion.png"))
