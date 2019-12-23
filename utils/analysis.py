import os
import pandas as pd
import xml.etree.cElementTree as ET
from dataloader import clean_profile_data, load_csv
from xml_maker import age_list, gender_list
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse


def _add_line_from_xml(df, xml):
    xml_tree = ET.parse(xml)

    predictions = xml_tree.getroot().attrib

    df = df.append({'userid':predictions['id'],
                    'age':predictions['age_group'],
                    'gender':predictions['gender'],
                    'ext':predictions['extrovert'],
                    'neu':predictions['neurotic'],
                    'agr':predictions['agreeable'],
                    'con':predictions['conscientious'],
                    'ope':predictions['open']},
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

    return y_merged


def confusion_mat(df_col1, df_col2, title, labels, save_path):
    confusion = confusion_matrix(df_col1, df_col2)
    
    ax = sns.heatmap(confusion,
                     annot=True,
                     fmt="d",
                     xticklabels=labels,
                     yticklabels=labels)

    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    plt.savefig(save_path)

    return confusion


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_dir', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-r', '--results_dir', help="path to the results data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/results/')
    args = parser.parse_args()

    y_merged = make_y_dataframe(args.data_dir, args.results_dir)

    gender_confusion = confusion_mat(
        y_merged['gender_gt'],
        y_merged['gender_pred'],
        title="Gender confusion matrix",
        labels=["male", "female"],
        save_path="analysis\\gender_confusion.png")

    age_confusion = confusion_mat(
        y_merged['age_gt'],
        y_merged['age_pred'],
        title="Age confusion matrix",
        labels=["xx-24", "25-34", "35-49", "50-xx"],
        save_path="analysis\\age_confusion.png")
