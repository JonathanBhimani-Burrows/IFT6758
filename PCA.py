import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
import argparse
import os
import Baseline
import pandas as pd
from utils.xml_maker import make_xml
import pdb
import utils.dataloader


def load_csv(datapath, dirname, filename):
    data_dirpath = os.path.join(datapath, dirname)
    if os.path.isdir(data_dirpath):
        data_filepath = os.path.join(data_dirpath, filename)

        if os.path.isfile(data_filepath):
            data = pd.read_csv(data_filepath)
            print("Loaded %s Data" % (dirname))

            return data

        else:
            print("Couldn't find file: %s" % (data_filepath))
            return None

    else:
        print("Couldn't find dir: %s" % (data_dirpath))
        return None


def load_data(datapath):
    """
    Function to load all csv files
    Input:  Path to Main Data dir
    Output: Pandas dataframes for each csv
    """

    nrc_text_data = load_csv(datapath, 'Text', 'nrc.csv')
    liwc_text_data = load_csv(datapath, 'Text', 'liwc.csv')
    relation_data = load_csv(datapath, 'Relation', 'Relation.csv')
    profile_data = load_csv(datapath, 'Profile', 'Profile.csv')
    image_data = load_csv(datapath, 'Image', 'oxford.csv')

    return nrc_text_data, liwc_text_data, relation_data, profile_data, image_data


class pca_kmeans():
    def __init__(self, components, splits, data):
        self.results = []
        self.comp = components
        self.splits = splits
        self.data = data

    def run(self):
        for i in range(self.comp):
            self.kfold(i)
            print("Component {} out of {} done".format(i + 1, self.comp))
            print(self.results[i])
        return self.results

    def kfold(self, num_comp):
        kf = KFold(n_splits= self.splits)
        temp = []
        for train_index, test_index in kf.split(data):
            self.train = data.iloc[train_index, :]
            self.test = data.iloc[test_index, :]
            self.pca(num_comp)
            t = sum(self.kmeans()) / test_index.shape[0]
            if t < 0.5:
                t = 1 - t
            temp.append(t)
        mean = sum(temp) / len(temp)
        self.results.append(mean)

    def pca(self, num_comp):
        pca = PCA(n_components=num_comp + 1)
        pca.fit(self.train)
        self.f = pca.transform(self.train)
        self.features = pca.transform(self.test)

    def kmeans(self):
        kmeans = KMeans(init='k-means++', n_clusters=2, n_init=25).fit(self.f)
        return kmeans.predict(self.features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/results/visualization')
    args = parser.parse_args()

    print('input path:', args.i)
    print('output path:', args.o)

    #1
    print('Loading...')
    data = load_data(args.i)
    data = data[4].iloc[:, 2:-1]

    #2
    print('PCA...')
    a = pca_kmeans(45, 5, data)
    result = a.run()
