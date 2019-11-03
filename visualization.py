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


def visualize_comparison(x, y, x_title, y_title, path):
    print('Visualize', y_title + '...')
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(os.path.join(path, y_title + '_by_' + x_title + '.png'))

def get_visualization(userids, image_data, profile, output_path):
    print('get_visualization...')
    beards = []
    mustaches = []
    genders = []
    for uid in userids:
        uid_data = image_data[image_data['userId']==uid]
        if len(uid_data) == 1:
            mustache = image_data[image_data['userId']==uid]['facialHair_mustache'].iloc[0]
            beard = image_data[image_data['userId']==uid]['facialHair_beard'].iloc[0]

            if beard > 0:
                beards.append(1)
            else:
                beards.append(0)


            if mustache > 0:
                mustaches.append(1)
            else:
                mustaches.append(0)

        else:
            beards.append(2)
            mustaches.append(2)

        genders.append(profile[profile['userId'] == uid]['gender'])


    visualize_comparison(genders, beards, 'gender', 'beard', output_path)
    visualize_comparison(genders, mustaches, 'gender', 'mustache', output_path)


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

    output_path = args.o
    if not os.path.exists(output_path):
        print('create output_path:', output_path)
        os.makedirs(output_path)


    print('Visualization...')
    profile_filename = 'Profile/Profile.csv'

    profile_path = os.path.join(args.i, profile_filename)

    profile = Baseline.load_data(profile_path)
    userids = profile['userid'].values


    _, _, _, _, image_data = utils.dataloader.load_data(args.i)

    get_visualization(userids, image_data, profile, args.o)
