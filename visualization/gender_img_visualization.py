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


# def visualize_comparison(x_fem, y_fem, x_male, y_male, path):

    # plt.savefig(os.path.join(path, gender + '_Beard-Mustache_bin.png'))




def get_visualization(userids, image_data, profile, output_path):
    print('get_visualization...')
    female = dict()
    male = dict()

    cols = image_data.columns.tolist()
    for col in cols:
        female[col] = []
        male[col] = []

    cpt = 0
    for uid in userids:
        cpt += 1
        if cpt % 10 == 0:
            print('progress:', cpt/len(userids) * 100)
        uid_data = image_data[image_data['userId']==uid]

        # female
        if profile[profile['userid'] == uid]['gender'].tolist()[0] == 1:
            if len(uid_data) == 1:
                for col in cols:
                    female[col].append(image_data[image_data['userId']==uid][col].iloc[0])
                # mustache = image_data[image_data['userId']==uid]['facialHair_mustache'].iloc[0]
                # beard = image_data[image_data['userId']==uid]['facialHair_beard'].iloc[0]

        # male
        else:
            if len(uid_data) == 1:
                for col in cols:
                    male[col].append(image_data[image_data['userId']==uid][col].iloc[0])
                # mustache = image_data[image_data['userId']==uid]['facialHair_mustache'].iloc[0]
                # beard = image_data[image_data['userId']==uid]['facialHair_beard'].iloc[0]

    with open(os.path.join(output_path, 'female_oxford.csv'), 'w') as f:
        for key in female.keys():
            f.write("%s,%s\n" % (key, female[key]))


    with open(os.path.join(output_path, 'male_oxford.csv'), 'w') as f:
        for key in female.keys():
            f.write("%s,%s\n" % (key, female[key]))


    # visualize_comparison(female_beards, female_mustaches, male_beards, male_mustaches, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='data/Train/')
                        # default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/results/visualization')
    args = parser.parse_args()

    print('input path:', args.i)
    print('output path:', args.o)

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
