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


def visualize_comparison(x, y, x_title, y_title, path):
    print('Visualize', y_title + '...')
    # pdb.set_trace()
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(os.path.join(path, y_title + '_by_' + x_title + '.png'))

def get_visualization(userids, image_data, profile, output_path):
    print('get_visualization...')
    beards = []
    mustaches = []
    genders = []
    genders_df = []
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

        genders_df.append(profile[profile['userid'] == uid]['gender'])

    for cpt_user in range(len(genders_df)):
        pdb.set_trace()
        genders.append(genders_df[cpt_user][1])



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
