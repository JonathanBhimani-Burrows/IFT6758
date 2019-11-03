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


def visualize_comparison(x_fem, y_fem, x_male, y_male, path):

    plt.savefig(os.path.join(path, gender + '_Beard-Mustache_bin.png'))




def get_visualization(userids, image_data, profile, output_path):
    print('get_visualization...')
    female = dict()
    male = dict()
    male_beards = []
    female_mustaches = []
    male_mustaches = []
    # female_genders = []
    # male_genders = []
    cpt_unconclusive_female = 0
    cpt_unconclusive_male = 0


    for uid in userids:
        uid_data = image_data[image_data['userId']==uid]

        # female
        pdb.set_trace()

        if profile[profile['userid'] == uid]['gender'].tolist()[0] == 1:
            if len(uid_data) == 1:
                mustache = image_data[image_data['userId']==uid]['facialHair_mustache'].iloc[0]
                beard = image_data[image_data['userId']==uid]['facialHair_beard'].iloc[0]

                female_beards.append(beard)
                female_mustaches.append(mustache)
            else:
                cpt_unconclusive_female += 1
        # male
        else:
            if len(uid_data) == 1:
                mustache = image_data[image_data['userId']==uid]['facialHair_mustache'].iloc[0]
                beard = image_data[image_data['userId']==uid]['facialHair_beard'].iloc[0]

                male_beards.append(beard)
                male_mustaches.append(mustache)
            else:
                cpt_unconclusive_male += 1

        # genders.append(profile[profile['userid'] == uid]['gender'].tolist()[0])

    # for cpt_user in range(len(genders_df)):
    #     pdb.set_trace()
    #     print(genders_df[cpt_user].tolist())
    #     genders.append(genders_df[cpt_user].tolist()[0])
    #     print(len(genders))


    visualize_comparison(female_beards, female_mustaches, male_beards, male_mustaches, output_path)


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
