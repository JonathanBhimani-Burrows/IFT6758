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


def visualize_comparison(x, y, gender, path):
    print('Visualize', gender + '...')
    # pdb.set_trace()
    plt.plot(x, y)
    plt.xlabel('beard')
    plt.ylabel('mustache')

    # Binning
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    hb = ax.hexbin(x, y, gridsize=11, cmap='inferno')
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_title(gender)
    ax.set_xlabel('Beard')
    ax.set_ylabel('Mustache')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Count')

    plt.savefig(os.path.join(path, gender + '_Beard-Mustache_bin.png'))


def table_compare(x, y, z, x_title, y_title, z_title, path):
    def has_attr(cpt, gender):
        if y[cpt] == 1:
            dict_gender[gender][y_title] += 1
        if z[cpt] == 1:
            dict_gender[gender][z_title] += 1

    dict_gender = {'male': {'beard': 0, 'mustache': 0, 'unconclusive_image': 0}, 'female': {'beard': 0, 'mustache': 0, 'unconclusive_image': 0}}
    for cpt in range(len(x)):
        if x[cpt] == 1:
            gender = 'female'
            has_attr(cpt, gender)
        else:
            gender = 'male'
            has_attr(cpt, gender)

    print('dict of gender')
    print(dict)




def get_visualization(userids, image_data, profile, output_path):
    print('get_visualization...')
    female_beards = []
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


    print('UNCONCLUSIVE ==> male:', cpt_unconclusive_male, ', female:', cpt_unconclusive_female)
    print('total:', len(userids), ' - perc of unconclusive:', (cpt_unconclusive_female + cpt_unconclusive_male)/len(userids)*100, '%' )
    visualize_comparison(female_beards, female_mustaches, 'female', output_path)
    visualize_comparison(male_beards, male_mustaches, 'male', output_path)


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
