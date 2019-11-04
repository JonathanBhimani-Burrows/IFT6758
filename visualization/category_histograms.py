import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import pdb
import os

import sys
sys.path.insert(1, 'C:\\Users\\fanny\\Documents\\IFT6758')
import utils.dataloader
import Baseline


def get_visualization(userids, image_data, profile, output_path):

    for row in image_data[2:].itertuples():
        pdb.set_trace()
        lst = []
        for x in row[2:]:
            lst.append(x)

        print(row[0])
        print(row[1:].means())

        # plt.hist(row[1:])
        # plt.title(row[0])
        # plt.savefig(os.path.join(output_path, row[0] + '_img.png'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='../data/')
                        # default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='../results/visualization/ages/')
                        # default='/home/mila/teaching/user06/submissions/IFT6758/results/visualization')
    args = parser.parse_args()

    print('input path:', args.i)
    print('output path:', args.o)

    output_path = args.o
    if not os.path.exists(output_path):
        print('create output_path:', output_path)
        os.makedirs(output_path)


    print('Visualization...')
    profile_filename = 'Train/Profile/Profile.csv'

    profile_path = os.path.join(args.i, profile_filename)

    profile = Baseline.load_data(profile_path)
    userids = profile['userid'].values


    # _, _, _, _, image_data = utils.dataloader.load_data(args.i)

    lst_file = os.listdir(os.path.join(args.i, 'visualization'))
    for file in lst_file:
        # _, _, _, _, image_data = utils.dataloader.load_data(os.path.join(args.i, file))
        image_data = pd.read_csv(os.path.join(args.i, 'visualization', file))
        # _, _, _, _, image_data = load_csv(datapath, 'Image', 'oxford.csv')(os.path.join(args.i, file), 'Image')

        get_visualization(userids, image_data, profile, args.o)
