import argparse
import os

import sys
sys.path.insert(1, 'C:\\Users\\fanny\\Documents\\IFT6758')
import utils.dataloader
import Baseline
import pdb

# def visualize_comparison(x_fem, y_fem, x_male, y_male, path):

    # plt.savefig(os.path.join(path, gender + '_Beard-Mustache_bin.png'))




def get_visualization(userids, image_data, profile, output_path):
    print('get_visualization...')
    age24 = dict()
    age34 = dict()
    age49 = dict()
    age50 = dict()

    cols = image_data.columns.tolist()
    for col in cols:
        age24[col] = []
        age34[col] = []
        age49[col] = []
        age50[col] = []

    cpt = 0
    for uid in userids:
        cpt += 1
        if cpt % 10 == 0:
            print('progress:', cpt/len(userids) * 100)
        uid_data = image_data[image_data['userId']==uid]

        # xx-24
        if profile[profile['userid'] == uid]['age'].tolist()[0] < 25:
            if len(uid_data) == 1:
                for col in cols:
                    age24[col].append(image_data[image_data['userId']==uid][col].iloc[0])
        # 25-34
        elif profile[profile['userid'] == uid]['age'].tolist()[0] < 35:
            if len(uid_data) == 1:
                for col in cols:
                    age34[col].append(image_data[image_data['userId']==uid][col].iloc[0])
        # 35-49
        elif profile[profile['userid'] == uid]['age'].tolist()[0] < 50:
            if len(uid_data) == 1:
                for col in cols:
                    age49[col].append(image_data[image_data['userId']==uid][col].iloc[0])
        # 50-xx
        else:
            if len(uid_data) == 1:
                for col in cols:
                    age50[col].append(image_data[image_data['userId']==uid][col].iloc[0])



    with open(os.path.join(output_path, 'xx-24_oxford.csv'), 'w') as f:
        for key in age24.keys():
            f.write("%s,%s\n" % (key, str(age24[key])[1:-1]))


    with open(os.path.join(output_path, '25-34_oxford.csv'), 'w') as f:
        for key in age34.keys():
            f.write("%s,%s\n" % (key, str(age24[key])[1:-1]))


    with open(os.path.join(output_path, '35-49_oxford.csv'), 'w') as f:
        for key in age49.keys():
            f.write("%s,%s\n" % (key, str(age24[key])[1:-1]))


    with open(os.path.join(output_path, '50-xx_oxford.csv'), 'w') as f:
        for key in age50.keys():
            f.write("%s,%s\n" % (key, str(age24[key])[1:-1]))


    # visualize_comparison(female_beards, female_mustaches, male_beards, male_mustaches, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='../data/Train/')
                        # default='/home/mila/teaching/user06/Train/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='../data/visualization')
                        # default='/home/mila/teaching/user06/submissions/IFT6758/results/visualization')
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
