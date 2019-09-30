import argparse
import os
import Baseline
import pandas as pd
from utils.xml_maker import make_xml
import pdb


def predict():

    profile_filename = 'Profile.csv'

    profile_path = os.path.join(args.i, profile_filename)
    output_path = args.o
    if not os.path.exists(output_path):
        print('create output_path:', output_path)
        os.makedirs(output_path)

    # Baseline.eval(path=profile_path)

    baseline_data_path = '/home/mila/teaching/user06/IFT6758/data/Baseline_data.csv'

    data = Baseline.load_data(baseline_data_path)
    profile = Baseline.load_data(profile_path)
    userids = profile['userid'].values

    for id in userids:
        # pdb.set_trace()
        make_xml(save_dir=output_path, uid=id, age_group=data['age'][0], gender=int(data['gender'][0]), extrovert=data['ext'][0],
                 neurotic=data['neu'][0], agreeable=data['agr'][0], conscientious=data['con'][0], _open=data['ope'][0])
    print('end')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Public_Test/Profile/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/results/')
    args = parser.parse_args()
    print('input path:', args.i)
    print('output path:', args.o)
    predict()
