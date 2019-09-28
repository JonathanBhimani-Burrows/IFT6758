import argparse
import os
import Baseline
import pandas as pd
import xml_maker


def test():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Public_Test/Profile/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='/home/mila/teaching/user06/IFT6758/results/')
    args = parser.parse_args()
    profile_filename = 'Profile.csv'

    profile_path = os.path.join(args.i, profile_filename)
    output_path = args.o
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Baseline.eval(path=profile_path)

    baseline_data_path = '/data/baseline_data.csv'

    data = Baseline.load_data(baseline_data_path)
    profile = Baseline.load_data(profile_path)

    for user in data['userid'].itertuple():
        xml_maker.make_xml(save_dir=output_path, uid=user, age_group=data['age'], gender=data['gender'], extrovert=data['ext'],
                           neurotic=data['neu'], agreeable=data['agr'], conscientious=data['con'], _open=data['ope'])


if __name__ == '__main__':
    test()