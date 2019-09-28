import argparse
import os

def test():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str, default='/home/mila/teaching/user06/Public_Test/Profile/')
    parser.add_argument('-o', help="path to the output data", type=str, default='/home/mila/teaching/user06/IFT6758/results/')
    args = parser.parse_args()
    profile_filename ='Profile.csv'

    profile_path = os.path.join(args.i, profile_filename)



if __name__ == '__main__':
    test()