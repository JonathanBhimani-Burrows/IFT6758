import argparse
import os
import Baseline
from utils.xml_maker import make_xml
import pdb
import utils.dataloader
from predictors.gender_predictor import simple_gender_predictor
from utils.save_model import load_model
import pickle
import pandas as pd
import numpy as np
from create_vector_pickle import create_merge_df


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def get_predictions(filename, X_test):
    # load the model from disk
    path = os.path.join('models', filename)
    loaded_model = pickle.load(open(path, 'rb'))
    predictions = loaded_model.score(X_test)
    print('predictions from', filename, 'done')
    return predictions

def predict():
    output_path = args.o
    if not os.path.exists(output_path):
        print('create output_path:', output_path)
        os.makedirs(output_path)

    nrc_data, liwc_data, relation_data, profile_data, image_data = utils.dataloader.load_data(args.i)

    df = create_merge_df(nrc_data, liwc_data, relation_data, profile_data, image_data)

    userids = profile_data['userid'].values

    i = 0

    for uid in userids:
        # Predict baseline
        prediction = [predict('age_model.pkl', df['oxford'][df['userid'] == uid]),
                        int(predict('gender_model.pkl', df['liwc'][df['userid'] == uid])),
                        predict('ext_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        predict('neu_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        predict('agr_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        predict('con_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        predict('ope_model.pkl', df['liwc_nrc'][df['userid'] == uid])]

        make_xml(save_dir=output_path, uid=uid, age_group=prediction[0], gender=prediction[1], extrovert=prediction[2],
                 neurotic=prediction[3], agreeable=prediction[4], conscientious=prediction[5], _open=prediction[6])

        i += 1
        if i % 100 == 0:
            print("Completed predictions for %5.0f users." % i)
    print('end')



if __name__ == '__main__':
    DEFAULT_INPUT = "/home/mila/teaching/user06/Public_Test/"
    DEFAULT_OUTPUT = "/home/mila/teaching/user06/submissions/IFT6758/results/"

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default=DEFAULT_INPUT)
    parser.add_argument('-o', help="path to the output data", type=str,
                        default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print('input path:', args.i)
    print('output path:', args.o)
    predict()
