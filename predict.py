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
from sklearn.ensemble import GradientBoostingRegressor

def get_psych_predictions(filename, X_test):
    # load the model from disk

    path = os.path.join('models', filename)
    loaded_model = pickle.load(open(path, 'rb'))
    predictions = loaded_model.predict([X_test.values[0]])
    return predictions[0]


def get_prediction_gender(filename, X_test):
    # load the model from disk

    path = os.path.join('models', filename)
    loaded_model = pickle.load(open(path, 'rb'))
    predictions = loaded_model.predict(X_test)
    return int(predictions[0])

def get_predictions(filename, X_test):
    # load the model from disk

    path = os.path.join('/home/mila/teaching/user06/submissions/IFT6758/models', filename)
    loaded_model = pickle.load(open(path, 'rb'))
    predictions = loaded_model.predict(X_test)
    return predictions[0]

def predict():
    output_path = args.o
    if not os.path.exists(output_path):
        print('create output_path:', output_path)
        os.makedirs(output_path)

    nrc_data, liwc_data, relation_data, profile_data, image_data = utils.dataloader.load_data(args.i)

    df = create_merge_df(nrc_data, liwc_data, relation_data, profile_data, image_data)

    baseline_data_path = '/home/mila/teaching/user06/submissions/IFT6758/data/Baseline_data.csv'

    baseline_data = Baseline.load_data(baseline_data_path)

    userids = profile_data['userid'].values
    df_merge = pd.merge(liwc_data, nrc_data, on="userId")

    i = 0

    for uid in userids:

        # pdb.set_trace()
        # Predict baseline
        prediction = [get_predictions('age_model.pkl', liwc_data[liwc_data['userId'] == uid].iloc[:, 1:]),
                        int(baseline_data['gender'][0]),
                        get_psych_predictions('ext_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        get_psych_predictions('neu_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        get_psych_predictions('agr_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        get_psych_predictions('con_model.pkl', df['liwc_nrc'][df['userid'] == uid]),
                        get_psych_predictions('ope_model.pkl', df['liwc_nrc'][df['userid'] == uid])]

        index_list = image_data.index[image_data['userId'] == uid].tolist()
        if len(index_list) == 1:
            gender_prediction = get_prediction_gender('gender_model.pkl', image_data.loc[index_list[0]][2:].values.reshape(1, -1))
            prediction[1] = gender_prediction
            # print('GENDER:', gender_prediction, 'id', image_data.loc[index_list[0]][1], 'id2', image_data.loc[index_list[0]][0])

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
