import argparse
import os
import Baseline
import pandas as pd
from utils.xml_maker import make_xml
import pdb
import utils.dataloader
from predictors.gender_predictor import simple_gender_predictor
from models.relations_agglomerator import relations_agglomerator
from utils.save_model import load_model

def predict():
    output_path = args.o
    if not os.path.exists(output_path):
        print('create output_path:', output_path)
        os.makedirs(output_path)

    baseline_data_path = '/home/mila/teaching/user06/submissions/IFT6758/data/Baseline_data.csv'

    baseline_data = Baseline.load_data(baseline_data_path)

    _, _, relation_data, profile_data, image_data = utils.dataloader.load_data(args.i)

    userids = profile_data['userid'].values

    agglo_model = load_model("models/relation_agglo.mdl")

    i = 0

    for uid in userids:
        # Predict everything based on relation agglomeration
        prediction = agglo_model.predict(relation_data, uid)

        if prediction == -1:
            # Predict baseline
            prediction = [baseline_data['age'][0],
                          int(baseline_data['gender'][0]),
                          baseline_data['ext'][0],
                          baseline_data['neu'][0],
                          baseline_data['agr'][0],
                          baseline_data['con'][0],
                          baseline_data['ope'][0]]

        # Predict gender based on facial hair
        gender_pred = simple_gender_predictor(uid, image_data)

        if gender_pred != -1:
            prediction[1] = int(gender_pred)

        make_xml(save_dir=output_path, uid=uid, age_group=prediction[0], gender=prediction[1], extrovert=prediction[2],
                 neurotic=prediction[3], agreeable=prediction[4], conscientious=prediction[5], _open=prediction[6])

        i += 1
        if i % 100 == 0:
            print("Completed predictions for %5.0f users." % i)
    print('end')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', help="path to the input data", type=str,
                        default='/home/mila/teaching/user06/Public_Test/')
    parser.add_argument('-o', help="path to the output data", type=str,
                        default='/home/mila/teaching/user06/submissions/IFT6758/results/')
    args = parser.parse_args()

    print('input path:', args.i)
    print('output path:', args.o)
    predict()
