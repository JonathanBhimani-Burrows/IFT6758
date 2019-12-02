import os
import pdb
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


from sklearn.ensemble import GradientBoostingRegressor
import pickle


from math import sqrt


def train_baseline(train, test, label, pred):

  train_y = train[label].tolist()
  test_y = test[label].tolist()

  predictions = np.full(len(test_y), pred)
  # pdb.set_trace()
  rmse = sqrt(mean_squared_error(test_y, predictions))
  print('RMSE Baseline', label, '==', rmse)
  return rmse



def train_gbr_final(df, preprocess, label, criterion, min_split, lr, n_est):
  train_x = df[preprocess].tolist()
  train_y = df[label].tolist()

  params = {'n_estimators': n_est, 'max_depth': 4, 'min_samples_split': min_split,
            'learning_rate': lr, 'loss': 'ls', 'criterion': criterion}

  clf = GradientBoostingRegressor(**params)
  clf.fit(train_x, train_y)

  return clf


def train_gbr(train, test, preprocess, label, criterion, min_split, lr, n_est):

  train_x = train[preprocess].tolist()
  train_y = train[label].tolist()
  test_x = test[preprocess].tolist()
  text_y = test[label].tolist()

  params = {'n_estimators': n_est, 'max_depth': 4, 'min_samples_split': min_split,
            'learning_rate': lr, 'loss': 'ls', 'criterion': criterion}

  clf = GradientBoostingRegressor(**params)
  clf.fit(train_x, train_y)

  predictions = clf.predict(test_x)
  rmse = sqrt(mean_squared_error(text_y, predictions))

  return rmse, clf


if __name__ == '__main__':
    file_path = 'data/df_vector.pkl'
    df = pd.read_pickle(file_path)

    if not os.path.exists('results/'):
        os.mkdir('results/')

    if not os.path.exists('models/'):
        os.mkdir('models/')

    f = open("results/output.txt", "w+")


    f_model = open("models/output_model.txt", "w+")

    baseline = [3.9086905263157825, 3.445616842105264, 3.486857894736829, 3.5839042105263155, 2.7324242105263203]
    labels = ['con', 'ext', 'agr', 'ope', 'neu']

    kf = KFold(n_splits=10, shuffle=True)

    for cpt_label in range(len(labels)):
        all_best = []
        all_baseline = []
        best = 1
        label = labels[cpt_label]
        base_pred = baseline[cpt_label]

        baseline_rmse = np.array([])
        for train_index, test_index in kf.split(df):
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            baseline_rmse = np.append(baseline_rmse, train_baseline(train, test, label, base_pred))

        base_rmse = baseline_rmse.mean()

        for vector in ['merge', 'liwc', 'nrc']:
            for min_split in [2]: #, 4, 8, 16]:
                for criterion in ['friedman_mse']:  # , 'mae', 'mse']:
                    for lr in [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
                        lr_array = []
                        for n_est in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
                            if len(lr_array) > 2:
                                if round(lr_array[-1], 3) > round(lr_array[-2], 3) \
                                        and round(lr_array[-2], 3) > round(lr_array[-3], 3):
                                    continue
                            rmse = np.array([])
                            for train_index, test_index in kf.split(df):
                                train = df.iloc[train_index]
                                test = df.iloc[test_index]
                                temp, model = train_gbr(train, test, vector, label, criterion, min_split, lr, n_est)
                                rmse = np.append(rmse, temp)

                            res = rmse.mean()
                            lr_array.append(res)
                            if res < best:
                                best = res
                                if best < base_rmse:
                                    print('WIN!!!')
                                    # pickle.dump(model, open(label + '_model.pkl', 'wb'))
                                    model_final = train_gbr_final(df, vector, label, criterion, min_split, lr, n_est)

                                    pickle.dump(model_final,
                                                open('models/' + label + '_model.pkl',
                                                     'wb'))

                                    f_model.write(
                                        'RMSE GradientBoosting - vector: ' + vector + ' - criterion: ' + criterion + ' - min_split: ' + str(
                                            min_split) + ' - lr: ' +
                                        str(lr) + ' - n_est: ' + str(n_est) + ' - ' + label + ' == ' + str(res) + " \r\n")
                                    f_model.write('Baseline: ' + str(base_rmse) + ' - Best: ' + str(best) + ' \r\n')
                                    f_model.flush()

                            print('RMSE GradientBoosting - vector:', vector, '- criterion:', criterion, '- min_split:', min_split, '- lr:',
                                  lr, '- n_est:', n_est, '-', label, '==', res)
                            f.write('RMSE GradientBoosting - vector: ' + vector + ' - criterion: ' + criterion + ' - min_split: ' + str(min_split) + ' - lr: ' +
                                  str(lr) + ' - n_est: ' + str(n_est) + ' - ' + label + ' == ' + str(res) + " \r\n")
                            print('Baseline:', base_rmse, '- Best:', best, '- Model:', res)
                            f.write('Baseline: ' + str(base_rmse) + ' - Best: ' + str(best) + ' - Model: ' + str(res) + ' \r\n')
                            f.flush()
        all_best.append(best)
        all_baseline.append(base_rmse)

    for cpt in range(len(labels)):
        print('\n')
        print(labels[cpt])
        print('Baseline:', all_baseline[cpt])
        print('Best:', all_best[cpt])
        print('\n')

        f.write('\r\n')
        f.write(labels[cpt] + ' \r\n')
        f.write('Baseline: ' + str(all_baseline[cpt]) + ' \r\n')
        f.write('Best: ' + str(all_best[cpt]) + ' \r\n')
        f.write('\r\n')

    print('\n')
    f.write('\r\n')
    f.flush()