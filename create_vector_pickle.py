import pandas as pd
import numpy as np
import os

def load_csv(datapath, dirname, filename):
    data_dirpath = os.path.join(datapath, dirname)
    if os.path.isdir(data_dirpath):
        data_filepath = os.path.join(data_dirpath, filename)

        if os.path.isfile(data_filepath):
            data = pd.read_csv(data_filepath)
            print("Loaded %s Data" % (dirname))

            return data

        else:
            print("Couldn't find file: %s" % (data_filepath))
            return None

    else:
        print("Couldn't find dir: %s" % (data_dirpath))
        return None


def load_data(datapath):
    """
    Function to load all csv files
    Input:  Path to Main Data dir
    Output: Pandas dataframes for each csv
    """

    nrc_text_data = load_csv(datapath, 'Text', 'nrc.csv')
    liwc_text_data = load_csv(datapath, 'Text', 'liwc.csv')
    relation_data = load_csv(datapath, 'Relation', 'Relation.csv')
    profile_data = load_csv(datapath, 'Profile', 'Profile.csv')
    image_data = load_csv(datapath, 'Image', 'oxford.csv')

    return nrc_text_data, liwc_text_data, relation_data, profile_data, image_data


if __name__ == '__main__':

    data = load_data('data/Train')

    df_nrc = data[0].iloc[:, :]
    df_liwc = data[1].iloc[:, :]
    df_relation = data[2].iloc[:, :]
    df_profile = data[3].iloc[:, :]
    df_image = data[1].iloc[:, :]

    df = pd.DataFrame()
    df = df_profile.copy(deep=True)
    df['liwc_nrc'] = np.nan
    df['liwc'] = np.nan
    df['nrc'] = np.nan
    df['oxford'] = np.nan
    df = df.astype('object')

    print(df.columns)

    len_df = len(df)
    print(int(len_df / 100))
    for user in df['userId']:
        placement = df['userId'] == user

        vec = []
        for col in df_liwc.columns[1:]:
            vec.append(df_liwc[col].loc[df_liwc['userId'] == user])
        df['liwc'].loc[placement] = vec

        vec = []
        for col in df_nrc.columns[1:]:
            vec.append(df_nrc[col].loc[df_nrc['userId'] == user])
        df['nrc'].loc[placement] = vec

        vec = []
        for col in df_image.columns[1:]:
            vec.append(df_image[col].loc[df_image['userId'] == user])
        df['oxford'].loc[placement] = vec

        df['liwc_nrc'].loc[placement] = df['liwc'].loc[placement] + df['nrc'].loc[placement]

    if not os.path.exists('data/'):
        os.mkdir('data/')
    df.to_pickle('data/df_vector.pkl')