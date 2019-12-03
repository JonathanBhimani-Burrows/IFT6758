import pandas as pd
import numpy as np
import os
import pdb

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


def create_merge_df(df_nrc, df_liwc, df_relation, df_profile, df_image):
    df = pd.DataFrame()
    df = df_profile.copy(deep=True)
    df['liwc_nrc'] = np.nan
    df['liwc'] = np.nan
    df['nrc'] = np.nan
    df['oxford'] = np.nan
    df = df.astype('object')

    print(df.columns)

    for user in df['userid']:
        placement = df.index[df['userid'] == user][0]

        vec = []
        for col in df_liwc.columns[1:]:
            vec.append(df_liwc[col].loc[df_liwc['userId'] == user].values[0])
        df['liwc'].loc[placement] = vec

        vec = []
        for col in df_nrc.columns[1:]:
            vec.append(df_nrc[col].loc[df_nrc['userId'] == user].values[0])
        df['nrc'].loc[placement] = vec

        vec = []
        # df.merge()
        # for col in df_image.columns[1:]:
        #
        #     pdb.set_trace()
        #     index = df.index[df_image['userId'] == user]
        #     row = df_image[col].loc[index]
        #     value = row.values[0]
        #     vec.append(value)
        df['oxford'].loc[placement] = vec

        df['liwc_nrc'].loc[placement] = df['liwc'].loc[placement] + df['nrc'].loc[placement]

    return df


if __name__ == '__main__':

    data = load_data('/home/mila/teaching/user06/Public_Test/')

    df_nrc = data[0].iloc[:, :]
    df_liwc = data[1].iloc[:, :]
    df_relation = data[2].iloc[:, :]
    df_profile = data[3].iloc[:, :]
    df_image = data[1].iloc[:, :]

    df = create_merge_df(df_nrc, df_liwc, df_relation, df_profile, df_image)

    if not os.path.exists('data/'):
        os.mkdir('data/')
    df.to_pickle('data/df_vector.pkl')