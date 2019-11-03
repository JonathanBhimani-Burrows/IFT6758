import os
import argparse
import pandas as pd


def load_csv(datapath, dirname, filename):
    data_dirpath = os.path.join(datapath, dirname)
    if os.path.isdir(data_dirpath):
        data_filepath = os.path.join(data_dirpath, filename)
        
        if os.path.isfile(data_filepath):
            data = pd.read_csv(data_filepath)
            print("Loaded %s Data" %(dirname))
            
            return data
        
        else:
            print("Couldn't find file: %s" %(data_filepath))
            return None
    
    else:
        print("Couldn't find dir: %s" %(data_dirpath))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', help="Path to dir containing dataset", type=str,
                        default='../../../Train/')

    args = parser.parse_args()

    nrc_text_data, liwc_text_data, relation_data, profile_data, image_data = load_data(args.data)

    print(nrc_text_data)
    print(liwc_text_data)
    print(relation_data)
    print(profile_data)
    print(image_data)