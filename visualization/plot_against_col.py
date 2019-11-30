import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def plot_against_col(df, cols):
    """
    Plot columns of a dataframe based on feature in first column
    df:     dataframe containing merged data
    cols:   list of the cols to plot against the first col in cols
    """
    x_vals = np.unique(np.array(df[2].values))

    for col in cols[1:]:
        y_col = [df[col][df[cols[0]] == x_val].mean() for x_val in x_vals]
        plt.scatter(x_vals, y_col, color='blue')

        plt.title(col)
        plt.axvline(x=25, color='green')
        plt.axvline(x=35, color='orange')
        plt.axvline(x=50, color='red')

        plt.show()


with open("data\\combined_df_text.pkl", "rb") as df_file:
    df = pickle.load(df_file)

cols = [col for col in df.columns]

col_id = [2, 16, 17, 20, 34, 40, 41, 43, 70, 71, 72, 78, 79, 82]
cols = [cols[idx] for idx in col_id]

plot_against_col(df, cols)

