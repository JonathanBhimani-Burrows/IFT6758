import pandas as pd


def main(path):
    dat = load_data(path)
    out = find_baseline(dat)

def load_data(path):
    data = pd.read_csv(path)
    return data


def find_baseline(data):
    ope = data['ope'].mean()
    con = data['con'].mean()
    ext = data['ext'].mean()
    agr = data['agr'].mean()
    neu = data['neu'].mean()
    gender = data['gender'].mode()
    bins = [0, 24, 34, 49, 1000]
    age_bins = pd.cut(data['age'], bins)
    max_age_group = data.groupby(['age', age_bins]).size().unstack().sum().idxmax()
    output = [gender.iloc[0], 0, ope, con, ext, agr, neu]
    print(output)
    #print(max_age_group)
    df1 = pd.DataFrame(output, columns = ['gender','age','ope','con','ext','agr','neu'])



# def write_output(data):


if __name__ == "__main__":
    in_path = r'C:\Users\jonat\Documents\School\DataScience\Project\Train\Profile\Profile.csv'
    main(in_path)
