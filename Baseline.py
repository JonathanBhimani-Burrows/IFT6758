import pandas as pd


def main(inpath, outpath):
    dat = load_data(inpath)
    out = find_baseline(dat)
    save_file(out, outpath)


def load_data(path):
    data = pd.read_csv(path)
    return data


def save_file(file, path):
    file.to_csv(path)

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
    output = [{'gender':gender.iloc[0], 'age': 0, 'ope': ope, 'con': con, 'ext': ext,'agr': agr, 'neu': neu}]
    df1 = pd.DataFrame(columns = ['gender','age','ope','con','ext','agr','neu'])
    df1 = df1.append(output, ignore_index=False)
    return df1


if __name__ == "__main__":
    in_path = r'C:\Users\jonat\Documents\School\DataScience\Project\Train\Profile\Profile.csv'
    out_path = r'C:\Users\jonat\Documents\School\DataScience\Project\Train\Profile\Baseline_data.csv'
    main(in_path, out_path)
