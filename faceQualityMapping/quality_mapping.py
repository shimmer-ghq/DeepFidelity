import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.utils import shuffle
import random

def createScore(csv_read_f, csv_read_r, csv_train, csv_val):
    data_f = pd.read_csv(csv_read_f)
    path_name_f = data_f['img'].tolist()
    fqa_f = np.array(data_f['score']).reshape(-1, 1)
    data_r = pd.read_csv(csv_read_r)
    path_name_r = data_r['img'].tolist()
    fqa_r = np.array(data_r['score']).reshape(-1, 1)

    min_max_scaler_f = preprocessing.MinMaxScaler(feature_range=(0, 0.4))
    fqa_f = min_max_scaler_f.fit_transform(fqa_f)

    min_max_scaler_r = preprocessing.MinMaxScaler(feature_range=(0.6, 1))
    fqa_r = min_max_scaler_r.fit_transform(fqa_r)

    fqa_train = list(np.squeeze(fqa_f))
    fqa_train.extend(np.squeeze(fqa_r))
    img_train = path_name_f
    img_train.extend(path_name_r)

    df_train = pd.DataFrame({'img': img_train, 'score': fqa_train})
    df_train = shuffle(df_train)
    df_train.to_csv(csv_train, sep=',', index=None)


if __name__ == "__main__":
    '''
        IMG
    '''
    '''
        Train & Val
    '''
    csv_read_f = r'fqa_train_f.csv'
    csv_read_r = r'fqa_train_r.csv'
    csv_train = r'train.csv'
    csv_val = r''
    createScore(csv_read_f, csv_read_r, csv_train, csv_val)