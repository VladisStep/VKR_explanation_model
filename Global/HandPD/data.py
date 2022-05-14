import numpy as np
import pandas as pd
from sklearn import preprocessing

import my_utils

path_to_project = my_utils.path_to_project


def load_hand_pd():
    df = pd.read_csv(path_to_project + "Datasets/HandPD-dataset.csv")  # import dataset

    le = preprocessing.LabelEncoder()
    # df = df.apply(le.fit_transform)
    df['CLASS_TYPE'] = le.fit_transform(df['CLASS_TYPE'])
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['RIGH/LEFT-HANDED'] = le.fit_transform(df['RIGH/LEFT-HANDED'])

    label_index = 3
    Y = np.array(df.iloc[:, label_index])
    X = df.drop(["CLASS_TYPE", "IMAGE_NAME", "_ID_EXAM", "ID_PATIENT"], axis=1)
    feature_names = X.columns
    target_names = ["healthy", "parkinson"]
    # target_names = set(Y)

    return X, Y, feature_names, target_names
