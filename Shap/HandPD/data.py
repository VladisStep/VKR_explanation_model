import pandas as pd
import numpy as np
from sklearn import preprocessing


def load_hand_pd():
    df = pd.read_csv("/Users/vladis_step/VKR_explanation_model/Datasets/HandPD-dataset.csv")  # import dataset

    le = preprocessing.LabelEncoder()
    df['GENDER'] = le.fit_transform(df['GENDER'])
    df['RIGH/LEFT-HANDED'] = le.fit_transform(df['RIGH/LEFT-HANDED'])

    label_index = 3
    Y = np.array(df.iloc[:, label_index])
    X = df.drop(["CLASS_TYPE", "IMAGE_NAME", "_ID_EXAM", "ID_PATIENT"], axis=1)

    return X, Y
