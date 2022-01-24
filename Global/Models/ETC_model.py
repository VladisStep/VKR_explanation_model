from joblib import dump
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

import my_utils
from Global.HandPD.data import load_hand_pd

path_to_project = my_utils.path_to_project


def create_ETC(filename, X_train, Y_train):
    etc = ExtraTreesClassifier(n_estimators=100)
    etc.fit(X_train, Y_train)

    dump(etc, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_ETC(path_to_project + "Global/Trained_models/ETC.joblib", X_train, Y_train)
