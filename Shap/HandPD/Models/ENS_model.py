from joblib import dump
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

import my_utils
from Shap.HandPD.data import load_hand_pd

path_to_project = my_utils.path_to_project


def create_ENS(filename, X_train, Y_train):
    # standard bagged decision tree ensemble model
    model = BaggingClassifier()
    model.fit(X_train, Y_train)

    # define evaluation procedure
    # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # save model
    dump(model, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_ENS(path_to_project + "Models/ENS.joblib", X_train, Y_train)
