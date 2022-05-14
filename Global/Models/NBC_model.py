from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import my_utils
from Global.HandPD.data import load_hand_pd

path_to_project = my_utils.path_to_project


def create_NBC(filename, X_train, Y_train):
    model = GaussianNB()
    model.fit(X_train, Y_train)

    # save model
    dump(model, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_NBC(path_to_project + "Global/Trained_models/NBC.joblib", X_train, Y_train)
