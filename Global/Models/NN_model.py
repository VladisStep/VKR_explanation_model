from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import my_utils
from Global.HandPD.data import load_hand_pd

path_to_project = my_utils.path_to_project


def create_NN(filename, X_train, Y_train):

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(5,), activation='logistic', max_iter=10000, learning_rate='invscaling',
                      random_state=0)
    )
    model.fit(X_train, Y_train)

    dump(model, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_NN(path_to_project + "Global/Trained_models/NN.joblib", X_train, Y_train)
