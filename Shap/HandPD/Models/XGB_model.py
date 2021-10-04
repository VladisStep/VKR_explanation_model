import xgboost

from joblib import dump
from sklearn.model_selection import train_test_split

from Shap.HandPD.data import load_hand_pd

path_to_project = 'C:/GitReps/VKR_explanation_model/'


def create_XGB(filename, X_train, Y_train):
    model = xgboost.XGBRegressor().fit(X_train, Y_train)

    # save model
    dump(model, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_XGB(path_to_project + "Models/XGB.joblib", X_train, Y_train)
