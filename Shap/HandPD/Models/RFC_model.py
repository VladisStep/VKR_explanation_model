from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Shap.HandPD.data import load_hand_pd
from joblib import dump

# https://www.kaggle.com/claytonteybauru/spiral-handpd-recogna

path_to_project = 'C:/GitReps/VKR_explanation_model/'


def create_RFC(filename, X_train, Y_train):
    # train a RFC classifier
    # random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X_train, Y_train)

    # save model
    dump(random_forest, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_RFC("/Users/vladis_step/VKR_explanation_model/Models/RFC.joblib", X_train, Y_train)
