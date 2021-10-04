from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from Shap.HandPD.data import load_hand_pd
from joblib import dump
from sklearn.svm import SVC, LinearSVC

path_to_project = 'C:/GitReps/VKR_explanation_model/'


def create_SVM(filename, X_train, Y_train):
    # train a SVM classifier
    svm = SVC(kernel='linear', probability=True)
    # For large datasets consider using LinearSVC or SGDClassifier
    # svm = SGDClassifier(loss='hinge',class_weight='balanced')
    svm.fit(X_train, Y_train)

    # save model
    dump(svm, filename)


if __name__ == "__main__":
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)
    create_SVM(path_to_project + "Models/SVM.joblib", X_train, Y_train)
