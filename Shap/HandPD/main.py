import matplotlib.pyplot as plt
import shap
import matplotlib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from Shap.HandPD.Models.SVM_model import create_SVM
from Shap.HandPD.Models.RFC_model import create_RFC
from data import load_hand_pd
from joblib import load
from sklearn import metrics


def shap_svm():
    # create_SVM(SVM_model_filename, X_train, Y_train)
    svm = load(SVM_model_filename)

    # which row from data should shap show?
    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]
    # data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

    # use Kernel SHAP to explain test set predictions
    explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")

    shap_values = explainer.shap_values(data_for_prediction, nsamples=100)
    shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction, link="logit")
    shap.save_html('/Users/vladis_step/VKR_explanation_model/Shap/HandPD/Graphs/graph.html', shap_display)

def shap_rfc():
    rfc = load(RFC_model_filename)
    view = shap.TreeExplainer(rfc)
    # shap.summary_plot(shap_values[1], X_train)

    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]

    shap_values = view.shap_values(data_for_prediction)

    shap_display = shap.force_plot(view.expected_value[1], shap_values[1], data_for_prediction)
    shap.save_html('/Users/vladis_step/VKR_explanation_model/Shap/HandPD/Graphs/graph.html', shap_display)


def print_acc(classifier, X_test, Y_test):
    preds = classifier.predict(X_test)
    print(classification_report(Y_test, preds))
    print('The accuracy of this model is :\t', metrics.accuracy_score(preds, Y_test))


if __name__ == "__main__":
    RFC_model_filename = '/Users/vladis_step/VKR_explanation_model/Models/RFC.joblib'
    SVM_model_filename = '/Users/vladis_step/VKR_explanation_model/Models/SVM.joblib'
    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)

    shap_svm()
    shap_rfc()
    shap_xgb()
    shap_knn()
