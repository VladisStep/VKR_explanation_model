import matplotlib.pyplot as plt
import shap
import matplotlib
import my_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from Shap.HandPD.Models.SVM_model import create_SVM
from Shap.HandPD.Models.RFC_model import create_RFC
from data import load_hand_pd
from joblib import load
from sklearn import metrics

import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

path_to_project = my_utils.path_to_project


def shap_svm():
    # create_SVM(SVM_model_filename, X_train, Y_train)
    method_name = "SVM"

    svm = load(SVM_model_filename)
    explainer = shap.KernelExplainer(svm.predict_proba, X_train.values)

    # which row from data should shap show?
    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]

    print_acc(svm, X_test, Y_test, method_name, data_for_prediction.values.reshape(1, -1))
    # plot_graphs(explainer, data_for_prediction, X_train, method_name)


def shap_rfc():
    method_name = "RFC"

    rfc = load(RFC_model_filename)
    view = shap.TreeExplainer(rfc)
    # view = shap.KernelExplainer(rfc.predict_proba, X_train.values)

    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]

    print_acc(rfc, X_test, Y_test, method_name, data_for_prediction.values.reshape(1, -1))
    # plot_graphs(view, data_for_prediction, X_train, method_name)


def shap_knn():
    method_name = "KNN"

    knn = load(KNN_model_filename)
    view = shap.KernelExplainer(knn.predict_proba, X_train.values)

    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]

    print_acc(knn, X_test, Y_test, method_name, data_for_prediction.values.reshape(1, -1))
    # plot_graphs(view, data_for_prediction, X_train, method_name)


def print_acc(classifier, X_test, Y_test, method_name, data_for_prediction_array):
    print("-------------------------------------------------")
    print("Prediction: ", classifier.predict_proba(data_for_prediction_array))
    print(method_name, ":")
    preds = classifier.predict(X_test.values)
    print(classification_report(Y_test, preds))
    print('The accuracy of this model is :\t', metrics.accuracy_score(preds, Y_test))


def plot_graphs(explainer, data_for_prediction, X, method_name):
    shap_values = explainer.shap_values(data_for_prediction)
    shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
    shap.save_html(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/force_plot.html', shap_display)

    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], data_for_prediction, show=False)
    plt.savefig(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/waterfall.png')
    plt.clf()

    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/summary.png')
    plt.clf()

    # shap.plots.heatmap(shap_values[0], show=False)
    # plt.savefig(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/heatmap_class0.png')
    # plt.clf()
    #
    # shap.plots.heatmap(shap_values[1], show=False)
    # plt.savefig(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/heatmap_class1.png')
    # plt.clf()


if __name__ == "__main__":
    RFC_model_filename = path_to_project + 'Models/RFC.joblib'
    SVM_model_filename = path_to_project + 'Models/SVM.joblib'
    KNN_model_filename = path_to_project + 'Models/KNN.joblib'

    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)

    shap_svm()
    shap_rfc()
    shap_knn()
