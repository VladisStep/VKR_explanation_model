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

path_to_project = 'C:/GitReps/VKR_explanation_model/'


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
    shap.save_html(path_to_project + 'Shap/HandPD/Graphs/svm_force_plot.html', shap_display)

    print_acc(svm, X_test, Y_test)

    # shap.plots.beeswarm(shap_values)


def shap_rfc():
    rfc = load(RFC_model_filename)
    view = shap.TreeExplainer(rfc)
    # shap.summary_plot(shap_values[1], X_train)

    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]

    shap_values = view.shap_values(data_for_prediction)

    shap_display = shap.force_plot(view.expected_value[0], shap_values[0], data_for_prediction)
    shap.save_html(path_to_project + 'Shap/HandPD/Graphs/rfc_force_plot.html', shap_display)
    print_acc(rfc, X_test, Y_test)

    # shap.plots.beeswarm(shap_values)


def shap_xgb():
    xgb = load(XGB_model_filename)
    view = shap.Explainer(xgb)

    choosen_instance = 5
    shap_values = view(X_test)

    shap.plots.waterfall(shap_values[choosen_instance], show=False)
    plt.savefig(path_to_project + 'Shap/HandPD/Graphs/xgb_waterfall')
    shap_display = shap.plots.force(shap_values[choosen_instance])
    shap.save_html(path_to_project + 'Shap/HandPD/Graphs/xgb_force_plot.html', shap_display)
    # print_acc(xgb, X_test, Y_test)


def shap_knn():
    knn = load(KNN_model_filename)
    view = shap.KernelExplainer(knn.predict_proba, X_train)

    choosen_instance = 5
    data_for_prediction = X_test.iloc[choosen_instance]
    shap_values = view.shap_values(data_for_prediction)

    shap_display = shap.force_plot(view.expected_value[0], shap_values[0], data_for_prediction)
    shap.save_html(path_to_project + 'Shap/HandPD/Graphs/knn_force_plot.html', shap_display)
    print_acc(knn, X_test, Y_test)

    # shap.plots.beeswarm(shap_values)


def print_acc(classifier, X_test, Y_test):
    preds = classifier.predict(X_test)
    print(classification_report(Y_test, preds))
    print('The accuracy of this model is :\t', metrics.accuracy_score(preds, Y_test))


if __name__ == "__main__":
    RFC_model_filename = path_to_project + 'Models/RFC.joblib'
    SVM_model_filename = path_to_project + 'Models/SVM.joblib'
    XGB_model_filename = path_to_project + 'Models/XGB.joblib'
    KNN_model_filename = path_to_project + 'Models/KNN.joblib'

    test_size = 0.5
    X, Y = load_hand_pd()
    X_train, X_test, Y_train, Y_test = train_test_split(*[X, Y], test_size=test_size, random_state=0)

    shap_svm()
    shap_rfc()
    shap_xgb()
    shap_knn()
