import warnings

import matplotlib.pyplot as plt
import shap
import sklearn.exceptions
from joblib import load
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import my_utils
from Shap.HandPD.Models.KNN_model import create_KNN
from Shap.HandPD.Models.RFC_model import create_RFC
from Shap.HandPD.Models.SVM_model import create_SVM
from Shap.Parkinson_voice.data import load_parkinson_voice

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

path_to_project = my_utils.path_to_project


class ParkinsonVoiceShap:
    def __init__(self):
        self.path_to_project = my_utils.path_to_project
        self.RFC_model_filename = self.path_to_project + 'Models/RFC.joblib'
        self.SVM_model_filename = self.path_to_project + 'Models/SVM.joblib'
        self.KNN_model_filename = self.path_to_project + 'Models/KNN.joblib'
        self.NN_model_filename = self.path_to_project + 'Models/NN.joblib'
        self.dataset_name = 'Parkinson_voice'

        X, Y, self.feature_names = load_parkinson_voice()
        test_size = 0.5
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*[X, Y], test_size=test_size,
                                                                                random_state=0)

    def shap_svm(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_SVM(self.SVM_model_filename, self.X_train, self.Y_train)
        method_name = "SVM"

        svm = load(self.SVM_model_filename)
        explainer = shap.KernelExplainer(svm.predict_proba, self.X_train)

        data_for_prediction = self.X_test[chosen_instance]

        self.print_acc(svm, self.X_test, self.Y_test, method_name, data_for_prediction.reshape(1, -1))
        self.plot_graphs(explainer, data_for_prediction, self.X_train, method_name)

    def shap_rfc(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_RFC(self.RFC_model_filename, self.X_train, self.Y_train)
        method_name = "RFC"

        rfc = load(self.RFC_model_filename)
        view = shap.TreeExplainer(rfc)

        data_for_prediction = self.X_test[chosen_instance]

        self.print_acc(rfc, self.X_test, self.Y_test, method_name, data_for_prediction.reshape(1, -1))
        self.plot_graphs(view, data_for_prediction, self.X_train, method_name)

    def shap_knn(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_KNN(self.KNN_model_filename, self.X_train, self.Y_train)
        method_name = "KNN"

        knn = load(self.KNN_model_filename)
        view = shap.KernelExplainer(knn.predict_proba, self.X_train)

        data_for_prediction = self.X_test[chosen_instance]

        self.print_acc(knn, self.X_test, self.Y_test, method_name, data_for_prediction.reshape(1, -1))
        self.plot_graphs(view, data_for_prediction, self.X_train, method_name)

    def print_acc(self, classifier, X_test, Y_test, method_name, data_for_prediction_array):
        print("-------------------------------------------------")
        print("Prediction: ", classifier.predict_proba(data_for_prediction_array))
        print(method_name, ":")
        preds = classifier.predict(X_test)
        print(classification_report(Y_test, preds))
        print('The accuracy of this model is :\t', metrics.accuracy_score(preds, Y_test))

    def plot_graphs(self, explainer, data_for_prediction, X, method_name):
        shap_values = explainer.shap_values(data_for_prediction)
        shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                       feature_names=self.feature_names)
        shap.save_html(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/force_plot.html',
                       shap_display)

        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                               show=False,
                                               feature_names=self.feature_names)
        plt.savefig(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/waterfall.png')
        plt.clf()

        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False, feature_names=self.feature_names)
        plt.savefig(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/summary.png')
        plt.clf()


if __name__ == "__main__":
    parkinsonvoice = ParkinsonVoiceShap()

    is_need_to_create = True

    # ACC: 0.837
    parkinsonvoice.shap_svm(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # ACC: 0.898
    parkinsonvoice.shap_rfc(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # ACC: 0.857
    parkinsonvoice.shap_knn(is_need_to_create_model=is_need_to_create, chosen_instance=5)
