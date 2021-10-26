import matplotlib.pyplot as plt
import numpy as np
import shap
from joblib import load
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import my_utils
from Shap.HandPD.Models.ENS_model import create_ENS
from Shap.HandPD.Models.KNN_model import create_KNN
from Shap.HandPD.Models.NN_model import create_NN
from Shap.HandPD.Models.RFC_model import create_RFC
from Shap.HandPD.Models.SVM_model import create_SVM
from Shap.HandPD.data import load_hand_pd


class HandPDShap():
    def __init__(self):
        self.path_to_project = my_utils.path_to_project
        self.RFC_model_filename = self.path_to_project + 'Models/RFC.joblib'
        self.SVM_model_filename = self.path_to_project + 'Models/SVM.joblib'
        self.KNN_model_filename = self.path_to_project + 'Models/KNN.joblib'
        self.NN_model_filename = self.path_to_project + 'Models/NN.joblib'
        self.ENS_model_filename = self.path_to_project + 'Models/ENS.joblib'
        self.dataset_name = 'HandPD'

        X, Y = load_hand_pd()
        test_size = 0.5
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*[X, Y], test_size=test_size,
                                                                                random_state=0)

    def shap_svm(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_SVM(self.SVM_model_filename, self.X_train, self.Y_train)
        method_name = "SVM"

        svm = load(self.SVM_model_filename)
        explainer = shap.KernelExplainer(svm.predict_proba, self.X_train.values)

        # which row from data should shap show?
        data_for_prediction = self.X_test.iloc[chosen_instance]

        self.print_acc(svm, self.X_test, self.Y_test, method_name, data_for_prediction.values.reshape(1, -1))
        self.plot_graphs(explainer, data_for_prediction, self.X_train, method_name)

    def shap_rfc(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_RFC(self.RFC_model_filename, self.X_train, self.Y_train)
        method_name = "RFC"

        rfc = load(self.RFC_model_filename)
        view = shap.TreeExplainer(rfc)
        # view = shap.KernelExplainer(rfc.predict_proba, X_train.values)

        data_for_prediction = self.X_test.iloc[chosen_instance]

        self.print_acc(rfc, self.X_test, self.Y_test, method_name, data_for_prediction.values.reshape(1, -1))
        self.plot_graphs(view, data_for_prediction, self.X_train, method_name)

    def shap_knn(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_KNN(self.KNN_model_filename, self.X_train, self.Y_train)
        method_name = "KNN"

        knn = load(self.KNN_model_filename)
        view = shap.KernelExplainer(knn.predict_proba, self.X_train.values)

        data_for_prediction = self.X_test.iloc[chosen_instance]

        self.print_acc(knn, self.X_test, self.Y_test, method_name, data_for_prediction.values.reshape(1, -1))
        self.plot_graphs(view, data_for_prediction, self.X_train, method_name)

    def shap_nn(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_NN(self.NN_model_filename, self.X_train, self.Y_train)
        method_name = "NN"

        nn = load(self.NN_model_filename)
        view = shap.KernelExplainer(nn.predict_proba, self.X_train.values)

        data_for_prediction = self.X_test.iloc[chosen_instance]

        self.print_acc(nn, self.X_test, self.Y_test, method_name, data_for_prediction.values.reshape(1, -1))
        self.plot_graphs(view, data_for_prediction, self.X_train, method_name)

    def shap_ens(self, is_need_to_create_model, chosen_instance):
        if is_need_to_create_model:
            create_ENS(self.ENS_model_filename, self.X_train, self.Y_train)
        method_name = "ENS"

        ens = load(self.ENS_model_filename)
        view = shap.KernelExplainer(ens.predict_proba, self.X_train.values)

        data_for_prediction = self.X_test.iloc[chosen_instance]

        self.print_acc(ens, self.X_test, self.Y_test, method_name, data_for_prediction.values.reshape(1, -1))
        self.plot_graphs(view, data_for_prediction, self.X_train, method_name)

    def print_acc(self, classifier, X_test, Y_test, method_name, data_for_prediction_array):
        print("-------------------------------------------------")
        print("Prediction: ", classifier.predict_proba(data_for_prediction_array))
        print(method_name, ":")
        preds = classifier.predict(X_test.values)
        print(classification_report(Y_test, preds))
        print('The accuracy of this model is :\t', metrics.accuracy_score(preds, Y_test))

    def plot_graphs(self, explainer, data_for_prediction, X, method_name):
        shap_values = explainer.shap_values(data_for_prediction)
        shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
        shap.save_html(
            self.path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/force_plot.html',
            shap_display)

        shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                               show=False)
        plt.savefig(self.path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/waterfall.png')
        plt.clf()

        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(self.path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/summary.png')
        plt.clf()

        shap.dependence_plot('GENDER', shap_values[1], self.X_train, interaction_index='AGE', show=False)
        plt.savefig(self.path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/dependence.png')
        plt.clf()

        # shap.plots.heatmap(shap_values[0], show=False)
        # plt.savefig(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/heatmap_class0.png')
        # plt.clf()
        #
        # shap.plots.heatmap(shap_values[1], show=False)
        # plt.savefig(path_to_project + 'Shap/HandPD/Graphs/' + method_name + '/heatmap_class1.png')
        # plt.clf()


if __name__ == "__main__":
    handpdshap = HandPDShap()

    is_need_to_create = True

    # ACC: 0.897
    handpdshap.shap_svm(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # ACC: 0.94
    handpdshap.shap_rfc(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    ACC: 0.815
    handpdshap.shap_knn(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # ACC: 0.902
    handpdshap.shap_nn(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # ACC: 0.951
    handpdshap.shap_ens(is_need_to_create_model=is_need_to_create, chosen_instance=5)
