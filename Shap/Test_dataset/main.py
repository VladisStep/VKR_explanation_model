import time
import warnings

from matplotlib import pyplot as plt
import shap
import sklearn.exceptions
from joblib import load
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sklearn.datasets as datasets

import my_utils
from Shap.HandPD.Models.ENS_model import create_ENS
from Shap.HandPD.Models.ETC_model import create_ETC
from Shap.HandPD.Models.KNN_model import create_KNN
from Shap.HandPD.Models.NN_model import create_NN
from Shap.HandPD.Models.RFC_model import create_RFC
from Shap.HandPD.Models.SVM_model import create_SVM
from Shap.Test_dataset.data_football import load_fifa2018_stat
from Shap.Test_dataset.data_heart_failure import load_heart_failure

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

path_to_project = my_utils.path_to_project


class TestDatasetShap:
    def __init__(self):
        self.path_to_project = my_utils.path_to_project
        self.dataset_name = 'Test_dataset'

        # WINE DATASET
        # dataset = datasets.load_wine()
        # X = dataset['data']
        # Y = dataset['target']
        # self.feature_names = dataset['feature_names']
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)

        # IRIS DATASET
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2,
        #                                                                         random_state=0)
        # self.feature_names = self.X_test.columns.values
        # self.target_names = load_iris().target_names

        # FOOTBALL DATASET
        # X, Y, self.feature_names = load_fifa2018_stat()
        # self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)

        # HEART FAILURE
        X, Y, self.feature_names = load_heart_failure()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        self.target_names = [0, 1]

    def shap(self, is_need_to_create_model, chosen_instance, create_foo, method_name, model_filename, explainer,
             isKernelExplainer):

        if is_need_to_create_model:
            create_foo(model_filename, self.X_train, self.Y_train)

        model = load(model_filename)
        if isKernelExplainer:
            # expl = explainer(model, self.X_train)
            expl = explainer(model.predict_proba, self.X_train)
        else:
            expl = explainer(model)

        # which row from data should shap show?
        data_for_prediction = self.X_test.values[chosen_instance]
        # data_for_prediction = self.X_test[chosen_instance]
        # print("Real: ", self.Y_test[chosen_instance])

        self.print_acc(model, self.X_test, self.Y_test, method_name, data_for_prediction.reshape(1, -1))
        self.plot_graphs(expl, data_for_prediction, self.X_train, method_name)

    def print_acc(self, classifier, X_test, Y_test, method_name, data_for_prediction_array):
        preds = classifier.predict(X_test)
        s = method_name + ':\n' + 'Prediction: ' + str(classifier.predict_proba(data_for_prediction_array)) + '\n' + \
            str(classification_report(Y_test, preds)) + '\n' + 'The accuracy of this model is :\t' \
            + str(metrics.accuracy_score(preds, Y_test)) + '\nMean Squared Error: ' \
            + str(mean_squared_error(Y_test, preds))

        f = open(self.path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/log.txt', 'w')
        f.write(s)
        f.close()

    def plot_graphs(self, explainer, data_for_prediction, X, method_name):
        shap_values = explainer.shap_values(data_for_prediction)
        # print("Shap_values: ", shap_values)
        # print(explainer(X).compute_time)
        shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                       feature_names=self.feature_names)
        shap.save_html(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/force_plot.html',
                       shap_display)

        shap_values = explainer.shap_values(X)
        # FOR EACH CLASS
        # shap.summary_plot(shap_values[1], X, show=False, feature_names=self.feature_names, plot_size=[15, 7])
        # FOR ALL CLASSES
        shap.summary_plot(shap_values, X, show=False, feature_names=self.feature_names, plot_size=[15, 7])
        plt.savefig(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/summary.png')
        plt.clf()

        # IRIS:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
        feature_ind = 0

        for j in range(axes.shape[0]):
            for k in range(axes.shape[1]):
                for i in range(len(shap_values)):
                    x_ax = X.values[:, feature_ind]
                    y_ax = shap_values[i][:, feature_ind]
                    if j == 0 and k == 0:
                        axes[j, k].plot(x_ax, y_ax, label=self.target_names[i], linestyle='dashdot')
                    else:
                        axes[j, k].plot(x_ax, y_ax, linestyle='dashdot')

                axes[j, k].set_ylabel('SHAP')
                axes[j, k].set_xlabel(self.feature_names[feature_ind])
                axes[j, k].legend()
                feature_ind = feature_ind + 1

        fig.suptitle("SHAP explainer")
        plt.savefig(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/shap_test.png')
        plt.clf()


if __name__ == "__main__":
    testDataset = TestDatasetShap()

    is_need_to_create_model = True
    chosen_instance = 5

    RFC_model_filename = path_to_project + 'Models/RFC.joblib'
    SVM_model_filename = path_to_project + 'Models/SVM.joblib'
    KNN_model_filename = path_to_project + 'Models/KNN.joblib'
    NN_model_filename = path_to_project + 'Models/NN.joblib'
    ENS_model_filename = path_to_project + 'Models/ENS.joblib'
    ETC_model_filename = path_to_project + 'Models/ETC.joblib'

    # SVM. TIME: 5.228054761886597
    # LinearExplainer
    time_start = time.time()
    testDataset.shap(is_need_to_create_model, chosen_instance, create_SVM, "SVM", SVM_model_filename,
                     shap.KernelExplainer, True)
    print('Time spent: {0}'.format(time.time() - time_start))

    # RFC. TIME: 1.1130003929138184
    time_start = time.time()
    testDataset.shap(is_need_to_create_model, chosen_instance, create_RFC, "RFC", RFC_model_filename,
                     shap.TreeExplainer, False)
    print('Time spent: {0}'.format(time.time() - time_start))

    # KNN. TIME: 5.644734621047974
    # PartitionExplainer
    time_start = time.time()
    testDataset.shap(is_need_to_create_model, chosen_instance, create_KNN, "KNN", KNN_model_filename,
                     shap.KernelExplainer, True)
    print('Time spent: {0}'.format(time.time() - time_start))

    # NN. TIME: 8.47510313987732
    time_start = time.time()
    testDataset.shap(is_need_to_create_model, chosen_instance, create_NN, "NN", NN_model_filename,
                     shap.SamplingExplainer, True)
    print('Time spent: {0}'.format(time.time() - time_start))

    # ENS. TIME: 3.95400071144104
    # PermutationExplainer
    time_start = time.time()
    testDataset.shap(is_need_to_create_model, chosen_instance, create_ENS, "ENS", ENS_model_filename,
                     shap.KernelExplainer, True)
    print('Time spent: {0}'.format(time.time() - time_start))

    # ETC. TIME: 1.0870003700256348
    time_start = time.time()
    testDataset.shap(is_need_to_create_model, chosen_instance, create_ETC, "ETC", ETC_model_filename,
                     shap.TreeExplainer, False)
    print('Time spent: {0}'.format(time.time() - time_start))
