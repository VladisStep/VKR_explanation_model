import math
import time
import warnings

import numpy as np
import pandas as pd
import shap
import sklearn.exceptions
from joblib import load
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import my_utils
from Global.Test_dataset.data_football import load_fifa2018_stat
from Global.Test_dataset.data_heart_failure import load_heart_failure

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class TestDatasetShap:
    def __init__(self):
        self.dataset_name = 'Test_dataset'

    def iris(self):
        self.dataset_name = self.dataset_name + '/Iris'
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2,
                                                                                random_state=0)
        self.feature_names = self.X_test.columns.values
        self.target_names = load_iris().target_names
        self.path_to_log = my_utils.PATH_TO_IRIS
        self.nrows = 2
        self.ncols = 2
        self.figwidth = 10
        self.figheight = 7

    def wine(self):
        self.dataset_name = self.dataset_name + '/Wine'
        dataset = datasets.load_wine()
        X = dataset['data']
        Y = dataset['target']
        self.feature_names = dataset['feature_names']
        self.target_names = list(set(Y))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        self.path_to_log = my_utils.PATH_TO_WINE
        self.nrows = 5
        self.ncols = 3
        self.figwidth = 18
        self.figheight = 14

    def football(self):
        self.dataset_name = self.dataset_name + '/Football'
        X, Y, self.feature_names = load_fifa2018_stat()
        self.target_names = list(set(Y))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        self.path_to_log = my_utils.PATH_TO_FOOTBALL
        self.nrows = 9
        self.ncols = 2
        self.figwidth = 14
        self.figheight = 28

    def heart_failure(self):
        self.dataset_name = self.dataset_name + '/Heart_failure'
        X, Y, self.feature_names = load_heart_failure()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        # self.target_names = list(set(Y))
        self.target_names = ["not dead", "dead"]
        self.path_to_log = my_utils.PATH_TO_HEART_FAILURE
        self.nrows = 2
        self.ncols = 6
        self.figwidth = 30
        self.figheight = 8

    def shap(self, is_need_to_create_model, chosen_instance, create_foo, model_name, model_filename, explainer,
             isKernelExplainer):

        time_start = None
        time_end = None
        if is_need_to_create_model:
            time_start = time.time()
            create_foo(model_filename, self.X_train, self.Y_train)
            time_end = time.time()

        model = load(model_filename)
        if isKernelExplainer:
            method_name = 'KernelExplainer'
            expl = explainer(model.predict_proba, self.X_train)
        else:
            method_name = 'TreeExplainer'
            expl = explainer(model)

        if isinstance(self.X_test, pd.DataFrame):
            data_for_prediction = self.X_test.values[chosen_instance]
        else:
            data_for_prediction = self.X_test[chosen_instance]
        # print("Real: ", self.Y_test[chosen_instance])

        my_utils.shap_log(model, self.X_test, self.Y_test, model_name, method_name,
                          data_for_prediction.reshape(1, -1), self.path_to_log, time_start, time_end)
        self.plot_graphs(expl, data_for_prediction, self.X_train, model_name, method_name)

    def plot_graphs(self, explainer, data_for_prediction, X, model_name, method_name):
        shap_values = explainer.shap_values(data_for_prediction)
        shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                       feature_names=self.feature_names)
        shap.save_html(self.path_to_log + method_name + '/' + model_name + '/force_plot.html',
                       shap_display)

        shap_values = explainer.shap_values(X)
        # FOR EACH CLASS
        # shap.summary_plot(shap_values[1], X, show=False, feature_names=self.feature_names, plot_size=[15, 7])
        # FOR ALL CLASSES
        shap.summary_plot(shap_values, X, show=False, feature_names=self.feature_names, plot_size=[16, 7])
        plt.title(method_name + "; " + model_name)
        plt.savefig(self.path_to_log + method_name + '/' + model_name + '/summary.png')
        plt.clf()

        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(self.figwidth, self.figheight))
        feature_ind = 0

        for j in range(axes.shape[0]):
            for k in range(axes.shape[1]):
                if feature_ind > len(self.feature_names) - 1:
                    break

                for i in range(len(shap_values)):
                    if isinstance(self.X_test, pd.DataFrame):
                        x_ax = X.values[:, feature_ind]
                    else:
                        x_ax = X[:, feature_ind]
                    y_ax = shap_values[i][:, feature_ind]
                    d = 2
                    theta = np.polyfit(x_ax, y_ax, deg=d)
                    model = np.poly1d(theta)
                    if j == 0 and k == 0:
                        axes[j, k].plot(x_ax, model(x_ax), label=self.target_names[i], linestyle='dashdot')
                    else:
                        axes[j, k].plot(x_ax, model(x_ax), linestyle='dashdot')

                axes[j, k].set_ylabel('SHAP')
                axes[j, k].set_xlabel(self.feature_names[feature_ind])
                axes[j, k].legend()
                feature_ind = feature_ind + 1

        fig.suptitle(method_name + '; ' + model_name)
        plt.savefig(self.path_to_log + method_name + '/' + model_name + '/as_ALE.png')
        plt.clf()
