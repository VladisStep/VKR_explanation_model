import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import shap
import sklearn.exceptions
from joblib import load
from sklearn.model_selection import train_test_split

import my_utils
from Global.Parkinson_voice.data import load_parkinson_voice

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


class ParkinsonVoiceShap:
    def __init__(self):
        self.path_to_project = my_utils.path_to_project
        self.dataset_name = 'Parkinson_voice'

        X, Y, self.feature_names, self.target_names = load_parkinson_voice()
        test_size = 0.5
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*[X, Y], test_size=test_size,
                                                                                random_state=0)

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

        data_for_prediction = self.X_test[chosen_instance]

        my_utils.shap_log(model, self.X_test, self.Y_test, model_name, method_name,
                          data_for_prediction.reshape(1, -1), my_utils.PATH_TO_PARKINSON_LOG, time_start, time_end)
        self.plot_graphs(expl, data_for_prediction, self.X_train, model_name, method_name)

    def plot_graphs(self, explainer, data_for_prediction, X, model_name, method_name):
        shap_values = explainer.shap_values(data_for_prediction)
        shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                       feature_names=self.feature_names)
        shap.save_html(my_utils.PATH_TO_PARKINSON_LOG + method_name + '/' + model_name + '/force_plot.html',
                       shap_display)

        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False, feature_names=self.feature_names, plot_size=[15, 7])
        plt.title(method_name + "; " + model_name)
        plt.savefig(my_utils.PATH_TO_PARKINSON_LOG + method_name + '/' + model_name + '/summary.png')
        plt.clf()

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 12))
        feature_ind = 0

        for j in range(axes.shape[0]):
            for k in range(axes.shape[1]):
                for i in range(len(shap_values)):
                    x_ax = X[:, feature_ind]
                    y_ax = shap_values[i][:, feature_ind]
                    d = 2
                    theta = np.polyfit(x_ax, y_ax, deg=d)
                    model = np.poly1d(theta)
                    if j == 0 and k == 0:
                        axes[j, k].plot(x_ax, model(x_ax), label=list(self.target_names)[i], linestyle='dashdot')
                    else:
                        axes[j, k].plot(x_ax, model(x_ax), linestyle='dashdot')

                axes[j, k].set_ylabel('SHAP')
                axes[j, k].set_xlabel(self.feature_names[feature_ind])
                axes[j, k].legend()
                feature_ind = feature_ind + 1

        fig.suptitle(method_name + '; ' + model_name)
        plt.savefig(my_utils.PATH_TO_PARKINSON_LOG + method_name + '/' + model_name + '/as_ALE.png')
        plt.clf()

        # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], data_for_prediction,
        #                                        show=False, feature_names=self.feature_names)
        # plt.savefig(my_utils.PATH_TO_PARKINSON_LOG + method_name + '/' + model_name + '/waterfall.png')
        # plt.clf()
