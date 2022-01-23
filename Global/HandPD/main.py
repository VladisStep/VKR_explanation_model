import time

import matplotlib.pyplot as plt
import shap
from joblib import load
from sklearn.model_selection import train_test_split

import my_utils
from Global.HandPD.data import load_hand_pd


class HandPDShap:
    def __init__(self):
        self.dataset_name = 'HandPD'

        X, Y, self.feature_names, self.target_names = load_hand_pd()
        test_size = 0.5
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*[X, Y], test_size=test_size,
                                                                                random_state=0)

    def shap(self, is_need_to_create_model, chosen_instance, create_foo, model_name, model_filename, explainer,
             is_kernel_explainer):

        time_start = None
        time_end = None
        if is_need_to_create_model:
            time_start = time.time()
            create_foo(model_filename, self.X_train, self.Y_train)
            time_end = time.time()

        model = load(model_filename)
        if is_kernel_explainer:  # KernelExplainer
            method_name = 'KernelExplainer'
            expl = explainer(model.predict_proba, self.X_train.values)
        else:  # TreeExplainer
            method_name = 'TreeExplainer'
            expl = explainer(model)

        # which row from data should shap show?
        data_for_prediction = self.X_test.iloc[chosen_instance]

        my_utils.shap_log(model, self.X_test, self.Y_test, model_name, method_name,
                          data_for_prediction.values.reshape(1, -1), my_utils.PATH_TO_HANDPD_LOG, time_start, time_end)
        self.plot_graphs(expl, data_for_prediction, self.X_train, model_name, method_name)

    def plot_graphs(self, explainer, data_for_prediction, X, model_name, method_name):
        shap_values = explainer.shap_values(data_for_prediction)
        shap_display = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
        shap.save_html(my_utils.PATH_TO_HANDPD_LOG + method_name + '/' + model_name + '/force_plot.html', shap_display)

        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, show=False, plot_size=[15, 7])
        plt.title(method_name + "; " + model_name)
        plt.savefig(my_utils.PATH_TO_HANDPD_LOG + method_name + '/' + model_name + '/summary.png')
        plt.clf()

        shap.dependence_plot('GENDER', shap_values[1], self.X_train, interaction_index='AGE', show=False)
        plt.title(method_name + "; " + model_name)
        plt.savefig(my_utils.PATH_TO_HANDPD_LOG + method_name + '/' + model_name + '/dependence.png')
        plt.clf()

        # shap.plots.heatmap(shap_values[0], show=False)
        # plt.savefig(my_utils.PATH_TO_HANDPD_LOG + method_name + '/' + model_name + '/heatmap_class0.png')
        # plt.clf()
        #
        # shap.plots.heatmap(shap_values[1], show=False)
        # plt.savefig(my_utils.PATH_TO_HANDPD_LOG + method_name + '/' + model_name + '/heatmap_class1.png')
        # plt.clf()
        #
        # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[0], data_for_prediction,
        #                                        show=False)
        # plt.savefig(my_utils.PATH_TO_HANDPD_LOG + method_name + '/' + model_name + '/waterfall.png')
        # plt.clf()
