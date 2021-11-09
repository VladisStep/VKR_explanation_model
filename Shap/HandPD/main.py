import matplotlib.pyplot as plt
import shap
from joblib import load
from sklearn import metrics
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split

import my_utils
from Shap.HandPD.Models.ENS_model import create_ENS
from Shap.HandPD.Models.ETC_model import create_ETC
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
        self.ETC_model_filename = self.path_to_project + 'Models/ETC.joblib'
        self.dataset_name = 'HandPD'

        X, Y = load_hand_pd()
        test_size = 0.5
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*[X, Y], test_size=test_size,
                                                                                random_state=0)

    def shap(self, is_need_to_create_model, chosen_instance, create_foo, method_name, model_filename, explainer,
             isKernelExplainer):
        if is_need_to_create_model:
            create_foo(model_filename, self.X_train, self.Y_train)

        model = load(model_filename)
        if isKernelExplainer:
            expl = explainer(model.predict_proba, self.X_train.values)
        else:
            expl = explainer(model)

        # which row from data should shap show?
        data_for_prediction = self.X_test.iloc[chosen_instance]

        self.print_acc(model, self.X_test, self.Y_test, method_name, data_for_prediction.values.reshape(1, -1))
        self.plot_graphs(expl, data_for_prediction, self.X_train, method_name)

    def print_acc(self, classifier, X_test, Y_test, method_name, data_for_prediction_array):
        preds = classifier.predict(X_test.values)
        s = method_name + ':\n' + 'Prediction: ' + str(classifier.predict_proba(data_for_prediction_array)) + '\n' + \
            str(classification_report(Y_test, preds)) + '\n' + 'The accuracy of this model is :\t' \
            + str(metrics.accuracy_score(preds, Y_test)) + '\nMean Squared Error: ' \
            + str(mean_squared_error(Y_test, preds))

        f = open(self.path_to_project + 'Shap/' + self.dataset_name + '/Graphs/' + method_name + '/log.txt', 'w')
        f.write(s)
        f.close()

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

    # is_need_to_create = True
    #
    # # ACC: 0.897
    # handpdshap.shap_svm(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # # ACC: 0.94
    # handpdshap.shap_rfc(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # # ACC: 0.815
    # handpdshap.shap_knn(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # # ACC: 0.902
    # handpdshap.shap_nn(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # # ACC: 0.951
    # handpdshap.shap_ens(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    # # ACC: 0.924
    # handpdshap.shap_etc(is_need_to_create_model=is_need_to_create, chosen_instance=5)
