from alibi.explainers.ale import ALE, plot_ale
from joblib import load
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import my_utils
from Global.Models.SVM_model import create_SVM
from Global.Test_dataset.data_football import load_fifa2018_stat
from Global.Test_dataset.data_heart_failure import load_heart_failure

path_to_project = my_utils.path_to_project


class TestALE:
    def __init__(self):
        self.dataset_name = 'Test_dataset'

    def iris(self):
        self.dataset_name = self.dataset_name + '/Iris'
        data = load_iris()
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        X = data.data
        y = data.target
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        self.path_to_log = my_utils.PATH_TO_IRIS
        self.nrows = 2
        self.ncols = 2
        self.figwidth = 8
        self.figheight = 5

    def wine(self):
        self.dataset_name = self.dataset_name + '/Wine'
        dataset = datasets.load_wine()
        X = dataset['data']
        Y = dataset['target']
        self.feature_names = dataset['feature_names']
        self.target_names = set(Y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        self.path_to_log = my_utils.PATH_TO_WINE
        self.nrows = 5
        self.ncols = 3
        self.figwidth = 16
        self.figheight = 12

    def football(self):
        self.dataset_name = self.dataset_name + '/Football'
        X, Y, self.feature_names = load_fifa2018_stat()
        self.target_names = set(Y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.path_to_log = my_utils.PATH_TO_FOOTBALL
        self.nrows = 6
        self.ncols = 3
        self.figwidth = 16
        self.figheight = 12

    def heart_failure(self):
        self.dataset_name = self.dataset_name + '/Heart_failure'
        X, Y, self.feature_names = load_heart_failure()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, random_state=0)
        self.target_names = set(Y)
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.path_to_log = my_utils.PATH_TO_HEART_FAILURE
        self.nrows = 4
        self.ncols = 3
        self.figwidth = 14
        self.figheight = 10

    def calculate_ale(self, is_need_to_create_model, create_foo, model_name, model_filename):
        if is_need_to_create_model:
            create_foo(model_filename, self.X_train, self.Y_train)

        model = load(model_filename)

        ale = ALE(model.predict_proba, feature_names=self.feature_names, target_names=list(self.target_names))
        exp = ale.explain(self.X_train)

        fig, ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(self.figwidth, self.figheight))

        ax = plot_ale(exp, n_cols=self.ncols, fig_kw={'figwidth': self.figwidth, 'figheight': self.figheight},
                      sharey=None)

        fig.suptitle("ALE explainer")
        plt.savefig(self.path_to_log + 'ALE/' + model_name + '/ale.png')
        plt.clf()


if __name__ == "__main__":
    ale = TestALE()

    SVM_model_filename = path_to_project + 'Global/Trained_models/SVM.joblib'

    create_model = True
    ale.calculate_ale(create_model, create_SVM, SVM_model_filename)
