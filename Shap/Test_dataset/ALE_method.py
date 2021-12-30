from alibi.explainers.ale import ALE, plot_ale
from joblib import load
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import my_utils
from Shap.HandPD.Models.SVM_model import create_SVM

path_to_project = my_utils.path_to_project


class IrisALE:
    def __init__(self):
        self.path_to_project = my_utils.path_to_project
        self.dataset_name = 'Test_dataset'

        data = load_iris()
        self.feature_names = data.feature_names
        self.target_names = data.target_names
        X = data.data
        y = data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    def calculate_ale(self, is_need_to_create_model, create_foo, model_filename):
        if is_need_to_create_model:
            create_foo(model_filename, self.X_train, self.y_train)

        model = load(model_filename)

        ale = ALE(model.predict_proba, feature_names=self.feature_names, target_names=self.target_names)
        exp = ale.explain(self.X_train)

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5))

        ax = plot_ale(exp, n_cols=2, fig_kw={'figwidth': 8, 'figheight': 5}, sharey=None)

        fig.suptitle("ALE explainer")
        plt.savefig(path_to_project + 'Shap/' + self.dataset_name + '/Graphs/ale_test.png')
        plt.clf()


if __name__ == "__main__":
    ale = IrisALE()

    NN_model_filename = path_to_project + 'Models/NN.joblib'
    SVM_model_filename = path_to_project + 'Models/SVM.joblib'

    create_model = True
    ale.calculate_ale(create_model, create_SVM, SVM_model_filename)
