from alibi.explainers.ale import ALE, plot_ale
from joblib import load
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import my_utils
from Global.Parkinson_voice.data import load_parkinson_voice


class ParkinsonALE:
    def __init__(self):
        self.dataset_name = 'Parkinson_voice'
        self.method_name = 'ALE'

        X, Y, self.feature_names, self.target_names = load_parkinson_voice()
        test_size = 0.5
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(*[X, Y], test_size=test_size,
                                                                                random_state=0)

    def calculate_ale(self, is_need_to_create_model, create_foo, model_name, model_filename):
        if is_need_to_create_model:
            create_foo(model_filename, self.X_train, self.Y_train)

        model = load(model_filename)

        ale = ALE(model.predict_proba, feature_names=self.feature_names, target_names=list(self.target_names))
        exp = ale.explain(self.X_train)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))

        ax = plot_ale(exp, n_cols=2, fig_kw={'figwidth': 16, 'figheight': 10}, sharey=None)

        fig.suptitle(self.method_name + '; ' + model_name)
        plt.savefig(my_utils.PATH_TO_PARKINSON_LOG + self.method_name + '/' + model_name + '/plot_ale.png')
        plt.clf()
