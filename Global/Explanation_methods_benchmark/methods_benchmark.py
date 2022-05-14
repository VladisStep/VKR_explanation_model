import numpy
import shap
import shap.benchmark
import shap.maskers
from alibi.explainers import ALE
from joblib import load
from matplotlib import pyplot as plt
from shap.benchmark.experiments import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import my_utils
from Global.Models.ETC_model import create_ETC


def score_map(true, pred):
    """ Computes local accuracy as the normalized standard deviation of numerical scores.
    """
    return np.std(pred - true) / (np.std(true) + 1e-6)


def to_array(*args):
    # REMEMBER THIS
    return [a.values if str(type(a)).endswith("'pandas.core.frame.DataFrame'>") else a for a in args]


def strip_list(attrs):
    """ This assumes that if you have a list of outputs you just want the second one (the second class is the '1' class)
    """
    if isinstance(attrs, list):
        return attrs[1]
    else:
        return attrs


def local_accuracy(X_train, X_test, attr_test, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = trained_model.predict(X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))


def bench(is_need_to_create, model_filename):
    data = load_iris()
    # X_train, X_test, Y_train, Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    feature_names = data.feature_names
    target_names = data.target_names
    X = data.data
    y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # X_eval = X_test.values[:]
    X_eval = X_test[:]
    y_eval = Y_test[:]

    if is_need_to_create:
        create_ETC(model_filename, X_train, Y_train)

    model = load(model_filename)
    masker = shap.maskers.Independent(data=X_eval)

    explainers = [
        ("KernelExplainer", shap.KernelExplainer(model.predict_proba, X_train, masker=masker)),
        ("TreeExplainer", shap.TreeExplainer(model, masker=masker))
    ]
    results = {}
    explanation_error_arr = []
    local_accuracy_arr = []
    ind = 0

    for name, exp in explainers:
        shap_values = exp.shap_values(X_eval)
        a = shap_values[ind]
        smasker = shap.benchmark.ExplanationError(masker, model.predict, shap_values[ind])
        explanation_error_arr.append(smasker(shap_values[ind], name=name))
        local_accuracy_arr.append(
            shap.benchmark.BenchmarkResult(
                "local accuracy", name, value=local_accuracy(X_train, X_test, shap_values[ind], score_map, model)
            )
        )

    ale = ALE(model.predict_proba, feature_names=feature_names, target_names=target_names)
    exp = ale.explain(X_train)
    # ale_values = exp.ale_values
    # tmp = []
    # for j in range(len(ale_values)):
    #     tmp2 = []
    #     for m in range(ale_values[j].shape[0]):
    #         tmp2.append((ale_values[j])[m][ind])
    #     tmp.append(tmp2)
    # new_ale_values = []
    # min_len = len(tmp[0])
    # for j in tmp:
    #     if len(j) < min_len:
    #         min_len = len(j)
    # for j in range(min_len):
    #     tmp2 = []
    #     for m in range(len(tmp)):
    #         tmp2.append(tmp[m][j])
    #     new_ale_values.append(tmp2)
    # new_ale_values = np.array(new_ale_values)
    # smasker = shap.benchmark.ExplanationError(masker, model.predict, new_ale_values)
    # explanation_error_arr.append(smasker(new_ale_values, name='ALE'))
    # local_accuracy_arr.append(
    #     shap.benchmark.BenchmarkResult(
    #         "local accuracy", 'ALE', value=local_accuracy(X_train, X_test[:min_len], new_ale_values, score_map, model)
    #     )
    # )
    a0 = exp.ale_values[0]
    a1 = exp.ale_values[1]
    a2 = exp.ale_values[2]
    a3 = exp.ale_values[3]

    max_shape = [0, 0]
    for a in [a0, a1, a2, a3]:
        if max_shape[0] < a.shape[0]:
            max_shape[0] = a.shape[0]
        if max_shape[1] < a.shape[1]:
            max_shape[1] = a.shape[1]
    arrays = []
    for a in [a0, a1, a2, a3]:
        arrays.append(np.pad(a, pad_width=((0, max_shape[0] - a.shape[0]),
                                           (0, max_shape[1] - a.shape[1])),
                             mode='constant'))
    stacked_array = np.stack(arrays)
    ale_values = np.transpose(stacked_array, [2, 1, 0])
    smasker = shap.benchmark.ExplanationError(masker, model.predict, ale_values[ind])
    explanation_error_arr.append(smasker(ale_values[ind], name='ALE'))
    local_accuracy_arr.append(
        shap.benchmark.BenchmarkResult(
            "local accuracy", 'ALE', value=local_accuracy(X_train, X_test[:14], ale_values[ind], score_map, model)
        )
    )

    """   This benchmark metric measures the discrepancy between the output of the model predicted by an
       attribution explanation vs. the actual output of the model.  """
    results["explanation error"] = explanation_error_arr
    results["local accuracy"] = local_accuracy_arr

    num_plot_rows = len(results) // 2 + len(results) % 2
    fig, ax = plt.subplots(num_plot_rows, 2, figsize=(12, 5 * num_plot_rows))

    for i, k in enumerate(results):
        plt.subplot(num_plot_rows, 2, i + 1)
        shap.plots.benchmark(results[k], show=False)

    plt.tight_layout()
    plt.savefig(my_utils.PATH_TO_METHODS_BENCHMARK + 'methods_benchmark.png')
    plt.clf()


if __name__ == "__main__":
    create = True
    BAG_model_filename = my_utils.path_to_project + 'Global/Trained_models/BAG.joblib'

    bench(create, BAG_model_filename)
