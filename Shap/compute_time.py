import time

import shap
from alibi.explainers import ALE
from joblib import load
from sklearn.datasets import load_iris

import my_utils
from Shap.HandPD.Models.RFC_model import create_RFC
from Shap.HandPD.Models.SVM_model import create_SVM

path_to_project = my_utils.path_to_project


def compute_time_ale(time_start, model, feature_names, target_names, X, model_str):
    ale = ALE(model.predict_proba, feature_names=feature_names, target_names=target_names)
    time_end = time.time()

    s = '----\nALE ' + model_str + ' compute time:\n' + str(time_end - time_start) + '\n'

    f = open(path_to_project + 'Shap/compute_time.txt', 'a')
    f.write(s)
    f.close()


def compute_time_kernel_explainer(time_start, model, X, model_str):
    explainer = shap.KernelExplainer(model.predict_proba, X)
    explainer.shap_values(X)
    time_end = time.time()

    s = '----\nKernel explainer ' + model_str + ' compute time:\n' + str(time_end - time_start) + '\n'

    f = open(path_to_project + 'Shap/compute_time.txt', 'a')
    f.write(s)
    f.close()


def compute_time_tree_explainer(time_start, model, X, model_str):
    explainer = shap.TreeExplainer(model)
    explainer.shap_values(X)
    time_end = time.time()

    s = '----\nTree explainer ' + model_str + ' compute time:\n' + str(time_end - time_start) + '\n'

    f = open(path_to_project + 'Shap/compute_time.txt', 'a')
    f.write(s)
    f.close()


def compute_time_without_model(model_filename, feature_names, target_names, X, method_name):
    model = load(model_filename)

    time_start = time.time()
    if method_name == 'ALE':
        compute_time_ale(time_start, model, feature_names, target_names, X, 'without model')
    elif method_name == 'KE':
        compute_time_kernel_explainer(time_start, model, X, 'without model')
    elif method_name == 'TE':
        compute_time_tree_explainer(time_start, model, X, 'without model')


def compute_time_with_model(model_filename, create_foo, feature_names, target_names, X, y, method_name):
    time_start = time.time()
    create_foo(model_filename, X, y)
    model = load(model_filename)
    if method_name == 'ALE':
        compute_time_ale(time_start, model, feature_names, target_names, X, 'with model')
    elif method_name == 'KE':
        compute_time_kernel_explainer(time_start, model, X, 'with model')
    elif method_name == 'TE':
        compute_time_tree_explainer(time_start, model, X, 'with model')


if __name__ == "__main__":
    SVM_model_filename = path_to_project + 'Models/SVM.joblib'
    RFC_model_filename = path_to_project + 'Models/RFC.joblib'

    data = load_iris()

    compute_time_with_model(RFC_model_filename, create_RFC, data.feature_names, data.target_names,
                            data.data, data.target, 'ALE')
    compute_time_without_model(RFC_model_filename, data.feature_names, data.target_names, data.data, 'ALE')

    compute_time_with_model(RFC_model_filename, create_RFC, data.feature_names, data.target_names,
                            data.data, data.target, 'KE')
    compute_time_without_model(RFC_model_filename, data.feature_names, data.target_names, data.data, 'KE')

    compute_time_with_model(RFC_model_filename, create_RFC, data.feature_names, data.target_names,
                            data.data, data.target, 'TE')
    compute_time_without_model(RFC_model_filename, data.feature_names, data.target_names, data.data, 'TE')

    # compute_time_with_model(SVM_model_filename, create_SVM, data.feature_names, data.target_names,
    #                         data.data, data.target, 'ALE')
    # compute_time_without_model(SVM_model_filename, data.feature_names, data.target_names, data.data, 'ALE')
    #
    # compute_time_with_model(SVM_model_filename, create_SVM, data.feature_names, data.target_names,
    #                         data.data, data.target, 'KE')
    # compute_time_without_model(SVM_model_filename, data.feature_names, data.target_names, data.data, 'KE')
    #
    # compute_time_with_model(SVM_model_filename, create_SVM, data.feature_names, data.target_names,
    #                         data.data, data.target, 'TE')
    # compute_time_without_model(SVM_model_filename, data.feature_names, data.target_names, data.data, 'TE')
