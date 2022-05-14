import time

import shap

import my_utils
from Global.Models.BAG_model import create_BAG
from Global.Models.ETC_model import create_ETC
from Global.Models.KNN_model import create_KNN
from Global.Models.NBC_model import create_NBC
from Global.Models.NN_model import create_NN
from Global.Models.RFC_model import create_RFC
from Global.Models.SVM_model import create_SVM
from Global.HandPD.main import HandPDShap
from Global.Parkinson_voice.main import ParkinsonVoiceShap
from Global.Test_dataset.main import TestDatasetShap

path_to_project = my_utils.path_to_project


def print_time(time_start, time_end, filename):
    f = open(filename, 'a')
    f.write('\nTime spent training the model and explaining the result, seconds: ' + str(time_end - time_start))
    f.close()


class MyShap:
    def __init__(self, dataset_obj):
        self.path_to_project = my_utils.path_to_project
        self.RFC_model_filename = self.path_to_project + 'Global/Trained_models/RFC.joblib'
        self.SVM_model_filename = self.path_to_project + 'Global/Trained_models/SVM.joblib'
        self.KNN_model_filename = self.path_to_project + 'Global/Trained_models/KNN.joblib'
        self.NN_model_filename = self.path_to_project + 'Global/Trained_models/NN.joblib'
        self.ENS_model_filename = self.path_to_project + 'Global/Trained_models/BAG.joblib'
        self.ETC_model_filename = self.path_to_project + 'Global/Trained_models/ETC.joblib'
        self.NBC_model_filename = self.path_to_project + 'Global/Trained_models/NBC.joblib'
        self.dataset_obj = dataset_obj

    def shap_svm(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_SVM, "SVM", self.SVM_model_filename,
                              explainer, is_kernel_explainer)

    def shap_rfc(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_RFC, "RFC", self.RFC_model_filename,
                              explainer, is_kernel_explainer)

    def shap_knn(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_KNN, "KNN", self.KNN_model_filename,
                              explainer, is_kernel_explainer)

    def shap_nn(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_NN, "NN", self.NN_model_filename,
                              explainer, is_kernel_explainer)

    def shap_bag(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_BAG, "BAG", self.ENS_model_filename,
                              explainer, is_kernel_explainer)

    def shap_etc(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_ETC, "ETC", self.ETC_model_filename,
                              explainer, is_kernel_explainer)

    def shap_nbc(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_NBC, "NBC", self.NBC_model_filename,
                              explainer, is_kernel_explainer)

    def shap(self, is_need_to_create_model, chosen_instance, explainer, is_kernel_explainer):
        if is_kernel_explainer:
            method_name = 'KernelExplainer'
        else:
            method_name = 'TreeExplainer'

        if is_kernel_explainer:  # TreeExplainer doesn't support SVM
            print('SVM')
            time_start = time.time()
            self.shap_svm(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
                          explainer=explainer, is_kernel_explainer=is_kernel_explainer)
            time_end = time.time()
            print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                       + '/Log/' + method_name + '/SVM/log.txt')

        print('RFC')
        time_start = time.time()
        self.shap_rfc(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
                      explainer=explainer, is_kernel_explainer=is_kernel_explainer)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + method_name + '/RFC/log.txt')

        if is_kernel_explainer:  # TreeExplainer doesn't support KNN
            print('KNN')
            time_start = time.time()
            self.shap_knn(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
                          explainer=explainer, is_kernel_explainer=is_kernel_explainer)
            time_end = time.time()
            print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                       + '/Log/' + method_name + '/KNN/log.txt')

        if is_kernel_explainer:  # TreeExplainer doesn't support NN
            print('NN')
            time_start = time.time()
            self.shap_nn(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
                         explainer=explainer, is_kernel_explainer=is_kernel_explainer)
            time_end = time.time()
            print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                       + '/Log/' + method_name + '/NN/log.txt')

        # ------------- НЕ ИСПОЛЬЗУЕТСЯ --------------
        # if is_kernel_explainer:  # TreeExplainer doesn't support BAG
        #     print('BAG')
        #     time_start = time.time()
        #     self.shap_bag(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
        #                   explainer=explainer, is_kernel_explainer=is_kernel_explainer)
        #     time_end = time.time()
        #     print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
        #                + '/Log/' + method_name + '/BAG/log.txt')

        print('ETC')
        time_start = time.time()
        self.shap_etc(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
                      explainer=explainer, is_kernel_explainer=is_kernel_explainer)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + method_name + '/ETC/log.txt')

        if is_kernel_explainer:  # TreeExplainer doesn't support NBC
            print('NBC')
            time_start = time.time()
            self.shap_nbc(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance,
                          explainer=explainer, is_kernel_explainer=is_kernel_explainer)
            time_end = time.time()
            print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                       + '/Log/' + method_name + '/NBC/log.txt')


if __name__ == "__main__":
    is_need_to_create = True
    instance = 5

    # -------------- HAND PD --------------
    handpd = HandPDShap()
    myshap = MyShap(handpd)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.KernelExplainer,
                is_kernel_explainer=True)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.TreeExplainer,
                is_kernel_explainer=False)

    # -------------- PARKINSON VOICE --------------
    parkinson_voice = ParkinsonVoiceShap()
    myshap = MyShap(parkinson_voice)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.KernelExplainer,
                is_kernel_explainer=True)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.TreeExplainer,
                is_kernel_explainer=False)

    # -------------- IRIS --------------
    test_dataset = TestDatasetShap()
    test_dataset.iris()
    myshap = MyShap(test_dataset)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.KernelExplainer,
                is_kernel_explainer=True)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.TreeExplainer,
                is_kernel_explainer=False)

    # -------------- WINE --------------
    test_dataset = TestDatasetShap()
    test_dataset.wine()
    myshap = MyShap(test_dataset)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.KernelExplainer,
                is_kernel_explainer=True)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.TreeExplainer,
                is_kernel_explainer=False)

    # -------------- FOOTBALL --------------
    test_dataset = TestDatasetShap()
    test_dataset.football()
    myshap = MyShap(test_dataset)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.KernelExplainer,
                is_kernel_explainer=True)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.TreeExplainer,
                is_kernel_explainer=False)

    # -------------- HEART FAILURE --------------
    test_dataset = TestDatasetShap()
    test_dataset.heart_failure()
    myshap = MyShap(test_dataset)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.KernelExplainer,
                is_kernel_explainer=True)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance, explainer=shap.TreeExplainer,
                is_kernel_explainer=False)
