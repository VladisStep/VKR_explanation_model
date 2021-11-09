import time

import shap

import my_utils
from Shap.HandPD.Models.ENS_model import create_ENS
from Shap.HandPD.Models.ETC_model import create_ETC
from Shap.HandPD.Models.KNN_model import create_KNN
from Shap.HandPD.Models.NN_model import create_NN
from Shap.HandPD.Models.RFC_model import create_RFC
from Shap.HandPD.Models.SVM_model import create_SVM
from Shap.HandPD.main import HandPDShap
from Shap.Parkinson_voice.main import ParkinsonVoiceShap

path_to_project = my_utils.path_to_project


def print_time(time_start, time_end, filename):
    f = open(filename, 'a')
    f.write('\nTime spent, seconds: ' + str(time_end - time_start))
    f.close()


class MyShap:
    def __init__(self, dataset_obj):
        self.path_to_project = my_utils.path_to_project
        self.RFC_model_filename = self.path_to_project + 'Models/RFC.joblib'
        self.SVM_model_filename = self.path_to_project + 'Models/SVM.joblib'
        self.KNN_model_filename = self.path_to_project + 'Models/KNN.joblib'
        self.NN_model_filename = self.path_to_project + 'Models/NN.joblib'
        self.ENS_model_filename = self.path_to_project + 'Models/ENS.joblib'
        self.ETC_model_filename = self.path_to_project + 'Models/ETC.joblib'
        self.dataset_obj = dataset_obj

    def shap_svm(self, is_need_to_create_model, chosen_instance):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_SVM, "SVM", self.SVM_model_filename,
                              shap.KernelExplainer, True)

    def shap_rfc(self, is_need_to_create_model, chosen_instance):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_RFC, "RFC", self.RFC_model_filename,
                              shap.TreeExplainer, False)

    def shap_knn(self, is_need_to_create_model, chosen_instance):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_KNN, "KNN", self.KNN_model_filename,
                              shap.KernelExplainer, True)

    def shap_nn(self, is_need_to_create_model, chosen_instance):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_NN, "NN", self.NN_model_filename,
                              shap.KernelExplainer, True)

    def shap_ens(self, is_need_to_create_model, chosen_instance):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_ENS, "ENS", self.ENS_model_filename,
                              shap.KernelExplainer, True)

    def shap_etc(self, is_need_to_create_model, chosen_instance):
        self.dataset_obj.shap(is_need_to_create_model, chosen_instance, create_ETC, "ETC", self.ETC_model_filename,
                              shap.TreeExplainer, False)

    def shap(self, is_need_to_create_model, chosen_instance):
        time_start = time.time()
        self.shap_svm(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance)
        time_end = time.time()
        print_time(time_start, time_end, self.path_to_project + 'Shap/' + self.dataset_obj.dataset_name
                   + '/Graphs/SVM/log.txt')

        time_start = time.time()
        self.shap_rfc(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance)
        time_end = time.time()
        print_time(time_start, time_end, self.path_to_project + 'Shap/' + self.dataset_obj.dataset_name
                   + '/Graphs/RFC/log.txt')

        time_start = time.time()
        self.shap_knn(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance)
        time_end = time.time()
        print_time(time_start, time_end, self.path_to_project + 'Shap/' + self.dataset_obj.dataset_name
                   + '/Graphs/KNN/log.txt')

        time_start = time.time()
        self.shap_nn(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance)
        time_end = time.time()
        print_time(time_start, time_end, self.path_to_project + 'Shap/' + self.dataset_obj.dataset_name
                   + '/Graphs/NN/log.txt')

        time_start = time.time()
        self.shap_ens(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance)
        time_end = time.time()
        print_time(time_start, time_end, self.path_to_project + 'Shap/' + self.dataset_obj.dataset_name
                   + '/Graphs/ENS/log.txt')

        time_start = time.time()
        self.shap_etc(is_need_to_create_model=is_need_to_create_model, chosen_instance=chosen_instance)
        time_end = time.time()
        print_time(time_start, time_end, self.path_to_project + 'Shap/' + self.dataset_obj.dataset_name
                   + '/Graphs/ETC/log.txt')


if __name__ == "__main__":
    is_need_to_create = True
    instance = 5

    handpd = HandPDShap()
    myshap = MyShap(handpd)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance)

    parkinson_voice = ParkinsonVoiceShap()
    myshap = MyShap(parkinson_voice)
    myshap.shap(is_need_to_create_model=is_need_to_create, chosen_instance=instance)
