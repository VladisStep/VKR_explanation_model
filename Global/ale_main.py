import time

import my_utils
from Global.HandPD.ALE_method import HandPdALE
from Global.Models.BAG_model import create_BAG
from Global.Models.ETC_model import create_ETC
from Global.Models.KNN_model import create_KNN
from Global.Models.NN_model import create_NN
from Global.Models.RFC_model import create_RFC
from Global.Models.SVM_model import create_SVM
from Global.Parkinson_voice.ALE_method import ParkinsonALE


def print_time(time_start, time_end, filename):
    f = open(filename, 'w')
    f.write('\nTime spent training the model and explaining the result, seconds: ' + str(time_end - time_start))
    f.close()


class MyALE:
    def __init__(self, dataset_obj):
        self.path_to_project = my_utils.path_to_project
        self.RFC_model_filename = self.path_to_project + 'Global/Trained_models/RFC.joblib'
        self.SVM_model_filename = self.path_to_project + 'Global/Trained_models/SVM.joblib'
        self.KNN_model_filename = self.path_to_project + 'Global/Trained_models/KNN.joblib'
        self.NN_model_filename = self.path_to_project + 'Global/Trained_models/NN.joblib'
        self.BAG_model_filename = self.path_to_project + 'Global/Trained_models/BAG.joblib'
        self.ETC_model_filename = self.path_to_project + 'Global/Trained_models/ETC.joblib'
        self.dataset_obj = dataset_obj
        self.method_name = 'ALE'

    def ale(self, is_need_to_create_model):
        print('SVM')
        time_start = time.time()
        self.dataset_obj.calculate_ale(is_need_to_create_model, create_SVM, "SVM", self.SVM_model_filename)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + self.method_name + '/SVM/log.txt')

        print('RFC')
        time_start = time.time()
        self.dataset_obj.calculate_ale(is_need_to_create_model, create_RFC, "RFC", self.RFC_model_filename)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + self.method_name + '/RFC/log.txt')

        print('KNN')
        time_start = time.time()
        self.dataset_obj.calculate_ale(is_need_to_create_model, create_KNN, "KNN", self.KNN_model_filename)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + self.method_name + '/KNN/log.txt')

        print('NN')
        time_start = time.time()
        self.dataset_obj.calculate_ale(is_need_to_create_model, create_NN, "NN", self.NN_model_filename)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + self.method_name + '/NN/log.txt')

        print('BAG')
        time_start = time.time()
        self.dataset_obj.calculate_ale(is_need_to_create_model, create_BAG, "BAG", self.BAG_model_filename)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + self.method_name + '/BAG/log.txt')

        print('ETC')
        time_start = time.time()
        self.dataset_obj.calculate_ale(is_need_to_create_model, create_ETC, "ETC", self.ETC_model_filename)
        time_end = time.time()
        print_time(time_start, time_end, my_utils.PATH_TO_GLOBAL + self.dataset_obj.dataset_name
                   + '/Log/' + self.method_name + '/ETC/log.txt')


if __name__ == "__main__":
    is_need_to_create = True

    # -------------- HAND PD --------------
    handpd = HandPdALE()
    myale = MyALE(handpd)
    myale.ale(is_need_to_create)

    # -------------- PARKINSON VOICE --------------
    parkinson_voice = ParkinsonALE()
    myale = MyALE(parkinson_voice)
    myale.ale(is_need_to_create)
