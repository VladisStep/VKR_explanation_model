import my_utils
from Shap.HandPD.main import HandPDShap
from Shap.Parkinson_voice.main import ParkinsonVoiceShap

path_to_project = my_utils.path_to_project


if __name__ == "__main__":
    handpd = HandPDShap()

    is_need_to_create = True

    handpd.shap_svm(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    handpd.shap_rfc(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    handpd.shap_knn(is_need_to_create_model=is_need_to_create, chosen_instance=5)

    parkinson_voice = ParkinsonVoiceShap()

    parkinson_voice.shap_svm(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    parkinson_voice.shap_rfc(is_need_to_create_model=is_need_to_create, chosen_instance=5)
    parkinson_voice.shap_knn(is_need_to_create_model=is_need_to_create, chosen_instance=5)
