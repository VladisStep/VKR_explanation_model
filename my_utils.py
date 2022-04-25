from sklearn import metrics
from sklearn.metrics import classification_report, mean_squared_error

# path_to_project = 'C:/GitReps/VKR_explanation_model/'
path_to_project = '/Users/vladis_step/PycharmProjects/VKR_explanation_model/'


# GLOBAL
PATH_TO_GLOBAL = path_to_project + 'Global/'
PATH_TO_HANDPD_LOG = PATH_TO_GLOBAL + 'HandPD/Log/'
PATH_TO_PARKINSON_LOG = PATH_TO_GLOBAL + 'Parkinson_voice/Log/'
PATH_TO_IRIS = PATH_TO_GLOBAL + 'Test_dataset/Iris/Log/'
PATH_TO_FOOTBALL = PATH_TO_GLOBAL + 'Test_dataset/Football/Log/'
PATH_TO_HEART_FAILURE = PATH_TO_GLOBAL + 'Test_dataset/Heart_failure/Log/'
PATH_TO_WINE = PATH_TO_GLOBAL + 'Test_dataset/Wine/Log/'

PATH_TO_METHODS_BENCHMARK = PATH_TO_GLOBAL + 'Explanation_methods_benchmark/'


def shap_log(classifier, X_test, Y_test, model_name, method_name, data_for_prediction_array, path_to_log,
             time_start=None, time_end=None):
    preds = classifier.predict(X_test)
    s = method_name + '; ' + model_name + ':\n' + 'Prediction: ' + \
        str(classifier.predict_proba(data_for_prediction_array)) + '\n' + \
        str(classification_report(Y_test, preds)) + '\n' + 'The accuracy of this model is :\t' + \
        str(metrics.accuracy_score(preds, Y_test)) + '\nMean Squared Error: ' + \
        str(mean_squared_error(Y_test, preds))

    if time_start is not None and time_end is not None:
        s = s + '\nTime spent training the model, seconds: ' + str(time_end - time_start)

    f = open(path_to_log + method_name + '/' + model_name + '/log.txt', 'w')
    f.write(s)
    f.close()
