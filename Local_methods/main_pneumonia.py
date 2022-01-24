import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os


from Local_methods import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer




PATH_PROJECT = my_utils.path_to_project
PATH_TRAIN_DS = PATH_PROJECT + "Datasets/chest_xray/train"
PATH_VALID_DS = PATH_PROJECT + "Datasets/chest_xray/test"
PATH_TEST_DS = PATH_PROJECT + "Datasets/chest_xray/test"
PATH_SAVE_MODEL = PATH_PROJECT + "Local_methods/Pneumonia/Models/" + "my_pneumonia_model.h5"
PATH_SAVE_GRAPHS = PATH_PROJECT + "Local_methods/Pneumonia/Graphs/"


model = keras.models.load_model(PATH_SAVE_MODEL)

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(180, 180))).astype("double")/ 255.0



images = [
    ['PNU_N1', '/NORMAL/IM-0001-0001.jpeg'],
    ['PNU_N41', '/NORMAL/IM-0041-0001.jpeg'],
    ['PNU_N60', '/NORMAL/NORMAL2-IM-0060-0001.jpeg'],
    ['PNU_P1', '/PNEUMONIA/person1_virus_6.jpeg'],
    ['PNU_P78', '/PNEUMONIA/person78_bacteria_381.jpeg'],
    ['PNU_P110', '/PNEUMONIA/person110_bacteria_531.jpeg'],

]

# model.summary()

for img in images:

    loaded_img = processImgToModel(PATH_TEST_DS + img[1])

    pred = str(model.predict(np.expand_dims(loaded_img, axis=0)))
    print(pred)


    MyLimeExplainer.explanation(model, loaded_img, num_samples=500,
                                image_name=img[0],
                                path_to_save=PATH_SAVE_GRAPHS,
                                pred_name=pred)
    MyAnchorExplainer.explanation(model, loaded_img,
                                image_name=img[0],
                                path_to_save=PATH_SAVE_GRAPHS,
                                pred_name=pred)
    MyIntegratedGradient.explanation(model, loaded_img,
                                image_name=img[0],
                                path_to_save=PATH_SAVE_GRAPHS,
                                pred_name=pred)

