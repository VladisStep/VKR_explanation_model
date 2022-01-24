import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os


from Local_methods import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer


PATH_PROJECT = my_utils.path_to_project
PATH_TRAIN_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/train"
PATH_VALID_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/test"
PATH_TEST_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/test"
PATH_SAVE_MODEL = PATH_PROJECT + "Local_methods/Alzheimer_two_classes/Models/" + "my_alzheimer_model.h5"
PATH_SAVE_GRAPHS = PATH_PROJECT + "Local_methods/Alzheimer_two_classes/Graphs/"

print(PATH_SAVE_MODEL)
model = keras.models.load_model(PATH_SAVE_MODEL)

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(176, 208))).astype("double")/ 255.0



images = [
    ['ALZ_D26', '/Demented/26.jpg'],
    ['ALZ_D26(21)', '/Demented/26 (21).jpg'],
    ['ALZ_D26(22)', '/Demented/26 (22).jpg'],

    # ['ALZ_N26', '/NonDemented/26.jpg'],
    # ['ALZ_N26(62)', '/NonDemented/26 (62).jpg'],
    # ['ALZ_N26(63)', '/NonDemented/26 (63).jpg'],
    # ['ALZ_N26(64)', '/NonDemented/26 (64).jpg'],
    # ['ALZ_N26(65)', '/NonDemented/26 (65).jpg'],

    ['ALZ_N26', '/NonDemented/26.jpg'],
    ['ALZ_N27', '/NonDemented/27.jpg'],
    ['ALZ_N28', '/NonDemented/28.jpg'],

    # ['ALZ_N29', '/NonDemented/29.jpg'],
    # ['ALZ_N30', '/NonDemented/30.jpg'],
    # ['ALZ_N31', '/NonDemented/31.jpg'],


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