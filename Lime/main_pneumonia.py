import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os


from Lime import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer

PATH_TO_DATASET = my_utils.path_to_project + 'Datasets/chest_xray'
CONTENT_DIR = PATH_TO_DATASET
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/test'
PATH_TO_MODEL = my_utils.path_to_project + "Lime/Pneumonia/Models/" + "my_pneumonia_model.h5"
PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + "Lime/Pneumonia/Graphs/"


print(PATH_TO_MODEL)
model = keras.models.load_model('/Users/vladis_step/VKR_explanation_model/Lime/Pneumonia/Models/my_pneumonia_model.h5')

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(180, 180))).astype("double")/ 255.0



images = [
    # ['D26', '/Demented/26.jpg'],
    # ['D26(19)', '/Demented/26 (19).jpg'],
    # ['D26(20)', '/Demented/26 (20).jpg'],
    # ['D26(21)', '/Demented/26 (21).jpg'],
    # ['D26(22)', '/Demented/26 (22).jpg'],

    # ['N26', '/NonDemented/26.jpg'],
    # ['N26(62)', '/NonDemented/26 (62).jpg'],
    # ['N26(63)', '/NonDemented/26 (63).jpg'],
    # ['N26(64)', '/NonDemented/26 (64).jpg'],
    # ['N26(65)', '/NonDemented/26 (65).jpg'],

    ['N1', '/NORMAL/IM-0001-0001.jpeg'],
    ['N41', '/NORMAL/IM-0041-0001.jpeg'],
    ['N60', '/NORMAL/NORMAL2-IM-0060-0001.jpeg'],
    ['P1', '/PNEUMONIA/person1_virus_6.jpeg'],
    ['P78', '/PNEUMONIA/person78_bacteria_381.jpeg'],
    ['P110', '/PNEUMONIA/person110_bacteria_531.jpeg'],


]

# model.summary()

for img in images:

    loaded_img = processImgToModel(VALID_DIR + img[1])

    print(model.predict(np.expand_dims(loaded_img, axis=0)))

    #
    # MyLimeExplainer.explanation(model, loaded_img, num_samples=500,
    #                             image_name=img[0],
    #                             path_to_save=PATH_TO_SAVE_GRAPHS)
    # MyAnchorExplainer.explanation(model, loaded_img,
    #                                     image_name=img[0],
    #                                     path_to_save=PATH_TO_SAVE_GRAPHS)
    # MyIntegratedGradient.explanation(model, loaded_img,
    #                                       image_name=img[0],
    #                                       path_to_save=PATH_TO_SAVE_GRAPHS)