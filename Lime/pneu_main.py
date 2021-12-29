import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os
import cv2


from Lime import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer

PATH_TO_DATASET = my_utils.path_to_project + 'Datasets/chest_xray/' + "test/PNEUMONIA/"

PATH_TO_MODEL = my_utils.path_to_project + "Lime/Pneumonia/Models/" + "xray_model.h5"
PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + "Lime/Pneumonia/Graphs/"


print(PATH_TO_MODEL)
model = keras.models.load_model(PATH_TO_MODEL)

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(180, 180))).astype("double")/ 255.0



images = [
    ['1', 'person1_virus_6.jpeg'],

]

for img in images:

    loaded_img = np.expand_dims(processImgToModel(PATH_TO_DATASET + img[1]), axis=0)

    pred = model.predict(loaded_img)

    MyLimeExplainer.explanation(model, loaded_img, num_samples=500,
                                image_name=img[0],
                                path_to_save=PATH_TO_SAVE_GRAPHS)
    print(pred)



