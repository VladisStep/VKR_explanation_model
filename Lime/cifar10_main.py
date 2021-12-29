import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os


from Lime import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer

PATH_TO_DATASET = my_utils.path_to_project + 'Datasets/CIFAR_10_test/'
CONTENT_DIR = PATH_TO_DATASET

PATH_TO_MODEL = my_utils.path_to_project + "Lime/CIFAR_10/Models/" + "keras_cifar10_trained_model.h5"
PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + "Lime/CIFAR_10/Graphs/"


print(PATH_TO_MODEL)
model = keras.models.load_model(PATH_TO_MODEL)

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(32, 32))).astype("double")/ 255.0



images = [
    ['1', '1.png'],
    ['2', '2.png'],
    ['3', '3.png'],
    ['4', '4.png'],
    ['5', '5.png'],
    ['6', '6.png'],
]

labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for img in images:

    loaded_img = processImgToModel(PATH_TO_DATASET + img[1])

    pred = model.predict(np.expand_dims(loaded_img, axis=0))


    MyLimeExplainer.explanation(model, loaded_img, num_samples=500,
                                        image_name=img[0],
                                        path_to_save=PATH_TO_SAVE_GRAPHS,
                                        pred_name=labels[np.argmax(pred)])
    MyAnchorExplainer.explanation(model, loaded_img,
                                        image_name=img[0],
                                        path_to_save=PATH_TO_SAVE_GRAPHS,
                                        pred_name=labels[np.argmax(pred)])
    MyIntegratedGradient.explanation(model, loaded_img,
                                        image_name=img[0],
                                        path_to_save=PATH_TO_SAVE_GRAPHS,
                                        pred_name=labels[np.argmax(pred)])