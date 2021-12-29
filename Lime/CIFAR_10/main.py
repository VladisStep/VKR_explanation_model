

import numpy as np
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os
from tensorflow.keras.datasets import cifar10
import tensorflow as tf




def processImgToModel(image_path):
    return  image.img_to_array(load_img(path=image_path, target_size=(32, 32))).astype("double") / 255.0

model = keras.models.load_model('/Users/vladis_step/VKR_explanation_model/Lime/CIFAR_10/Models/keras_cifar10_trained_model.h5')

img1 = processImgToModel('/Users/vladis_step/Desktop/test/5.png')
img2 = processImgToModel('/Users/vladis_step/Desktop/test/7.png')

print(model.predict(np.expand_dims(img1, axis=0)))
print(model.predict(np.expand_dims(img2, axis=0)))








