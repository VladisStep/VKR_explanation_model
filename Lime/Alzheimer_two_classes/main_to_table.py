from PIL import Image
import numpy as np
from IPython.display import display
# you may want to keep logging enabled when doing your own work
import logging
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)  # disable Tensorflow warnings for this tutorial
import warnings

warnings.simplefilter("ignore")  # disable Keras warnings for this tutorial
import keras
from keras.applications import mobilenet_v2

import eli5
from eli5 import show_prediction

model = mobilenet_v2.MobileNetV2(include_top=True, weights='imagenet', classes=1000)

# check the input format
print(model.input_shape)
dims = model.input_shape[1:3]  # -> (height, width)
print(dims)

# we start from a path / URI.
# If you already have an image loaded, follow the subsequent steps
image_uri = '/Users/vladis_step/VKR_explanation_model/Datasets/keras-image-classifiers_5_11.png'

# this is the original "cat dog" image used in the Grad-CAM paper
# check the image with Pillow
im = Image.open(image_uri)
print(type(im))

# we could resize the image manually
# but instead let's use a utility function from `keras.preprocessing`
# we pass the required dimensions as a (height, width) tuple
im = keras.preprocessing.image.load_img(image_uri, target_size=dims)  # -> PIL image
print(im)

# we use a routine from `keras.preprocessing` for that as well
# we get a 'doc', an object almost ready to be inputted into the model

doc = keras.preprocessing.image.img_to_array(im)  # -> numpy array
print(type(doc), doc.shape)

# dimensions are looking good
# except that we are missing one thing - the batch size

# we can use a numpy routine to create an axis in the first position
doc = np.expand_dims(doc, axis=0)
print(type(doc), doc.shape)

# `keras.applications` models come with their own input preprocessing function
# for best results, apply that as well

# mobilenetv2-specific preprocessing
# (this operation is in-place)
mobilenet_v2.preprocess_input(doc)
print(type(doc), doc.shape)

# take back the first image from our 'batch'
image = keras.preprocessing.image.array_to_img(doc[0])
print(image)

# make a prediction about our sample image
predictions = model.predict(doc)
print(type(predictions), predictions.shape)

# check the top 5 indices
# `keras.applications` contains a function for that

top = mobilenet_v2.decode_predictions(predictions)
top_indices = np.argsort(predictions)[0, ::-1][:5]

print(top)
print(top_indices)

print("-------------------------------")
print(eli5.format_as_html(show_prediction(model, doc)))

# we need to pass the network
# the input as a numpy array
# tf.compat.v1.disable_eager_execution()
# eli5.show_prediction(model, doc)
#
# eli5.show_prediction(model, doc, image=image)
#
# cat_idx = 282  # ImageNet ID for "tiger_cat" class, because we have a cat in the picture
# display(eli5.show_prediction(model, doc, targets=[cat_idx]))  # pass the class id
#
# window_idx = 904  # 'window screen'
# turtle_idx = 35  # 'mud turtle', some nonsense
# show_prediction(model, doc, targets=[window_idx])
# show_prediction(model, doc, targets=[turtle_idx])
