from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.xception import preprocess_input, decode_predictions
import numpy as np
import tensorflow as tf

import eli5

tf.compat.v1.disable_eager_execution()
import PIL
from PIL import Image
import requests
from io import BytesIO



# load the model
model = Xception(weights='imagenet', include_top=True)

# chose the URL image that you want
URL = "https://images.unsplash.com/photo-1529429617124-95b109e86bb8?ixlib=rb-1.2.1&amp;ixid=eyJhcHBfaWQiOjEyMDd9&amp;auto=format&amp;fit=crop&amp;w=500&amp;q=60"
# get the image
response = requests.get(URL)
img = Image.open(BytesIO(response.content))
# resize the image according to each model (see documentation of each model)
img = img.resize((299, 299))

# convert to numpy array
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

# return the top 20 detected objects
label = decode_predictions(features, top=20)

np.argsort(features)[0, ::-1][:10]

eli5.show_prediction(model, x, targets=[905])

eli5.show_prediction(model, x, targets=[424])