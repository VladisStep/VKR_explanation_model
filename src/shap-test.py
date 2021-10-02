import json
import os
import random

import cv2
import numpy as np
import shap
import torch
from matplotlib import pyplot as plt
from torchvision import models


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)).astype('float64'))
    return images


train_dir_healthy = "../parkinsons-drawings/spiral/training/healthy/"
train_dir_parkinson = "../parkinsons-drawings/spiral/training/parkinson/"
IMG_WIDTH, IMG_HEIGHT = (300, 300)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

i = 0


def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    # in addition, roll the axis so that they suit pytorch
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()


# load the model
model = models.vgg16(pretrained=True).eval()

# X, y = shap.datasets.imagenet50()

X = load_images_from_folder(train_dir_healthy)
X_parkinson = load_images_from_folder(train_dir_parkinson)

# 36
y = []
for i in range(0, 36):
    # y.append(random.randint(0, 1))
    y.append(0)
for i in range(0, 36):
    y.append(1)
y = np.array(y)
X = X + X_parkinson

to_explain = np.stack(np.array([X[0], X[1]]), axis=0)

# to_explain = image.img_to_array(load_img(path="./parkinsons-drawings/spiral/training/healthy/V02HE03.png",
#                                          target_size=(IMG_WIDTH, IMG_HEIGHT))).astype('double')

# load the ImageNet class names
# url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
# fname = shap.datasets.cache(url)
# with open(fname) as f:
#     class_names = json.load(f)

# class_names = ['0', '1']

e = shap.GradientExplainer((model, model.features[7]), normalize(np.stack(np.array(X), axis=0)))
print("first")
shap_values, indexes = e.shap_values(normalize(np.stack(np.array(to_explain), axis=0)), ranked_outputs=2, nsamples=2)
print("second")

# get the names for the classes
# index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
# index_names = np.vectorize(lambda x: print(str(x)))(indexes)
index_names = np.asarray([['healthy', 'parkinson'], ['healthy', 'parkinson']])
print("third")

# plot the explanations
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
print("fourth")

shap.image_plot(shap_values, to_explain, index_names)
# shap.image_plot(shap_values, to_explain)

print("END")
plt.show()
