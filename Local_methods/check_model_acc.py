import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shutil
import tqdm
from tensorflow import keras


import my_utils

PATH_TO_DATA = my_utils.path_to_project + '/Datasets/dogs-vs-cats'
CONTENT_DIR = PATH_TO_DATA + '/content'
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/valid'
PATH_TO_SAVE_MODEL = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/Models/" + "my_cats_vs_dogs_model.h5"

if not os.path.exists(CONTENT_DIR):
    # Extract dataset
    import zipfile
    with zipfile.ZipFile(PATH_TO_DATA + '/train.zip', 'r') as zipf:
        zipf.extractall(CONTENT_DIR)

    # Split cats and dogs images to train and valid datasets
    img_filenames = os.listdir(TRAIN_DIR)
    dog_filenames = [fn for fn in img_filenames if fn.startswith('dog')]
    cat_filenames = [fn for fn in img_filenames if fn.startswith('cat')]
    dataset_filenames = train_test_split(
        dog_filenames, cat_filenames, test_size=0.1, shuffle=True, random_state=42
    )
    # Move images
    make_dirs = [d + a for a in ['/dog', '/cat'] for d in [TRAIN_DIR, VALID_DIR]]
    for dir, fns in zip(make_dirs, dataset_filenames):
        os.makedirs(dir, exist_ok=True)
        for fn in tqdm.tqdm(fns):
            shutil.move(os.path.join(TRAIN_DIR, fn), dir)
        print('elements in {}: {}'.format(dir, len(os.listdir(dir))))

BATCH_SIZE = 32
IMAGE_SHAPE = 128

train_generator = ImageDataGenerator(rescale=1./255)
valid_generator = ImageDataGenerator(rescale=1./255)
train_data = train_generator.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)
valid_data = valid_generator.flow_from_directory(
    directory=VALID_DIR,
    target_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

PATH_TO_MODEL = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/Models/" + "my_cats_vs_dogs_model.h5"
model = keras.models.load_model(PATH_TO_MODEL)
model.summary()

# print('Cats vs dogs accuracy ' + str(model.evaluate(valid_data)[1]))

# path_to_project = my_utils.path_to_project
# BATCH_SIZE = 100
# IMAGE_SIZE = [176, 208]
# train_generator = ImageDataGenerator(rescale=1./255)
# test_ds = train_generator.flow_from_directory(
#         path_to_project + "Datasets/Alzheimer_two_classes/test",
#         target_size=IMAGE_SIZE,
#         batch_size=BATCH_SIZE,
#     )
# model = keras.models.load_model('/Users/vladis_step/VKR_explanation_model/'
#                                 +'Local_methods/Alzheimer_two_classes/Models/my_alzheimer_model.h5')
#
# # acc примерно 93%
# print('Alz auc ' + str(model.evaluate(test_ds)[1]))

# counts = 0
# correct = 0
# for x, y in test_ds:
#     pred = model.predict(x)
#     counts += BATCH_SIZE
    #
    # for i in range(len(pred)):
    #     if np.where(y[i] == max(y[i]))[0][0] == np.where(pred[i] == max(pred[i]))[0][0]:
    #         correct += 1

    # print(correct / counts)
#     print(counts)
# print('acc '+ str(correct/counts))

