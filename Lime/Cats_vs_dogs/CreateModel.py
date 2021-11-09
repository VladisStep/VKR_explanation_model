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

import my_utils

# https://www.kaggle.com/c/dogs-vs-cats/data

PATH_TO_DATA = my_utils.path_to_project + '/Datasets/dogs-vs-cats'
CONTENT_DIR = PATH_TO_DATA + '/content'
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/valid'
PATH_TO_SAVE_MODEL = my_utils.path_to_project + "Lime/Cats_vs_dogs/Models/" + "my_cats_vs_dogs_model.h5"

# for dirname, _, filenames in os.walk(path_to_data):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


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

mobilenet_model = tf.keras.applications.mobilenet.MobileNet(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SHAPE, IMAGE_SHAPE, 3),
    pooling='avg'
)
mobilenet_model.trainable = False # freeze convolutional layers

# new fully connected part of the network
dense_model = tf.keras.models.Sequential([
    Dense(units=1000, activation='relu'),
    Dropout(0.5),
    Dense(units=128, activation='relu'),
    Dropout(0.5),
    Dense(units=2, activation='softmax')
])
# build a new model
model2 = tf.keras.models.Sequential([
    mobilenet_model,
    dense_model
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(PATH_TO_SAVE_MODEL,
                                                   save_best_only=True)

EPOCHS = 10
train_data.reset()
valid_data.reset()
history = model2.fit_generator(
    train_data,
    steps_per_epoch=train_data.n // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=valid_data,
    validation_steps=valid_data.n // BATCH_SIZE,
    callbacks=[checkpoint_cb]
)

def show_graphs(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='valid')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend(loc='upper left')
    plt.title('Loss (sparse_categorical_crossentropy)')

    plt.show()


show_graphs(history)