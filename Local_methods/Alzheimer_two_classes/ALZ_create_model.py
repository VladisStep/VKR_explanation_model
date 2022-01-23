# https://www.kaggle.com/amyjang/alzheimer-mri-model-tensorflow-2-3-data-loading

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import my_utils

path_to_project = my_utils.path_to_project

PATH_PROJECT = my_utils.path_to_project
PATH_TRAIN_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/train"
PATH_VALID_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/test"
PATH_TEST_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/test"
PATH_SAVE_MODEL = PATH_PROJECT + "Local_methods/Alzheimer_two_classes/Models/"

def load_data_alzheimer_two_classes():
    BATCH_SIZE = 16
    IMAGE_SIZE = [176, 208]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_TRAIN_DS,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_VALID_DS,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PATH_TEST_DS,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    return train_ds, val_ds, test_ds


def data_transformation(train_ds, val_ds, test_ds):
    def one_hot_label(image, label):
        rescale = Rescaling(scale=1.0 / 255)
        label = tf.one_hot(label, NUM_CLASSES)
        return rescale(image), label

    train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, test_ds


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 16
IMAGE_SIZE = [176, 208]
EPOCHS = 100
class_names = ['Dementia', 'NonDementia']
NUM_CLASSES = len(class_names)

train, val, test = load_data_alzheimer_two_classes()
train, val, test = data_transformation(train, val, test)

train.class_names = class_names
val.class_names = class_names


def conv_block(filters):
    block = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.SeparableConv2D(filters, 3, activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D()
    ]
    )

    return block


def dense_block(units, dropout_rate):
    block = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout_rate)
    ])

    return block


def build_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*IMAGE_SIZE, 3)),

        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),

        conv_block(32),
        conv_block(64),

        conv_block(128),
        tf.keras.layers.Dropout(0.2),

        conv_block(256),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        dense_block(512, 0.7),
        dense_block(128, 0.5),
        dense_block(64, 0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


model = build_model()

METRICS = [tf.keras.metrics.AUC(name='auc')]

model.compile(
    optimizer='adam',
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=METRICS
)


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(PATH_SAVE_MODEL + "alzheimer_model.h5",
                                                   save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=10,
                                                     restore_best_weights=True)

history = model.fit(
    train,
    validation_data=val,
    callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
    epochs=EPOCHS
)

fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(['auc', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])

plt.show()
# _ = model.evaluate(test_ds)
