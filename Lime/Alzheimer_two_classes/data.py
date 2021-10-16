import tensorflow as tf

import my_utils

path_to_project = my_utils.path_to_project


def load_data_alzheimer_two_classes():
    BATCH_SIZE = 16
    IMAGE_SIZE = [176, 208]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_project + "Datasets/Alzheimer_two_classes/train",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_project + "Datasets/Alzheimer_two_classes/train",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_project + "Datasets/Alzheimer_two_classes/test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    return train_ds, val_ds, test_ds


