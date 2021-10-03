
import tensorflow as tf


path_to_project = '/Users/vladis_step/VKR_explanation_model/'

def load_data_alzheimer():
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = 16
    IMAGE_SIZE = [176, 208]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_project + "Datasets/Alzheimer_s Dataset/train",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_project + "Datasets/Alzheimer_s Dataset/train",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
    train_ds.class_names = class_names
    val_ds.class_names = class_names

    NUM_CLASSES = len(class_names)

    def one_hot_label(image, label):
        label = tf.one_hot(label, NUM_CLASSES)
        return image, label

    # train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    # val_ds = val_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    #
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        path_to_project + "Datasets/Alzheimer_s Dataset/test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    # test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds