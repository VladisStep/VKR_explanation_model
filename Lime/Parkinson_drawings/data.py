from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data_parkinson_drawings_model():
    path_to_project = '/Users/vladis_step/VKR_explanation_model/'
    train_dir = path_to_project + "Datasets/pd/parkinsons-drawings/spiral/training/"
    test_dir = path_to_project + "Datasets/pd/parkinsons-drawings/spiral/testing/"

    IMG_WIDTH, IMG_HEIGHT = (300, 300)
    BATCH_SIZE = 16

    train_datagen_aug = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.1
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train = train_datagen_aug.flow_from_directory(
        train_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
    )

    return train, test
