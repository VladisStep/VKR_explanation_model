import tensorflow
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from skimage import exposure
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from Lime.Parkinson_drawings.data import load_data_parkinson_drawings_model

path_to_project = '/Users/vladis_step/VKR_explanation_model/'


# https://www.kaggle.com/kwisatzhaderach/predicting-parkinson-s-with-deep-neural-nets

def nopamine_model(mode):
    if (mode == 'spirals') or (mode == 'spiral'):
        input_layer = Input(shape=(256, 256, 1), name=f'{mode}_input_layer')
    elif (mode == 'waves') or (mode == 'wave'):
        input_layer = Input(shape=(256, 512, 1), name=f'{mode}_input_layer')

    m1 = Conv2D(256, (5, 5), dilation_rate=4, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001),
                activation='relu', padding='same')(input_layer)
    p1 = MaxPool2D((9, 9), strides=3)(m1)
    m2 = Conv2D(128, (5, 5), dilation_rate=2, kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001),
                activation='relu', padding='same')(p1)
    p2 = MaxPool2D((7, 7), strides=3)(m2)
    m3 = Conv2D(64, (3, 3), kernel_initializer='glorot_normal', kernel_regularizer=l2(0.001), activation='relu',
                padding='same')(p2)
    p3 = MaxPool2D((5, 5), strides=2)(m3)
    f1 = Flatten()(p3)
    d1 = Dense(666, activation='relu')(f1)
    d2 = Dense(1, activation='sigmoid')(d1)

    this_model = Model(input_layer, d2)
    # print(this_model.summary())
    return this_model


def create_dnn_model():
    batch_size = 16

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=12, min_lr=1e-9, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=16, verbose=1)

    # histogram equalizer
    def eqz_plz(img):
        return exposure.equalize_hist(img)

    spiral_datagen = ImageDataGenerator(rotation_range=360,  # they're spirals.
                                        width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        brightness_range=(0.5, 1.5),
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        preprocessing_function=eqz_plz,
                                        vertical_flip=True)

    path_to_project = '/Users/vladis_step/VKR_explanation_model/'
    train_dir = path_to_project + "Datasets/pd/parkinsons-drawings/spiral/training/"
    test_dir = path_to_project + "Datasets/pd/parkinsons-drawings/spiral/testing/"

    train = spiral_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(256, 256),
                                               color_mode="grayscale",
                                               batch_size=batch_size,
                                               class_mode="binary",
                                               shuffle=True,
                                               seed=666)

    test = spiral_datagen.flow_from_directory(directory=test_dir,
                                              target_size=(256, 256),
                                              color_mode="grayscale",
                                              batch_size=batch_size,
                                              class_mode="binary",
                                              shuffle=True,
                                              seed=710)

    # spirals
    # train, test = load_data_parkinson_drawings_model()
    spiral_model = nopamine_model(mode='spirals')
    spiral_model.compile(optimizer=Adam(learning_rate=3.15e-5), loss='binary_crossentropy', metrics=['accuracy'])

    spiral_model.fit(train,
                     validation_data=test,
                     epochs=666,
                     steps_per_epoch=(2000 // batch_size),
                     validation_steps=(800 // batch_size),
                     callbacks=[reduce_lr, early_stop],
                     verbose=1)
    spiral_model.save(path_to_project + 'Models/dnn_model.h5')


if __name__ == "__main__":
    create_dnn_model()
