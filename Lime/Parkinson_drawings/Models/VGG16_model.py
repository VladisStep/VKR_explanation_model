import matplotlib.pyplot as plt
import tensorflow
import torch
from tensorflow import keras
from Lime.Parkinson_drawings.data import load_data_parkinson_drawings_model

path_to_project = '/Users/vladis_step/VKR_explanation_model/'


def create_vgg16():
    IMG_WIDTH, IMG_HEIGHT = (300, 300)
    BATCH_SIZE = 16
    train, test = load_data_parkinson_drawings_model()

    # PRETRAINED CONVNET
    conv_base = tensorflow.keras.applications.vgg16.VGG16(
        include_top=False,  # Не подключать полносвязный слой
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),  # Форма входных тензоров
        weights='imagenet'  # Источник весов
    )
    conv_base.trainable = False  # Заморозка сверточной основы

    # conv_base.summary()
    new_model = keras.Sequential()
    new_model.add(conv_base)
    new_model.add(keras.layers.Flatten())
    new_model.add(keras.layers.Dense(128, activation="relu"))
    new_model.add(keras.layers.Dense(2, activation='softmax'))

    conv_base.trainable = False

    checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint(path_to_project + "Models/parkinson_model.h5",
                                                               save_best_only=True)

    new_model.compile(
        optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=2e-5),
        loss='binary_crossentropy',
        metrics=['acc']
    )

    new_hist = new_model.fit(
        train,
        epochs=1,
        batch_size=BATCH_SIZE,
        validation_data=test,
        callbacks=[checkpoint_cb]
    )

    # print_statistic(new_hist, new_model, test)


def print_statistic(new_hist, new_model, test):
    acc = new_hist.history['acc']
    val_acc = new_hist.history['val_acc']
    loss = new_hist.history['loss']
    val_loss = new_hist.history['val_loss']

    plt.plot(acc, label="Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend()
    plt.show()

    plt.plot(loss, label="Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    # WRONG PREDICTIONS
    wrong_preds = []

    y_pred = []
    y_true = []
    for idx, (img, label) in enumerate(test):
        pred = torch.argmax(new_model(img.unsqueeze(0)))

        if pred != label:
            wrong_preds.append(idx)

        y_pred.append(pred)
        y_true.append(label)

    print(f"Wrong preds: {wrong_preds}")


if __name__ == "__main__":
    create_vgg16()
