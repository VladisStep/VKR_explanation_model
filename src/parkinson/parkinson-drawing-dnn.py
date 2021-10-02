import matplotlib.pyplot as plt
import tensorflow
import torch
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../../parkinsons-drawings/spiral/training/"
test_dir = "../../parkinsons-drawings/spiral/testing/"

IMG_WIDTH, IMG_HEIGHT = (300, 300)
EPOCHS = 50
BATCH_SIZE = 16
CLASSES_NO = 2
print("EPOCHS = {}".format(EPOCHS))
print("BATCH_SIZE = {}".format(BATCH_SIZE))
print("CLASSES_NO = {}".format(CLASSES_NO))

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

# def build_model(reg=False):
#     regularizers = keras.regularizers.l2(1e-3)
#
#     model = keras.Sequential()
#
#     model.add(keras.layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
#
#     model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#
#     model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers))
#     model.add(keras.layers.MaxPooling2D((2, 2)))
#
#     model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers))
#     model.add(keras.layers.MaxPooling2D(2, 2))
#
#     model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_regularizer=regularizers))
#     model.add(keras.layers.MaxPooling2D(2, 2))
#
#     model.add(keras.layers.Flatten())
#     model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(128, activation="relu"))
#     model.add(keras.layers.Dropout(0.5))
#     model.add(keras.layers.Dense(2, activation="softmax"))
#
#     model.summary()
#
#     return model
#
#
# model = build_model(reg=False)
# print(model)
#
# model.compile(
#     loss="binary_crossentropy",
#     optimizer="adam",
#     metrics=['acc']
# )
#
# history = model.fit(
#     train,
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     validation_data=test
# )
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# plt.plot(acc, label="Accuracy")
# plt.plot(val_acc, label="Validation Accuracy")
# plt.legend()
# plt.show()
#
# plt.plot(loss, label="Loss")
# plt.plot(val_loss, label="Validation Loss")
# plt.legend()
# plt.show()

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

checkpoint_cb = tensorflow.keras.callbacks.ModelCheckpoint("./parkinson-drawings-models/dnn/parkinson_model_20ep.h5",
                                                           save_best_only=True)

new_model.compile(
    optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['acc']
)

new_hist = new_model.fit(
    train,
    epochs=20,
    batch_size=BATCH_SIZE,
    validation_data=test,
    callbacks=[checkpoint_cb]
)

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
