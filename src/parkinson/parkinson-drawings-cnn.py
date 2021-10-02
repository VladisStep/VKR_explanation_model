import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import load_img

plt.style.use('dark_background')
plt.figure(figsize=(12, 12))
for i in range(1, 10, 1):
    plt.subplot(3, 3, i)
    img = load_img("./parkinsons-drawings/spiral/training/healthy/" +
                   os.listdir("../parkinsons-drawings/spiral/training/healthy")[i])
    plt.imshow(img)
plt.show()

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(activation='relu', units=128))
classifier.add(Dense(activation='sigmoid', units=1))

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

spiral_train_generator = train_datagen.flow_from_directory('./parkinsons-drawings/spiral/training',
                                                           target_size=(128, 128),
                                                           batch_size=32,
                                                           class_mode='binary')

spiral_test_generator = test_datagen.flow_from_directory('./parkinsons-drawings/spiral/testing',
                                                         target_size=(128, 128),
                                                         batch_size=32,
                                                         class_mode='binary')

wave_train_generator = train_datagen.flow_from_directory('./parkinsons-drawings/wave/training',
                                                         target_size=(128, 128),
                                                         batch_size=32,
                                                         class_mode='binary')

wave_test_generator = test_datagen.flow_from_directory('./parkinsons-drawings/wave/testing',
                                                       target_size=(128, 128),
                                                       batch_size=32,
                                                       class_mode='binary')

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True
                               )

reduce_learningrate = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        min_delta=0.0001)

callbacks_list = [early_stopping, reduce_learningrate]

epochs = 48

classifier.compile(loss='binary_crossentropy',
                   optimizer=tensorflow.keras.optimizers.Adam(lr=0.001),
                   metrics=['accuracy'])

history = classifier.fit_generator(
    spiral_train_generator,
    steps_per_epoch=spiral_train_generator.n // spiral_train_generator.batch_size,
    epochs=48,
    validation_data=spiral_test_generator,
    validation_steps=spiral_test_generator.n // spiral_test_generator.batch_size,
    callbacks=callbacks_list
)

plt.style.use('dark_background')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss', color='red')
plt.legend(loc='lower right')
plt.show()

from lime import lime_image
from tensorflow.keras.preprocessing import image

explainer = lime_image.LimeImageExplainer()

img = image.img_to_array(load_img("../parkinsons-drawings/spiral/training/parkinson/V01PE02.png"))
from skimage.transform import resize

img = resize(img, (128, 128, 3))
img = np.expand_dims(img, axis=0)
explanation = explainer.explain_instance(img[0].astype(
    'double'), classifier.predict,
    top_labels=2, hide_color=0, num_samples=1000)

from skimage.segmentation import mark_boundaries

temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=True)
temp_2, mask_2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10,
                                                hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
ax1.imshow(mark_boundaries(temp_1, mask_1))
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax1.axis('off')
ax2.axis('off')
plt.show()
