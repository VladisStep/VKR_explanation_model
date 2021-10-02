import matplotlib.image
import numpy
import tensorflow as tf
from keras.models import load_model

# alzheimer-mri-model-tf
# try:
#     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#     # print('Device:', tpu.master())
#     tf.config.experimental_connect_to_cluster(tpu)
#     tf.tpu.experimental.initialize_tpu_system(tpu)
#     strategy = tf.distribute.experimental.TPUStrategy(tpu)
# except:
#     strategy = tf.distribute.get_strategy()
# class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# NUM_CLASSES = len(class_names)
# BATCH_SIZE = 16 * strategy.num_replicas_in_sync
# IMAGE_SIZE = [176, 208]
# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "./Alzheimer_s Dataset/test",
#     image_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
# )
# def one_hot_label(image, label):
#     label = tf.one_hot(label, NUM_CLASSES)
#     return image, label
# test_ds = test_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
# test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
# model = load_model("alzheimer_model.h5")
# results = model.evaluate(test_ds)
# print("loss: ", results[0], "  auc: ", results[1], " acc: ", results[2])


# alzheimer-mri-model-tf

# from fastai.vision import *
# from fastai.vision.transform import *
#
# PATH = Path('./Alzheimer_s Dataset/')
# data = ImageDataBunch.from_folder(PATH, train="train/",
#                                       test="test/",
#                                       valid_pct=.4,
#                                       ds_tfms=transform,
#                                       size=112, bs=64,
#                                       ).normalize(imagenet_stats)
#
# model_export = load_learner('export.pkl')

# parkinson-drawing-dnn
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "../parkinsons-drawings/spiral/training/"
test_dir = "../parkinsons-drawings/spiral/testing/"
IMG_WIDTH, IMG_HEIGHT = (300, 300)
EPOCHS = 50
BATCH_SIZE = 16
CLASSES_NO = 2
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

from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
from lime import lime_image
from skimage.segmentation import mark_boundaries

# We use LIME to see at what the trained model is "looking" in the decision-making process.

path_to_img_to_explain = "../parkinsons-drawings/spiral/testing/healthy/V01HE01.png"

img = image.img_to_array(load_img(path=path_to_img_to_explain,
                                  target_size=(IMG_WIDTH, IMG_HEIGHT))).astype('double') / 255
x = numpy.expand_dims(img, axis=0)

model = load_model("../parkinson-drawings-models/dnn/parkinson_model_20ep.h5")
prediction = model.predict(x)
print("healthy:", int(prediction[0][0] * 100), "%", "parkinson:", int(prediction[0][1] * 100), "%")
pred_for_title = [prediction[0][1] * 100, "parkinson"] if prediction[0][0] < prediction[0][1] \
    else [prediction[0][0] * 100, "healthy"]

explainer = lime_image.LimeImageExplainer(random_state=42)  # Explains predictions on Image (i.e. matrix) data
explanation = explainer.explain_instance(img,
                                         model.predict,
                                         top_labels=1,
                                         # hide_color=0,
                                         # num_samples – size of the neighborhood to learn the linear model
                                         num_samples=1000,
                                         random_seed=42)

# temp, mask = explanation.get_image_and_mask(0, positive_only=True,
#                                             num_features=5, hide_rest=True)
# positive_only – if True, only take superpixels that positively contribute to the prediction of the label.
temp, mask = explanation.get_image_and_mask(model.predict(img.reshape((1, 300, 300, 3))).argmax(axis=1)[0],
                                            positive_only=False, hide_rest=False)

plt.axis("off")
plt.title("Explanation: " + str(round(pred_for_title[0], 3)) + "% sure it's " + str(pred_for_title[1]))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

# print("The prediction of the explanation model on the original instance: " + explanation.local_pred)

# Select the same class explained on the figures above.
ind = model.predict(img.reshape((1, 300, 300, 3))).argmax(axis=1)[0]

# Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = numpy.vectorize(dict_heatmap.get)(explanation.segments)

# Plot. The visualization makes more sense if a symmetrical colorbar is used.
plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
plt.colorbar()
plt.show()
