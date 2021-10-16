import numpy
import numpy as np
from keras.preprocessing.image import load_img
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from tensorflow import keras
from tensorflow.keras.preprocessing import image

import my_utils
from Lime.Alzheimer_two_classes.data import load_data_alzheimer_two_classes

path_to_project = my_utils.path_to_project

BATCH_SIZE = 16
train_ds, val_ds, test_ds = load_data_alzheimer_two_classes()
model = keras.models.load_model(my_utils.path_to_project + 'Models/my_alzheimer_model.h5')


def calculate_accuracy():
    counter = 0
    count_of_elements = 0
    for next_element in val_ds:
        # tf.print(next_element)
        d = model.predict(next_element[0])
        for i in range(BATCH_SIZE - 1):
            if np.argmax(d[i]) == np.array(next_element[1])[i]:
                counter += 1
            count_of_elements += 1

    print("Accuracy: ", counter / count_of_elements)


# calculate_accuracy()


IMG_WIDTH, IMG_HEIGHT = (176, 208)


def process_img(path):
    path_to_img_to_explain = path_to_project + path
    return image.img_to_array(load_img(path=path_to_img_to_explain,
                                       target_size=(IMG_WIDTH, IMG_HEIGHT))).astype('int32')


# path_to_img_to_explain = path_to_project + "Datasets/Alzheimer_two_classes/test/NonDemented/26 (80).jpg"
chosen_img = process_img("Datasets/Alzheimer_two_classes/test/Demented/27 (9).jpg")

img_path = [
    "Datasets/Alzheimer_two_classes/test/Demented/26 (19).jpg",
    # "Datasets/Alzheimer_two_classes/test/Demented/26 (20).jpg",
    # "Datasets/Alzheimer_two_classes/test/Demented/26 (21).jpg",
    # "Datasets/Alzheimer_two_classes/test/Demented/26 (22).jpg",
    # "Datasets/Alzheimer_two_classes/test/Demented/26 (23).jpg",
    #
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (62).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (63).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (64).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (65).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (66).jpg",

]

for i in range(len(img_path)):
    # x - преобразованный массив к виду, которая наша модель может "понять"
    x = numpy.expand_dims(process_img(img_path[i]), axis=0)
    prediction = model.predict(x)
    print("Prediction: ", prediction)
    if np.argmax(prediction[0]) == 0:
        predicted_class = "Demented"
        prediction = np.round(prediction[0][0] * 100, 3)
    else:
        predicted_class = "NonDemented"
        prediction = np.round(prediction[0][1] * 100, 3)

    num_samples = 500
    explainer = lime_image.LimeImageExplainer()  # Explains predictions on Image (i.e. matrix) data
    explanation = explainer.explain_instance(process_img(img_path[i]).astype('double') / 255,
                                             model.predict,
                                             top_labels=2,
                                             hide_color=0,
                                             # num_samples – size of the neighborhood to learn the linear model
                                             num_samples=num_samples,
                                             num_features=20)

    temp, mask = explanation.get_image_and_mask(
        model.predict(chosen_img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))).argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=3)

    plt.axis("off")
    plt.title("Explanation: " + str(prediction) + "% sure it's " + str(predicted_class))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.savefig(path_to_project + 'Lime/Alzheimer_two_classes/Graphs/alz_' + str(i) + '_' + str(num_samples) + '.png')
    plt.show()

    temp, mask = explanation.get_image_and_mask(1, negative_only=False, positive_only=False, hide_rest=False,
                                                num_features=5)
    plt.clf()
    plt.axis("off")
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.show()

# print("The prediction of the explanation model on the original instance: " + explanation.local_pred)

# Select the same class explained on the figures above.
# ind = model.predict(chosen_img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))).argmax(axis=1)[0]

# Map each explanation weight to the corresponding superpixel
#  a heatmap highlighting which pixels of the input image most strongly support the classification decision
# dict_heatmap = dict(explanation.local_exp[ind])
# heatmap = numpy.vectorize(dict_heatmap.get)(explanation.segments)

# dict_heatmap1 = dict(explanation.local_exp[explanation.top_labels[0]])
# heatmap1 = np.vectorize(dict_heatmap1.get)(explanation.segments)

# Plot. The visualization makes more sense if a symmetrical colorbar is used.

# plt.imshow(heatmap1, cmap='RdBu', vmax=heatmap1.max(), vmin=heatmap1.min())
# plt.colorbar()
# plt.savefig(path_to_project + 'Lime/Alzheimer_s/Graphs/alz.png')
# plt.show()
