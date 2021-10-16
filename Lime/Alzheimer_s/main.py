import numpy as np
from tensorflow import keras
from data import load_data_alzheimer
import numpy
from keras.models import load_model
from matplotlib import pyplot as plt
from Lime.Parkinson_drawings.data import load_data_parkinson_drawings_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
from lime import lime_image
from skimage.segmentation import mark_boundaries

path_to_project = '/Users/vladis_step/VKR_explanation_model/'

BATCH_SIZE = 16
train_ds, val_ds, test_ds = load_data_alzheimer()
model = keras.models.load_model('/Users/vladis_step/VKR_explanation_model/Models/alzheimer_model.h5')

counter = 0
count_of_elements = 0
for next_element in val_ds:
    # tf.print(next_element)
    d = model.predict(next_element[0])
    for i in range(BATCH_SIZE - 1):
        # print(np.argmax(d[i]), np.array(next_element[1])[i])
        if np.argmax(d[i]) == np.array(next_element[1])[i]:
            counter += 1
        count_of_elements += 1

print("Accuracy: ", counter / count_of_elements)







IMG_WIDTH, IMG_HEIGHT = (176, 208)

path_to_img_to_explain = path_to_project + "Datasets/Alzheimer_s Dataset/test/ModerateDemented/27.jpg"
chosen_img = image.img_to_array(load_img(path=path_to_img_to_explain,
                                         target_size=(IMG_WIDTH, IMG_HEIGHT))).astype('double') / 255

# x - преобразованный массив к виду, которая наша модель может "понять"
x = numpy.expand_dims(chosen_img, axis=0)

prediction = model.predict(x)
print("Prediction: ", prediction)

num_samples = 100
explainer = lime_image.LimeImageExplainer()  # Explains predictions on Image (i.e. matrix) data
explanation = explainer.explain_instance(chosen_img,
                                         model.predict,
                                         top_labels=3,
                                         hide_color=0,
                                         # num_samples – size of the neighborhood to learn the linear model
                                         num_samples=num_samples)

temp, mask = explanation.get_image_and_mask(model.predict(chosen_img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))).argmax(axis=1)[0],
                                            positive_only=False, hide_rest=False)

plt.axis("off")
# plt.title("Explanation: " + str(round(pred_for_title[0], 3)) + "% sure it's " + str(pred_for_title[1]))
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.savefig(path_to_project + 'Lime/Alzheimer_s/Graphs/alz.png')
plt.show()

# print("The prediction of the explanation model on the original instance: " + explanation.local_pred)

# Select the same class explained on the figures above.
# ind = model.predict(chosen_img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3))).argmax(axis=1)[0]

# Map each explanation weight to the corresponding superpixel
#  a heatmap highlighting which pixels of the input image most strongly support the classification decision
# dict_heatmap = dict(explanation.local_exp[ind])
# heatmap = numpy.vectorize(dict_heatmap.get)(explanation.segments)

dict_heatmap1 = dict(explanation.local_exp[explanation.top_labels[0]])
heatmap1 = np.vectorize(dict_heatmap1.get)(explanation.segments)

# Plot. The visualization makes more sense if a symmetrical colorbar is used.

# plt.imshow(heatmap1, cmap='RdBu', vmax=heatmap1.max(), vmin=heatmap1.min())
# plt.colorbar()
# plt.savefig(path_to_project + 'Lime/Alzheimer_s/Graphs/alz.png')
# plt.show()
