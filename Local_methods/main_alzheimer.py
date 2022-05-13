import numpy as np
from alibi.utils.visualization import visualize_image_attr
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from skimage.segmentation import slic

import my_utils
from tensorflow import keras
import os


from Local_methods import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer


PATH_PROJECT = my_utils.path_to_project
# PATH_TRAIN_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/train"
# PATH_VALID_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/test"
# PATH_TEST_DS = PATH_PROJECT + "Datasets/Alzheimer_two_classes/test"
PATH_SAVE_MODEL = PATH_PROJECT + "Local_methods/Alzheimer_two_classes/Models/" + "my_alzheimer_model.h5"
# PATH_SAVE_GRAPHS = PATH_PROJECT + "Local_methods/Alzheimer_two_classes/Graphs/"

print(PATH_SAVE_MODEL)
model = keras.models.load_model(PATH_SAVE_MODEL)

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(176, 208))).astype("double")/ 255.0


def getExplanations(pathList):
    for i in range(len(pathList)):

        loaded_img = processImgToModel(pathList[i])

        print('Prediction: ' + str(model(np.expand_dims(loaded_img, axis=0))))
        segmentation_method = lambda img: slic(img, n_segments=50, compactness=5, sigma=1,
                                               start_label=1)

        lime = MyLimeExplainer.explanation(model, loaded_img, segmentation_method=segmentation_method,
                                           num_samples=1000)
        anchor = MyAnchorExplainer.explanation(model, loaded_img, segmentation_method=segmentation_method)
        grad = MyIntegratedGradient.explanation(model, loaded_img)

        fig, ax = plt.subplots(nrows=2, ncols=2)

        ax[0, 0].set_title("Original")
        ax[0, 0].axis("off")
        ax[0, 0].imshow(loaded_img)

        ax[0, 1].set_title("Lime")
        ax[0, 1].axis("off")
        ax[0, 1].imshow(lime)

        ax[1, 0].set_title("Anchor explainer")
        ax[1, 0].axis("off")
        ax[1, 0].imshow(anchor)

        ax[1, 1].set_title("Integrated gradient")
        ax[1, 1].axis("off")
        visualize_image_attr(attr=grad, original_image=loaded_img, method='blended_heat_map',
                             sign='all', show_colorbar=False, title='Overlaid Attributions',
                             plt_fig_axis=(fig, ax[1, 1]), use_pyplot=False)

        plt.show()


print('Введите путь до файла')
path = input()

getExplanations([path])
