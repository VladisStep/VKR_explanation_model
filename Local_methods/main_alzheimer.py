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
PATH_TO_SAVE_GRAPHS = PATH_PROJECT + "Local_methods/Alzheimer_two_classes/Graphs/"

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

        lime, lime_heatmap = MyLimeExplainer.explanation(model, loaded_img,
                                                         segmentation_method=segmentation_method)
        anchor = MyAnchorExplainer.explanation(model, loaded_img,
                                               segmentation_method=segmentation_method)
        grad = MyIntegratedGradient.explanation(model, loaded_img)

        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(7, 10))
        a = model.predict(np.expand_dims(loaded_img, axis=0))
        fig.suptitle('Demented: ' + str(round(a[0][0], 5)) + ' NonDemented: ' + str(round(a[0][1], 5)))

        ax[0, 0].set_title("Original")
        ax[0, 0].axis("off")
        ax[0, 0].imshow(loaded_img)

        ax[0, 1].set_title("Segmentation")
        ax[0, 1].imshow(segmentation_method(loaded_img))
        ax[0, 1].axis("off")

        ax[1, 0].set_title("LIME")
        ax[1, 0].imshow(lime)
        ax[1, 0].axis("off")

        ax1 = ax[1, 1].imshow(np.array(lime_heatmap, dtype=np.float64), cmap="RdYlGn",
                              vmax=np.array(lime_heatmap, dtype=np.float64).max(),
                              vmin=-np.array(lime_heatmap, dtype=np.float64).max())
        fig.colorbar(ax1, ax=ax[1])
        ax[1, 1].set_title("Heatmap")
        ax[1, 1].axis("off")

        ax[2, 0].set_title("Anchor explainer")
        ax[2, 0].axis("off")
        ax[2, 0].imshow(anchor)

        ax[2, 1].axis("off")

        visualize_image_attr(attr=grad, original_image=loaded_img, method='blended_heat_map',
                             sign='all', show_colorbar=False, title='IG explainer',
                             plt_fig_axis=(fig, ax[3, 0]), use_pyplot=False)

        visualize_image_attr(attr=grad,
                             sign='all', show_colorbar=False,
                             plt_fig_axis=(fig, ax[3, 1]), use_pyplot=False)

        plt.savefig(PATH_TO_SAVE_GRAPHS + pathList[i].split(sep='/')[-1])
        plt.show()


def getTestImages():

    path = '/Users/vladis_step/PycharmProjects/VKR_explanation_model/Datasets/Alzheimer_two_classes/train/Demented'
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))
    return filelist
    #
    # images = [
    #     "/cat/cat.359.jpg"
    # ]
    # PATH_TO_DATASET = my_utils.path_to_project + 'Datasets/dogs-vs-cats/content/train/dog/dog.205.jpg'
    # return [PATH_TO_DATASET]




# print('Введите путь до файла')
#/Datasets/dogs-vs-cats/train/dog/dog.25.jpg
# path = input()


# if (path == 'test'):
getExplanations(getTestImages())
# else:
#     getExplanations([path], 'some name')
