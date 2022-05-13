import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import felzenszwalb, quickshift, slic
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os
from alibi.utils.visualization import visualize_image_attr

from Local_methods import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer


PATH_TO_MODEL = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/Models/" + "my_cats_vs_dogs_model.h5"
# PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/Graphs/"
PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + 'test/'
IMAGE_SHAPE = 128
model = keras.models.load_model(PATH_TO_MODEL)

def processImgToModel(image_path):
    return image.img_to_array(load_img(path=image_path, target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0


def getExplanations(pathList, name):
    for i in range(len(pathList)):

        loaded_img = processImgToModel(pathList[i])
        print('Prediction: ' + str(model(np.expand_dims(loaded_img, axis=0))))
        segmentation_method = lambda img: slic(img, n_segments=50, compactness=5, sigma=1,
                     start_label=1)

        # lime = MyLimeExplainer.explanation(model, loaded_img, segmentation_method=segmentation_method,
        #                             num_samples=1000,
        #                             image_name=name + str(i),
        #                             path_to_save=PATH_TO_SAVE_GRAPHS)
        # anchor = MyAnchorExplainer.explanation(model, loaded_img, segmentation_method=segmentation_method,
        #                             image_name=name + str(i),
        #                             path_to_save=PATH_TO_SAVE_GRAPHS)
        grad = MyIntegratedGradient.explanation(model, loaded_img,
                                      image_name=name + str(i),
                                      path_to_save=PATH_TO_SAVE_GRAPHS)

        # fig, ax = plt.subplots(nrows=2, ncols=2)
        # a = model.predict(np.expand_dims(loaded_img, axis=0))
        # fig.suptitle('Cat: ' + str(round(a[0][0], 5)) + ' Dog: ' + str(round(a[0][1], 5)))
        # ax[0, 0].set_title("Original")
        # ax[0, 0].axis("off")
        # ax[0, 0].imshow(segmentation_method(loaded_img))
        # ax[0, 1].set_title("LIME")
        # ax[0, 1].axis("off")
        # ax[0, 1].imshow(lime)
        # ax[1, 0].set_title("Anchor explainer")
        # ax[1, 0].axis("off")
        # ax[1, 0].imshow(anchor)
        # ax[1, 1].set_title("IG explainer")
        # ax[1, 1].axis("off")
        # visualize_image_attr(attr=grad, original_image=loaded_img, method='blended_heat_map',
        #                      sign='all', show_colorbar=False, title='Overlaid Attributions',
        #                      plt_fig_axis=(fig, ax[1, 1]), use_pyplot=False)
        # plt.savefig(PATH_TO_SAVE_GRAPHS + pathList[i].split(sep='/')[-1])
        # plt.show()


def getTestImages():
    #
    # path = '/Users/vladis_step/PycharmProjects/VKR_explanation_model/Datasets/dogs-vs-cats/content/train/dog'
    # filelist = []
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         filelist.append(os.path.join(root, file))
    # return filelist

    images = [
        "/cat/cat.359.jpg"
    ]
    PATH_TO_DATASET = my_utils.path_to_project + 'Datasets/dogs-vs-cats/content/train/cat/cat.239.jpg'
    return [PATH_TO_DATASET]




# print('Введите путь до файла')
#/Datasets/dogs-vs-cats/train/dog/dog.25.jpg
# path = input()


# if (path == 'test'):
getExplanations(getTestImages(), 'some name')
# else:
#     getExplanations([path], 'some name')

