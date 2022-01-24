import numpy as np
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras
import os


from Local_methods import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer

PATH_TO_DATASET = my_utils.path_to_project + '/Datasets/dogs-vs-cats'
CONTENT_DIR = PATH_TO_DATASET + '/content'
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/valid'
PATH_TO_MODEL = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/Models/" + "my_cats_vs_dogs_model.h5"
PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/Graphs/"
IMAGE_SHAPE = 128


model = keras.models.load_model(PATH_TO_MODEL)

images = [
    ["CVD_cat.3", "/cat/cat.3.jpg"],
    ["CVD_cat.13", "/cat/cat.13.jpg"],
    ["CVD_cat.16", "/cat/cat.38.jpg"],

    ["CVD_dog.0", "/dog/dog.0.jpg"],
    ["CVD_dog.7", "/dog/dog.7.jpg"],
    ["CVD_dog.31", "/dog/dog.31.jpg"],

    # ["tr-cat.0", "/cat/cat.0.jpg"],
    # ["tr-cat.1", "/cat/cat.1.jpg"],
    # ["tr-cat.2", "/cat/cat.2.jpg"],
    #
    # ["tr-dog.1", "/dog/dog.1.jpg"],
    # ["tr-dog.2", "/dog/dog.2.jpg"],
    # ["tr-dog.3", "/dog/dog.3.jpg"],
]



# MyLimeExplainer.explanation(model, image.img_to_array(load_img(path=VALID_DIR + images[0][1],
#                             target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0,
#                             path_to_save=my_utils.path_to_project +"Local_methods/Cats_vs_dogs/Graphs/",
#                             image_name='test')
# MyAnchorExplainer.explanation(model, image.img_to_array(load_img(path=VALID_DIR + images[0][1],
#                             target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0,
#                             path_to_save=my_utils.path_to_project +"Local_methods/Cats_vs_dogs/Graphs/",
#                             image_name='test')
# MyIntegratedGradient.explanation(model, image.img_to_array(load_img(
#                             path=VALID_DIR + images[0][1],
#                             target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0,
#                             path_to_save=my_utils.path_to_project +"Local_methods/Cats_vs_dogs/Graphs/",
#                             image_name='test')


def processImgToModel(image_path):
    return  image.img_to_array(load_img(path=image_path, target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0

def getMinMax():
    minPath = []
    maxPath = []
    EXAMPLE_COUNT = 3

    for dirname, _, filenames in os.walk(VALID_DIR):
        for filename in filenames:
            # print(os.path.join(dirname, filename))
            current_path = os.path.join(dirname, filename)
            pred = model.predict(
                np.expand_dims(processImgToModel(current_path), axis=0))
            current_value = abs(pred[0][0] - pred[0][1])

            if current_value < 0.1 and len(minPath) < EXAMPLE_COUNT:
                minPath.append(current_path)
                print("min: " + current_path)
            elif 0.95 > current_value > 0.9 and len(maxPath) < EXAMPLE_COUNT:
                maxPath.append(current_path)
                print("max: " + current_path)
            elif len(maxPath) >= EXAMPLE_COUNT and len(minPath) >= EXAMPLE_COUNT:
                break

    return minPath, maxPath


def getExplanations(pathList, name):
    for i in range(len(pathList)):

        loaded_img = processImgToModel(pathList[i])

        MyLimeExplainer.explanation(model, loaded_img, num_samples=500,
                                    image_name=name + str(i),
                                    path_to_save=PATH_TO_SAVE_GRAPHS)
        MyAnchorExplainer.explanation(model, loaded_img,
                                    image_name=name + str(i),
                                    path_to_save=PATH_TO_SAVE_GRAPHS)
        MyIntegratedGradient.explanation(model, loaded_img,
                                      image_name=name + str(i),
                                      path_to_save=PATH_TO_SAVE_GRAPHS)


def anchorTest(pathList, name):
    PATH_TO_ANCHOR_TESTS = my_utils.path_to_project + "Local_methods/Cats_vs_dogs/AnchorTests/"
    for i in range(len(pathList)):
        loaded_img = processImgToModel(pathList[i])
        MyAnchorExplainer.explanation(model, loaded_img,
                                      image_name=name + str(i),
                                      path_to_save=PATH_TO_ANCHOR_TESTS)





for im in images:
    getExplanations([VALID_DIR + im[1]],  im[0])
# minPath, maxPath = getMinMax()
# anchorTest(minPath, "min_")
# anchorTest(maxPath, "max_")

# getExplanations(minPath, "min_")
# getExplanations(maxPath, "max_")