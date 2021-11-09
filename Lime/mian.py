from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
import my_utils
from tensorflow import keras

from Lime import MyAnchorExplainer, MyIntegratedGradient, MyLimeExplainer

PATH_TO_DATASET = my_utils.path_to_project + '/Datasets/dogs-vs-cats'
CONTENT_DIR = PATH_TO_DATASET + '/content'
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/valid'
PATH_TO_MODEL = my_utils.path_to_project + "Lime/Cats_vs_dogs/Models/" + "my_cats_vs_dogs_model.h5"
PATH_TO_SAVE_GRAPHS = my_utils.path_to_project + "Lime/Cats_vs_dogs/Graphs/"
IMAGE_SHAPE = 128


model = keras.models.load_model(PATH_TO_MODEL)

images = [
    ["cat.3", "/cat/cat.3.jpg"],
    ["cat.13", "/cat/cat.13.jpg"],
    ["cat.16", "/cat/cat.38.jpg"],

    ["dog.0", "/dog/dog.0.jpg"],
    ["dog.7", "/dog/dog.7.jpg"],
    ["dog.31", "/dog/dog.31.jpg"],

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
#                             path_to_save=my_utils.path_to_project +"Lime/Cats_vs_dogs/Graphs/",
#                             image_name='test')
# MyAnchorExplainer.explanation(model, image.img_to_array(load_img(path=VALID_DIR + images[0][1],
#                             target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0,
#                             path_to_save=my_utils.path_to_project +"Lime/Cats_vs_dogs/Graphs/",
#                             image_name='test')
# MyIntegratedGradient.explanation(model, image.img_to_array(load_img(
#                             path=VALID_DIR + images[0][1],
#                             target_size=(IMAGE_SHAPE, IMAGE_SHAPE))).astype("double") / 255.0,
#                             path_to_save=my_utils.path_to_project +"Lime/Cats_vs_dogs/Graphs/",
#                             image_name='test')




for i in images:

    print(VALID_DIR + i[1])
    loaded_img = image.img_to_array(load_img(path=VALID_DIR + i[1], target_size=(IMAGE_SHAPE, IMAGE_SHAPE)))\
                     .astype("double") / 255.0

    # MyLimeExplainer.explanation(model, loaded_img, num_samples=500,
    #                             image_name=i[0],
    #                             path_to_save=PATH_TO_SAVE_GRAPHS)
    MyAnchorExplainer.explanation(model, loaded_img,
                                image_name=i[0],
                                path_to_save=PATH_TO_SAVE_GRAPHS)
    # MyIntegratedGradient.explanation(model, loaded_img,
    #                               image_name=i[0],
    #                               path_to_save=PATH_TO_SAVE_GRAPHS)
