import numpy
import numpy as np
import predictor as predictor
from keras.preprocessing.image import load_img
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import alibi
from alibi.explainers import AnchorImage
import my_utils

path_to_project = my_utils.path_to_project


class AnchorExplainer:

    def __init__(self, model_path, IMG_WIDTH, IMG_HEIGHT):
        self.model = keras.models.load_model(model_path)
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def process_img(self, path):
        path_to_img_to_explain = path_to_project + path
        return image.img_to_array(load_img(path=path_to_img_to_explain,
                                           target_size=(self.IMG_WIDTH, self.IMG_HEIGHT))).astype('int32')

    def predict_img(self, path):
        # x - преобразованный массив к виду, которая наша модель может "понять"
        x = numpy.expand_dims(self.process_img(path), axis=0)

        prediction = self.model.predict(x)

        print("Prediction: ", prediction)
        if np.argmax(prediction[0]) == 0:
            predicted_class = "Demented"
            prediction = np.round(prediction[0][0] * 100, 3)
        else:
            predicted_class = "NonDemented"
            prediction = np.round(prediction[0][1] * 100, 3)
        return predicted_class, prediction

    def get_explain(self, img_path, segmentation_fn='quickshift', save_fig=False, img_num=0):

        predicted_class, prediction = self.predict_img(img_path)
        image_shape = (self.IMG_WIDTH, self.IMG_HEIGHT, 3)

        np.random.seed(0)

        if segmentation_fn == 'quickshift':
            explainer = AnchorImage(self.model.predict, image_shape,
                                    segmentation_fn='quickshift',
                                    segmentation_kwargs={'kernel_size': 4, 'max_dist': 200, 'ratio': 0.2,
                                                         'random_seed': 0},
                                    images_background=None)
        else:
            explainer = AnchorImage(self.model.predict, image_shape,
                                    segmentation_fn='slic',
                                    segmentation_kwargs={'n_segments': 20, 'compactness': 20, 'sigma': .5},
                                    images_background=None)

        explanation = explainer.explain(self.process_img(img_path),
                                        # threshold=.95,
                                        # p_sample=.2,
                                        # tau=0.25,
                                        )

        plt.imshow(explanation.anchor)
        plt.title('AnchorExplainer segmentation_fn = ' + segmentation_fn + '\n' + img_path)
        if (save_fig):
            plt.savefig(path_to_project +
                        'Lime/Alzheimer_two_classes/Graphs/' +
                        str(img_num) + '_' +
                        'AnchorExplainer_' + segmentation_fn + '.png')
        plt.show()

        plt.imshow(explanation.segments)
        plt.title('AnchorExplainer segmentation_fn = ' + segmentation_fn + '\n' + img_path)
        if (save_fig):
            plt.savefig(path_to_project +
                        'Lime/Alzheimer_two_classes/Graphs/' +
                        str(img_num) + '_' +
                        'AnchorExplainer_segments_' + segmentation_fn + '.png')
        plt.show()
