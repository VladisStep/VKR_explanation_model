import numpy
import numpy as np
import skimage
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries


def explanation(model,
                image,
                segmentation_method=None
                ):

    model_prediction = model.predict(np.expand_dims(image, axis=0))

    np.random.seed(0)

    if segmentation_method == None:
        segmentation_fn = lambda x: skimage.segmentation.slic(x, n_segments=15, compactness=20, sigma=.5)
    else:
        segmentation_fn = segmentation_method

    explainer = lime_image.LimeImageExplainer()
    lime_explanation = explainer.explain_instance(image,
                                                  model.predict,
                                                  top_labels=2,
                                                  hide_color=0,
                                                  segmentation_fn = segmentation_fn,
                                                  num_samples=1000,
                                                  random_seed=0,
                                                  num_features=1000)


    temp, mask = lime_explanation.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)

    # Heatmap
    ind = model_prediction.argmax(axis=1)[0]
    dict_heatmap = dict(lime_explanation.local_exp[ind])
    heatmap = numpy.vectorize(dict_heatmap.get)(lime_explanation.segments)


    return mark_boundaries(temp / 2 + 0.5, mask), heatmap
