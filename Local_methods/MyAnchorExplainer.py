import numpy
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from alibi.explainers import AnchorImage


def explanation(model,
                image,
                segmentation_method=None
                ):
    np.random.seed(0)

    if segmentation_method == None:
        segmentation_fn = lambda x: skimage.segmentation.slic(x, n_segments=15, compactness=20, sigma=.5)
    else:
        segmentation_fn = segmentation_method

    explainer = AnchorImage(lambda x: model.predict(x), image.shape,
                            segmentation_fn= segmentation_fn,
                            images_background=None,
                            seed=0)

    explanation = explainer.explain(image,
                                    p_sample=.5,
                                    threshold=0.95,
                                    tau=0.25,
                                    beam_size=2,
                                    batch_size=100,
                                    coverage_samples=10000,
                                    random=0
                                    )

    return explanation.anchor
