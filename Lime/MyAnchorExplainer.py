import numpy
import numpy as np
import skimage
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from alibi.explainers import AnchorImage



def explanation(model, image, image_name=None, path_to_save=None):
    np.random.seed(0)
    #
    # explainer = AnchorImage(lambda x: model.predict(x), image.shape,
    #                                     segmentation_fn='slic',
    #                                     segmentation_kwargs={'n_segments': 15, 'compactness': 20, 'sigma': .5},
    #                                     images_background=None)

    explainer = AnchorImage(lambda x: model.predict(x), image.shape,
                            segmentation_fn=lambda x: skimage.segmentation.slic(x, n_segments=15,
                                                                                    compactness=20,
                                                                                    sigma=.5),
                            images_background=None,
                            seed=0)

    explanation = explainer.explain(image,
                                    threshold=1.0,
                                    p_sample=.5,
                                    tau=0.25,
                                    random=0,
                                    )


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))


    ax[0].imshow(explanation.anchor)
    ax[0].set_title("explanation")

    ax[1].imshow(explanation.segments)
    ax[1].set_title("segments")

    fig.suptitle("Anchor explainer")

    if path_to_save:
        plt.savefig(path_to_save +
                    image_name +
                    '_anchor'+'.png')

    plt.show()
