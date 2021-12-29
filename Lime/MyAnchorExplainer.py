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
                threshold=0.95,
                segmentation_method="slic",
                image_name=None,
                path_to_save=None,
                pred_name='--'):
    np.random.seed(0)

    explainer = AnchorImage(lambda x: model.predict(x), image.shape,
                            segmentation_fn=lambda x: skimage.segmentation.slic(x, n_segments=15,
                                                                                compactness=20,
                                                                                sigma=.5),
                            images_background=None,
                            seed=0)



    explanation = explainer.explain(image,
                                    threshold=threshold,
                                    p_sample=.05,
                                    tau=0.25,
                                    random=0,
                                    )

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(image)

    ax[0, 1].imshow(explanation.segments)
    ax[0, 1].set_title("segments")

    ax[1, 0].imshow(explanation.anchor)
    # ax[1, 0].set_title("explanation")

    explainer = AnchorImage(lambda x: model.predict(x), image.shape,
                            segmentation_fn=lambda x: skimage.segmentation.quickshift(x,
                                                                                      kernel_size=4,
                                                                                      max_dist=200,
                                                                                      ratio=0.2,
                                                                                      random_seed=0),
                            images_background=None,
                            seed=0)

    # explainer = AnchorImage(lambda x: model.predict(x), image.shape,
    #                         segmentation_fn=lambda x: skimage.segmentation.watershed(sobel(rgb2gray(x)), markers=250, compactness=0.1),
    #                         images_background=None,
    #                         seed=0)


    explanation = explainer.explain(image,
                                    threshold=threshold,
                                    p_sample=.5,
                                    tau=0.25,
                                    random=0,
                                    )

    ax[1, 1].imshow(explanation.anchor)
    # ax[1, 1].set_title("explanation")







    fig.suptitle("Anchor explainer " + str(threshold) + "\nClass: " + pred_name)

    for a in ax.ravel():
        a.set_axis_off()

    if path_to_save:
        plt.savefig(path_to_save +
                    image_name +
                    '_anchor_' + str(threshold) + '.png')

    plt.show()
