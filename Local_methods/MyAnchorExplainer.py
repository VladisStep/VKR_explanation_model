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
                segmentation_method=None,
                image_name=None,
                path_to_save=None,
                pred_name='--'):
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

    # ---------------------------------------------------------------------------- #
    # import time
    #
    # start_time = time.time()
    # explanation = explainer.explain(image,
    #                                 p_sample=.5,
    #                                 threshold=0.95,
    #                                 tau=0.25,
    #                                 beam_size=2,
    #                                 batch_size=100,
    #                                 coverage_samples=1000,
    #                                 random=0
    #                                 )
    # print("--- %s seconds ---" % (time.time() - start_time))
    #
    #
    # fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(15, 4), sharex=True, sharey=True)
    #
    # ax[0].imshow(explanation.anchor)
    # ax[0].set_title("coverage_samples = 1000\ntime = " + str(round(time.time() - start_time, 1)) + " sec")
    # ax[0].axis("off")
    #
    # start_time = time.time()
    # explanation = explainer.explain(image,
    #                                 p_sample=.5,
    #                                 threshold=0.95,
    #                                 tau=0.25,
    #                                 beam_size=2,
    #                                 batch_size=100,
    #                                 coverage_samples=5000,
    #                                 random=0
    #                                 )
    # print("--- %s seconds ---" % (time.time() - start_time))
    # ax[1].imshow(explanation.anchor)
    # ax[1].set_title("coverage_samples = 5000\ntime = " + str(round(time.time() - start_time, 1)) + " sec")
    # ax[1].axis("off")
    #
    # start_time = time.time()
    # explanation = explainer.explain(image,
    #                                 p_sample=.5,
    #                                 threshold=0.95,
    #                                 tau=0.25,
    #                                 beam_size=2,
    #                                 batch_size=100,
    #                                 coverage_samples=10000,
    #                                 random=0
    #                                 )
    # print("--- %s seconds ---" % (time.time() - start_time))
    # ax[2].imshow(explanation.anchor)
    # ax[2].set_title("coverage_samples = 10_000\ntime = " + str(round(time.time() - start_time, 1)) + " sec")
    # ax[2].axis("off")
    #
    # start_time = time.time()
    # explanation = explainer.explain(image,
    #                                 p_sample=.5,
    #                                 threshold=0.95,
    #                                 tau=0.25,
    #                                 beam_size=2,
    #                                 batch_size=500,
    #                                 coverage_samples=20000,
    #                                 random=0
    #                                 )
    # print("--- %s seconds ---" % (time.time() - start_time))
    # ax[3].imshow(explanation.anchor)
    # ax[3].set_title("coverage_samples = 20_000\ntime = " + str(round(time.time() - start_time, 1)) + " sec")
    # ax[3].axis("off")
    #
    # start_time = time.time()
    # explanation = explainer.explain(image,
    #                                 p_sample=.5,
    #                                 threshold=0.95,
    #                                 tau=0.25,
    #                                 beam_size=2,
    #                                 batch_size=1000,
    #                                 coverage_samples=50000,
    #                                 random=0
    #                                 )
    # print("--- %s seconds ---" % (time.time() - start_time))
    # ax[4].imshow(explanation.anchor)
    # ax[4].set_title("coverage_samples = 50_000\ntime = " + str(round(time.time() - start_time, 1)) + " sec")
    # ax[4].axis("off")
# ---------------------------------------------------------------------------- #
    # start_time = time.time()
    # explanation = explainer.explain(image,
    #                                 p_sample=.5,
    #                                 threshold=0.95,
    #                                 tau=0.25,
    #                                 beam_size=2,
    #                                 batch_size=2000,
    #                                 coverage_samples=10000,
    #                                 random=0
    #                                 )
    # print("--- %s seconds ---" % (time.time() - start_time))
    # ax[5].imshow(explanation.anchor)
    # ax[5].set_title("batch_size = 2000\ntime = " + str(round(time.time() - start_time, 1)) + " sec")
    # ax[5].axis("off")

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
    #
    # ax[0, 0].imshow(image)
    #
    # ax[0, 1].imshow(mark_boundaries(image, segmentation_fn(image)))
    # ax[0, 1].set_title("segments")
    #
    # ax[1, 0].imshow(explanation.anchor)
    # ax[1, 0].set_title("explanation")
    #
    # ax[1, 1].axis("off")
    #
    # fig.suptitle("Anchor explainer " + str(threshold) + "\nClass: " + pred_name)
    #
    # for a in ax.ravel():
    #     a.set_axis_off()
    #
    # if path_to_save:
    #     plt.savefig(path_to_save +
    #                 image_name +
    #                 '_anchor_' + str(threshold) + '.png')

    plt.show()

    return explanation.anchor
