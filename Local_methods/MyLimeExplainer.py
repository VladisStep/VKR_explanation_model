import numpy
import numpy as np
import skimage
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries


def explanation(model,
                image,
                num_samples=100,
                segmentation_method=None,
                image_name=None,
                path_to_save=None,
                pred_name='--'):

    model_prediction = model.predict(np.expand_dims(image, axis=0))

    np.random.seed(0)

    if segmentation_method == None:
        segmentation_fn = lambda x: skimage.segmentation.slic(x, n_segments=15, compactness=20, sigma=.5)
    else:
        segmentation_fn = segmentation_method

    explainer = lime_image.LimeImageExplainer()  # Explains predictions on Image (i.e. matrix) data
    lime_explanation = explainer.explain_instance(image,
                                                  model.predict,
                                                  top_labels=2,
                                                  hide_color=0,

                                                  segmentation_fn = segmentation_fn,
                                                  # num_samples – size of the neighborhood to learn the linear model
                                                  num_samples=10,
                                                  random_seed=0,
                                                  num_features=10)

    lime_explanation_2 = explainer.explain_instance(image,
                                                  model.predict,
                                                  top_labels=2,
                                                  hide_color=0,
                                                  segmentation_fn = segmentation_fn,
                                                  num_samples=100,
                                                  random_seed=0,
                                                  num_features=10)
    lime_explanation_3 = explainer.explain_instance(image,
                                                    model.predict,
                                                    top_labels=2,
                                                    hide_color=0,
                                                    segmentation_fn=segmentation_fn,
                                                    num_samples=500,
                                                    random_seed=0,
                                                    num_features=10)
    lime_explanation_4 = explainer.explain_instance(image,
                                                    model.predict,
                                                    top_labels=2,
                                                    hide_color=0,
                                                    segmentation_fn=segmentation_fn,
                                                    num_samples=1000,
                                                    random_seed=0,
                                                    num_features=10)
    lime_explanation_5 = explainer.explain_instance(image,
                                                    model.predict,
                                                    top_labels=2,
                                                    hide_color=0,
                                                    segmentation_fn=segmentation_fn,
                                                    num_samples=2000,
                                                    random_seed=0,
                                                    num_features=10)
    lime_explanation_6 = explainer.explain_instance(image,
                                                    model.predict,
                                                    top_labels=2,
                                                    hide_color=0,
                                                    segmentation_fn=segmentation_fn,
                                                    num_samples=5000,
                                                    random_seed=0,
                                                    num_features=10)

# --------------------------------------------------------------------------------------------------------#

    fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(15, 4))

    temp, mask = lime_explanation.get_image_and_mask(
            model_prediction.argmax(axis=1)[0],
            negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    ax[0].axis("off")
    ax[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax[0].set_title("num_samples = 10")

    temp, mask = lime_explanation_2.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    ax[1].axis("off")
    ax[1].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax[1].set_title("num_samples = 100")

    temp, mask = lime_explanation_3.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    ax[2].axis("off")
    ax[2].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax[2].set_title("num_samples = 500")

    temp, mask = lime_explanation_4.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    ax[3].axis("off")
    ax[3].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax[3].set_title("num_samples = 1000")


    temp, mask = lime_explanation_5.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    ax[4].axis("off")
    ax[4].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax[4].set_title("num_samples = 2000")

    temp, mask = lime_explanation_6.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    ax[5].axis("off")
    ax[5].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax[5].set_title("num_samples = 5000")

#--------------------------------------------------------------------------------------------------------#


    # fig, ax = plt.subplots(nrows=4, ncols=2, figsize = (7, 12))
    # ax[0, 0].axis("off")
    # ax[0, 0].imshow(image)
    # ax[0, 0].set_title("Original")
    # ax[0, 1].axis("off")
    # ax[0, 1].imshow(mark_boundaries(image, segmentation_fn(image)))
    # ax[0, 1].set_title("Segmentation")
    #
    # temp, mask = lime_explanation.get_image_and_mask(
    #     model_prediction.argmax(axis=1)[0],
    #     negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    # ax[1, 0].axis("off")
    # ax[1, 0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # ax[1, 0].set_title("num_samples = 200")
    # # Heatmap
    # ind = model_prediction.argmax(axis=1)[0]
    # dict_heatmap = dict(lime_explanation.local_exp[ind])
    # heatmap = numpy.vectorize(dict_heatmap.get)(lime_explanation.segments)
    # ax1 = ax[1, 1].imshow(np.array(heatmap, dtype=np.float64), cmap="RdYlGn", vmax=np.array(heatmap, dtype=np.float64).max(), vmin=-np.array(heatmap, dtype=np.float64).max())
    # fig.colorbar(ax1, ax=ax[1])
    # ax[1, 1].set_title("Heatmap")
    #
    # temp, mask = lime_explanation_2.get_image_and_mask(
    #     model_prediction.argmax(axis=1)[0],
    #     negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    # ax[2, 0].axis("off")
    # ax[2, 0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # ax[2, 0].set_title("num_samples = 200")
    # # Heatmap
    # ind = model_prediction.argmax(axis=1)[0]
    # dict_heatmap = dict(lime_explanation_2.local_exp[ind])
    # heatmap = numpy.vectorize(dict_heatmap.get)(lime_explanation_2.segments)
    # ax1 = ax[2, 1].imshow(np.array(heatmap, dtype=np.float64), cmap="RdYlGn",
    #                       vmax=np.array(heatmap, dtype=np.float64).max(),
    #                       vmin=-np.array(heatmap, dtype=np.float64).max())
    # fig.colorbar(ax1, ax=ax[2])
    # ax[2, 1].set_title("Heatmap")
    #
    # temp, mask = lime_explanation_3.get_image_and_mask(
    #     model_prediction.argmax(axis=1)[0],
    #     negative_only=False, positive_only=False, hide_rest=False, num_features=10)
    # ax[3, 0].axis("off")
    # ax[3, 0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # ax[3, 0].set_title("num_samples = 200")
    # # Heatmap
    # ind = model_prediction.argmax(axis=1)[0]
    # dict_heatmap = dict(lime_explanation_3.local_exp[ind])
    # heatmap = numpy.vectorize(dict_heatmap.get)(lime_explanation_3.segments)
    # ax1 = ax[3, 1].imshow(np.array(heatmap, dtype=np.float64), cmap="RdYlGn",
    #                       vmax=np.array(heatmap, dtype=np.float64).max(),
    #                       vmin=-np.array(heatmap, dtype=np.float64).max())
    # fig.colorbar(ax1, ax=ax[3])
    # ax[3, 1].set_title("Heatmap")



    # # Heatmap
    # ind = model_prediction.argmax(axis=1)[0]
    # dict_heatmap = dict(lime_explanation.local_exp[ind])
    # heatmap = numpy.vectorize(dict_heatmap.get)(lime_explanation.segments)
    # ax1 = ax[1, 1].imshow(np.array(heatmap, dtype=np.float64), cmap="RdYlGn", vmax=np.array(heatmap, dtype=np.float64).max(), vmin=-np.array(heatmap, dtype=np.float64).max())
    # fig.colorbar(ax1, ax=ax[1])
    # # ax[1, 1].set_title("Heatmap")
    #
    # fig.suptitle("Local_methods explainer" + "\nClass: " + pred_name)
    # if path_to_save :
    #     plt.savefig(path_to_save + image_name + '_lime_' + str(num_samples) + '.png')
    plt.show()

    return mark_boundaries(temp / 2 + 0.5, mask)
