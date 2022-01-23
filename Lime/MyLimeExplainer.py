import numpy
import numpy as np
import skimage
from lime import lime_image
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries


def explanation(model, image, num_samples=100, image_name=None, path_to_save=None, pred_name='--'):
    model_prediction = model.predict(np.expand_dims(image, axis=0))

    np.random.seed(0)

    explainer = lime_image.LimeImageExplainer()  # Explains predictions on Image (i.e. matrix) data
    lime_explanation = explainer.explain_instance(image,
                                                  model.predict,
                                                  top_labels=2,
                                                  hide_color=0,

                                                  segmentation_fn=lambda x: skimage.segmentation.slic(x, n_segments=15,
                                                                                                 compactness=20,
                                                                                                 sigma=.5),
                                                  # num_samples – size of the neighborhood to learn the linear model
                                                  num_samples=num_samples,
                                                  random_seed=0,
                                                  num_features=10)

    temp, mask = lime_explanation.get_image_and_mask(
        model_prediction.argmax(axis=1)[0],
        negative_only=False, positive_only=False, hide_rest=False, num_features=10)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].axis("off")
    ax[0].imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # ax[0].set_title("explanation " + str(model_prediction[0]))


    # Heatmap
    ind = model_prediction.argmax(axis=1)[0]
    dict_heatmap = dict(lime_explanation.local_exp[ind])
    heatmap = numpy.vectorize(dict_heatmap.get)(lime_explanation.segments)
    ax1 = ax[1].imshow(np.array(heatmap, dtype=np.float64), cmap="RdYlGn", vmax=np.array(heatmap, dtype=np.float64).max(), vmin=-np.array(heatmap, dtype=np.float64).max())
    fig.colorbar(ax1, ax=ax[1])
    ax[1].set_title("Heatmap")

    fig.suptitle("Lime explainer" + "\nClass: " + pred_name)
    if path_to_save :
        plt.savefig(path_to_save + image_name + '_lime_' + str(num_samples) + '.png')
    plt.show()
