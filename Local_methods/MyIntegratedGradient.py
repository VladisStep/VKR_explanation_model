from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from alibi.utils.visualization import visualize_image_attr


def explanation(model, image, image_name=None, path_to_save=None,
                pred_name='--'):
    predictions = model(np.expand_dims(image, axis=0)).numpy().argmax(axis=1)

    ig = IntegratedGradients(model,
                             layer=None,
                             method="gausslegendre",
                             internal_batch_size = 50)



    method_explanation = ig.explain(np.expand_dims(image, axis=0),
                                    baselines=None,
                                    target=predictions)
    attr = method_explanation.attributions[0]

    # fig, ax = plt.subplots(nrows=2, ncols=2)
    #
    # ax[0, 0].axis("off")
    # visualize_image_attr(attr=None, original_image=image, method='original_image',
    #                      title='Original Image', plt_fig_axis=(fig, ax[0, 0]), use_pyplot=False)
    # ax[0, 1].axis("off")
    # visualize_image_attr(attr=attr.squeeze(), original_image=image, method='blended_heat_map',
    #                      sign='all', show_colorbar=False, title='Overlaid Attributions',
    #                      plt_fig_axis=(fig, ax[0, 1]), use_pyplot=False)
    #
    # ax[1, 0].axis("off")
    # visualize_image_attr(attr=attr.squeeze(), original_image=image, method='blended_heat_map',
    #                      sign='positive', show_colorbar=False, title='positive',
    #                      plt_fig_axis=(fig, ax[1, 0]), use_pyplot=False)
    #
    # ax[1, 1].axis("off")
    # visualize_image_attr(attr=attr.squeeze(), original_image=image, method='blended_heat_map',
    #                      sign='negative', show_colorbar=False, title='negative',
    #                      plt_fig_axis=(fig, ax[1, 1]), use_pyplot=False)
    #
    # fig.suptitle("Integrated gradient explainer" + "\nClass: " + pred_name)
    #
    # if path_to_save:
    #     fig.savefig(path_to_save +
    #                 image_name +
    #                 '_intGrad'+'.png')

    # plt.show()

    return attr.squeeze()
