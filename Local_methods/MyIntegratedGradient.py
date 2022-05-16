from alibi.explainers import IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from alibi.utils.visualization import visualize_image_attr

def explanation(model,
                image
                ):
    predictions = model(np.expand_dims(image, axis=0)).numpy().argmax(axis=1)

    ig = IntegratedGradients(model, method="gausslegendre", n_steps=50)
    method_explanation = ig.explain(np.expand_dims(image, axis=0), baselines=None, target=predictions)
    attr = method_explanation.attributions[0]

    return attr.squeeze()
