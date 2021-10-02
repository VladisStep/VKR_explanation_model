import numpy
from keras.models import load_model
from matplotlib import pyplot as plt
from Lime.Parkinson_drawings.data import load_data_parkinson_drawings_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
from lime import lime_image
from skimage.segmentation import mark_boundaries

def lime_vgg16():
    IMG_WIDTH, IMG_HEIGHT = (300, 300)
    # train, test = load_data_parkinson_drawings_model()

    # We use LIME to see at what the trained model is "looking" in the decision-making process.
    path_to_project = '/Users/vladis_step/VKR_explanation_model/'
    path_to_img_to_explain = path_to_project + "Datasets/pd/parkinsons-drawings/spiral/testing/healthy/V01HE01.png"
    chosen_img = image.img_to_array(load_img(path=path_to_img_to_explain,
                                             target_size=(IMG_WIDTH, IMG_HEIGHT))).astype('double') / 255
    # x - преобразованный массив к виду, которая наша модель может "понять"
    x = numpy.expand_dims(chosen_img, axis=0)

    model = load_model(path_to_project + "Models/parkinson_model.h5")
    prediction = model.predict(x)
    print("Prediction. healthy:", int(prediction[0][0] * 100), "%, parkinson:", int(prediction[0][1] * 100), "%")
    pred_for_title = [prediction[0][1] * 100, "parkinson"] if prediction[0][0] < prediction[0][1] \
        else [prediction[0][0] * 100, "healthy"]

    num_samples = 10
    explainer = lime_image.LimeImageExplainer(random_state=42)  # Explains predictions on Image (i.e. matrix) data
    explanation = explainer.explain_instance(chosen_img,
                                             model.predict,
                                             top_labels=1,
                                             # hide_color=0,
                                             # num_samples – size of the neighborhood to learn the linear model
                                             num_samples=num_samples,
                                             random_seed=42)

    temp, mask = explanation.get_image_and_mask(model.predict(chosen_img.reshape((1, 300, 300, 3))).argmax(axis=1)[0],
                                                positive_only=False, hide_rest=False)

    plt.axis("off")
    plt.title("Explanation: " + str(round(pred_for_title[0], 3)) + "% sure it's " + str(pred_for_title[1]))
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.savefig(path_to_project + 'Lime/Parkinson_drawings/Graphs/VGG16_Lime_Explanation.png')
    plt.show()

    # print("The prediction of the explanation model on the original instance: " + explanation.local_pred)

    # Select the same class explained on the figures above.
    ind = model.predict(chosen_img.reshape((1, 300, 300, 3))).argmax(axis=1)[0]

    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = numpy.vectorize(dict_heatmap.get)(explanation.segments)

    # Plot. The visualization makes more sense if a symmetrical colorbar is used.
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.savefig(path_to_project + 'Lime/Parkinson_drawings/Graphs/VGG16_Lime_Heatmap.png')
    plt.show()


if __name__ == "__main__":
    lime_vgg16()
