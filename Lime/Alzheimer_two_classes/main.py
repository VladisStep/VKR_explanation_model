import AnchorExplainer
import LimeExplainer

anchor_explainer = AnchorExplainer.AnchorExplainer('/Users/vladis_step/VKR_explanation_model/Models/my_alzheimer_model.h5', 176, 208)
lime_explainer = LimeExplainer.LimeExplainer('/Users/vladis_step/VKR_explanation_model/Models/my_alzheimer_model.h5', 176, 208)

img_path = [
    # "Datasets/Alzheimer_two_classes/test/Demented/27 (9).jpg",

    "Datasets/Alzheimer_two_classes/test/Demented/26 (19).jpg",
    "Datasets/Alzheimer_two_classes/test/Demented/26 (20).jpg",
    "Datasets/Alzheimer_two_classes/test/Demented/26 (21).jpg",
    # "Datasets/Alzheimer_two_classes/test/Demented/26 (22).jpg",
    # "Datasets/Alzheimer_two_classes/test/Demented/26 (23).jpg",
    #
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (62).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (63).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (64).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (65).jpg",
    # "Datasets/Alzheimer_two_classes/test/NonDemented/26 (66).jpg",
]

# for i in range(len(img_path)):
#     lime_explainer.get_explain(img_path[i], num_samples=1000, save_fig=True, img_num=i)
#     anchor_explainer.get_explain(img_path[i], save_fig=True, img_num=i)
#     # anchor_explainer.get_explain(img_path[i], segmentation_fn='slic', save_fig=True, img_num=i)


for i in range(5):
    anchor_explainer.get_explain(img_path[0])
