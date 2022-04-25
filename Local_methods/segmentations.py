
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img

import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

import my_utils

PATH_TO_DATASET = my_utils.path_to_project + '/Datasets/dogs-vs-cats'
CONTENT_DIR = PATH_TO_DATASET + '/content'
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/valid'

IMAGE_SHAPE = 128

image_path = "/dog/dog.25.jpg"

def processImgToModel(image_path):
    return image.img_to_array(load_img(
        path=image_path,
        target_size=(IMAGE_SHAPE, IMAGE_SHAPE))
    ).astype("double") / 255.0



img = processImgToModel(TRAIN_DIR + image_path)

segments_fz = felzenszwalb(img, scale=50, sigma=0.5, min_size=50)
segments_slic = slic(img, n_segments=50, compactness=10, sigma=1,
                     start_label=1)
segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
gradient = sobel(rgb2gray(img))
segments_watershed = watershed(gradient, markers=50, compactness=0.001)

print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].set_title('SLIC')
ax[1, 0].imshow(mark_boundaries(img, segments_quick))
ax[1, 0].set_title('Quickshift')
ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
ax[1, 1].set_title('Compact watershed')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()