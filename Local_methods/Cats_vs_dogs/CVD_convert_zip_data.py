import os
from sklearn.model_selection import train_test_split
import shutil
import tqdm

import my_utils

# https://www.kaggle.com/c/dogs-vs-cats/data

PATH_TO_DATA = my_utils.path_to_project + '/Datasets/dogs-vs-cats'
CONTENT_DIR = PATH_TO_DATA + '/content'
TRAIN_DIR = CONTENT_DIR + '/train'
VALID_DIR = CONTENT_DIR + '/valid'


if not os.path.exists(CONTENT_DIR):
    # Extract dataset
    import zipfile
    with zipfile.ZipFile(PATH_TO_DATA + '/train.zip', 'r') as zipf:
        zipf.extractall(CONTENT_DIR)

    # Split cats and dogs images to train and valid datasets
    img_filenames = os.listdir(TRAIN_DIR)
    dog_filenames = [fn for fn in img_filenames if fn.startswith('dog')]
    cat_filenames = [fn for fn in img_filenames if fn.startswith('cat')]
    dataset_filenames = train_test_split(
        dog_filenames, cat_filenames, test_size=0.1, shuffle=True, random_state=42
    )
    # Move images
    make_dirs = [d + a for a in ['/dog', '/cat'] for d in [TRAIN_DIR, VALID_DIR]]
    for dir, fns in zip(make_dirs, dataset_filenames):
        os.makedirs(dir, exist_ok=True)
        for fn in tqdm.tqdm(fns):
            shutil.move(os.path.join(TRAIN_DIR, fn), dir)
        print('elements in {}: {}'.format(dir, len(os.listdir(dir))))