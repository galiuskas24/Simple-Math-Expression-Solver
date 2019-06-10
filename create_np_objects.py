import cv2
from os import listdir
import numpy as np
import json
import math

# -------------PARAMS-------------------
SYMBOLS_PATH = 'utility/symbols.json'
NEW_IMAGE_SIZE = 48
PAD_WITH = 255  # 255 -> white ; 0 -> black

TRAIN_DIR = 'dataset_divided/train'
TEST_DIR = 'dataset_divided/test'

# without .npy
TRAIN_IMAGES = 'utility/train_images'
TRAIN_LABELS = 'utility/train_labels'
TEST_IMAGES = 'utility/test_images'
TEST_LABELS = 'utility/test_labels'
# ----------------------------------------------


def get_padded_image(image):
    rows, columns = image.shape
    real_size = NEW_IMAGE_SIZE
    col_pad_ceil = int(math.ceil((real_size - columns) / 2.))
    col_pad_floor = int(math.floor((real_size - columns) / 2.))
    rows_pad_ceil = int(math.ceil((real_size - rows) / 2.))
    rows_pad_floor = int(math.floor((real_size - rows) / 2.))
    pad_size = ((rows_pad_ceil, rows_pad_floor), (col_pad_ceil, col_pad_floor))
    return np.pad(image, pad_size, 'constant', constant_values=PAD_WITH)


if __name__ == '__main__':
    # load dictionary
    with open(SYMBOLS_PATH) as f:
        symbols_dict = json.load(f)

    train_data = (TRAIN_DIR, TRAIN_IMAGES, TRAIN_LABELS)
    test_data = (TEST_DIR, TEST_IMAGES, TEST_LABELS)

    for directory, name_img, name_label in [train_data, test_data]:

        print('Loading data from ', directory)
        images = []
        labels = []

        for file in listdir(directory):
            image = cv2.imread(directory + '/' + file, 0)

            # Prepare image for training
            image = get_padded_image(image)
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
            image = image.astype(np.float32) / 255.
            image = image.flatten()
            images.append(image)

            if len(labels) % 10000 == 0:
                print('Loaded: {} images.'.format(len(labels)))

            num, class_type = file.split('.')[0].split('_')
            labels.append(int(symbols_dict[class_type]))

        np.save(name_img, np.asarray(images))
        np.save(name_label, np.asarray(labels))

