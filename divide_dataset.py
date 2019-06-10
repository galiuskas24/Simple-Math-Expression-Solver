import cv2
from os import listdir
import json

# -------------PARAMS------------------
SYMBOLS_PATH = 'utility/symbols.json'
SOURCE_DIR = 'dataset/'
SAVE_DIR = 'dataset_divided/'
TEST_PART = 0.1
# --------------------------------------

if __name__ == '__main__':
    # Load symbols dictionary
    with open(SYMBOLS_PATH) as f:
        symbols_dict = json.load(f)

    for directory in symbols_dict.keys():
        dir_path = SOURCE_DIR + directory
        dir_files = listdir(dir_path)
        test_size = int(len(dir_files) * TEST_PART)
        i = 0

        for file in dir_files:
            image = cv2.imread(dir_path + '/' + file, 0)

            # save image
            tt_dir = 'test' if i < test_size else 'train'
            new_path = SAVE_DIR + tt_dir + '/' + str(i) + '_' + directory + '.jpg'
            cv2.imwrite(new_path, image)
            i += 1

        print("Finished '" + directory + "' directory.")
