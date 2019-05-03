import tensorflow as tf
import cv2
import math
import numpy as np
from network import cnn_model_fn
from skimage.util import invert
from scipy.ndimage.measurements import center_of_mass


class Solver:
    def __init__(self, model_dir, labels_file, train_mean, train_std_dev):
        self.__train_mean = train_mean
        self.__train_std_dev = train_std_dev

        # Load model
        self.__classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=model_dir
        )

        # Load labels
        self.__labels = []
        self.__latex = []
        with open(labels_file) as f:
            for line in f.readlines():
                line = line.rstrip()
                label, latex = line.split()
                self.__labels.append(label)
                self.__latex.append(latex)

        # Constants
        self.__bb_size_threshold = 8
        self.__bb_color = [255, 0, 0]

    def solve(self, image):
        """
        bb -> bounding box
        :param image:
        :return:
        """
        # ----------LEXICAL ANALYSIS-------------
        bBoxes = self.__get_bounding_boxes(image)
        self.__normalize_bb(bBoxes)

        # Create input for prediction
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.array([bb.area_norm for bb in bBoxes])},
            shuffle=False
        )

        # Prediction
        predictions = self.__classifier.predict(input_fn=eval_input_fn)

        for prediction, bb in zip(predictions, bBoxes):
            index = prediction['classes']
            bb.symbol = index
            bb.sym_accuracy = prediction['probabilities'][index]

        print('aaaa')

        # ----------SYNTAX ANALYSIS---------------
        bBoxes = sorted(bBoxes, key=lambda x: (x.xmin, x.ymin))

        # ----------SEMATNIC ANALYSIS--------------



        return "latex", "rezz"

    def get_image_with_bb(self, image):
        bBoxes = self.__get_bounding_boxes(image)

        image_arr = np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

        for bb in bBoxes:
            image_arr[bb.ymin:bb.ymax, bb.xmax-1] = self.__bb_color
            image_arr[bb.ymin:bb.ymax, bb.xmin] = self.__bb_color
            image_arr[bb.ymin, bb.xmin:bb.xmax] = self.__bb_color
            image_arr[bb.ymax - 1, bb.xmin:bb.xmax] = self.__bb_color

        return image_arr

    def __get_bounding_boxes(self, image):
        bounding_boxes = []
        _, image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove too small bounding boxes
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if min(width, height) > self.__bb_size_threshold:
                bounding_boxes.append(_BoundingBox(x, y, width, height, image))

        return bounding_boxes

    def __normalize_bb(self, bounding_boxes):

        for bb in bounding_boxes:
            bb.normalize()

            bb.area_norm -= self.__train_mean
            bb.area_norm /= self.__train_std_dev

class _BoundingBox:

    def __init__(self, x, y, width, height, image):
        self.xmin = x
        self.xmax = x + width
        self.ymin = y
        self.ymax = y + height
        self.xcenter = x + width/2
        self.ycenter = y + height/2
        self.area = image[y:(y+height), x:(x+height)]
        self.area_norm = None
        self.symbol = None
        self.sym_accuracy = None

        # Constants
        self.__norm_resize = 40
        self.__norm_pad_size = 48

    def add_prediction(self, symbol, accuracy):
        self.symbol = symbol
        self.sym_accuracy = accuracy

    def normalize(self):
        area = np.copy(self.area).astype(np.float32)

        # 1.------> normalize all to 0-1
        area /= np.max(area)

        # 2.------> resize to 40x40
        rows, columns = area.shape
        new_size = self.__norm_resize

        if rows > columns:
            columns = max(int(round(columns*(new_size/rows))), 2)
            rows = new_size

        else:
            rows = max(int(round(rows*(new_size/columns))), 2)
            columns = new_size

        area = cv2.resize(area, (columns, rows))

        # 3.------> pad to 48x48
        real_size = self.__norm_pad_size
        col_pad_ceil = int(math.ceil((real_size-columns)/2.))
        col_pad_floor = int(math.floor((real_size-columns)/2.))
        rows_pad_ceil = int(math.ceil((real_size-rows)/2.))
        rows_pad_floor = int(math.floor((real_size-rows)/2.))
        pad_size = ((col_pad_ceil, col_pad_floor),(rows_pad_ceil, rows_pad_floor))
        real_area = np.pad(area, pad_size, 'constant', constant_values=(1, 1))

        # 4.------> center the mass
        inverted = invert(real_area)
        centerY, centerX = center_of_mass(inverted)
        shiftX = np.round(columns/2. - centerX).astype(int)
        shiftY = np.round(rows/2. - centerY).astype(int)

        dim = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        real_area = cv2.warpAffine(real_area, dim, (columns, rows), borderValue=1)

        self.area_norm = real_area

