import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from classes import network as net
import cv2
import math

from classes.boundingBox import BoundingBox
from classes.expression import Expression
from classes.filters import standard_filter


class Solver:

    def __init__(
            self,
            model_dir,
            labels_file,
            use_train_data=False,
            train_mean=None,
            train_std_dev=None,
            image_filter=standard_filter,
            bb_plot=False
            ):
        # Constants
        self.__use_train_data = use_train_data
        self.__train_mean = train_mean
        self.__train_std_dev = train_std_dev
        self.__bb_ploting = bb_plot
        self.__plot_data = None
        self.__bb_size_threshold = 2  # ignore 2x2 bb
        self.__bb_color = [255, 0, 0]  # red
        self.filter = image_filter # must be public

        # Load labels
        self.__labels = []
        self.__latex = []
        with open(labels_file) as f:
            for line in f.readlines():
                if line.startswith('END'): break
                line = line.rstrip()
                label, latex = line.split()
                self.__labels.append(label)
                self.__latex.append(latex)

        self.__labels_dic = dict(zip(range(len(self.__labels)), self.__labels))

        # Load model
        net.NUM_OF_LABELS = len(self.__labels)
        self.__classifier = tf.estimator.Estimator(
            model_fn=net.cnn_model_fn,
            model_dir=model_dir
        )

    def solve(self, image):
        """
        This method recognize equation (only constants) and solve it.
        :param image:  non-filtered image (original - 3 channel)
        :return: latex code and result of equation
        """
        # ---------------LEXICAL ANALYSIS---------------
        bounding_boxes = self.__get_bounding_boxes(image)

        # Prepare bounding boxes for classification
        self.__prepare_bb_for_classification(bounding_boxes)

        # Flatten images
        my_eval_data = np.array([bb.area_norm.flatten() for bb in bounding_boxes])

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': my_eval_data},
            shuffle=False
        )

        # Prediction
        predictions = self.__classifier.predict(input_fn=eval_input_fn)

        for prediction, bb in zip(predictions, bounding_boxes):
            index = prediction['classes']
            bb.add_prediction(
                symbol=self.__labels_dic[index],
                accuracy=prediction['probabilities'][index]
            )

        # Sort image left to right
        bounding_boxes = sorted(bounding_boxes, key=lambda x: (x.xmin, x.ymin))
        self.__plot_data = [(x.symbol, x.area_norm) for x in bounding_boxes]

        # Plotting
        if self.__bb_ploting: self.plot_prediction()

        # ---------------SYNTAX AND SEMANTIC ANALYSIS---------------
        latex, result = Expression(symbols=bounding_boxes).resolve()

        return latex, result

    def plot_prediction(self):
        # Plot image with all bb
        plt.figure(figsize=(20, 10))
        plt.imshow(self.__image_with_bb, cmap="gray")
        plt.show()

        # Plot separated
        bb_len = math.ceil(math.sqrt(len(self.__plot_data)))
        f, ar = plt.subplots(bb_len, bb_len, figsize=(10,10))
        f.suptitle('Predictions', fontsize=20, y=0.1)
        printed = 0

        for i in range(bb_len):
            for j in range(bb_len):

                if printed < len(self.__plot_data):
                    symbol, img = self.__plot_data[printed]
                    ar[i, j].imshow(img, cmap="gray")
                    ar[i, j].axis('off')
                    ar[i, j].set_title(symbol, fontsize=15)
                    printed += 1
                else:
                    ar[i, j].axis('off')

        f.tight_layout()
        plt.show()

    def __merge_bb_with_image(self, bBoxes, image):
        image_arr = np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

        for bb in bBoxes:
            image_arr[bb.ymin:bb.ymax, bb.xmax-1] = self.__bb_color
            image_arr[bb.ymin:bb.ymax, bb.xmin] = self.__bb_color
            image_arr[bb.ymin, bb.xmin:bb.xmax] = self.__bb_color
            image_arr[bb.ymax - 1, bb.xmin:bb.xmax] = self.__bb_color

        return image_arr

    def __get_bounding_boxes(self, image):
        bounding_boxes = []
        real_image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_with_recognize_filter = self.filter(image)

        _, contours, _ = cv2.findContours(image_with_recognize_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove too small bounding boxes
        id = 0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if min(width, height) > self.__bb_size_threshold:
                bounding_boxes.append(BoundingBox(id, x, y, width, height, real_image_gray))
                id += 1

        # Create image with bounding boxes
        self.__image_with_bb = self.__merge_bb_with_image(bounding_boxes, image_with_recognize_filter)

        return bounding_boxes

    def __prepare_bb_for_classification(self, bounding_boxes):
        # Preparation must be like train images preparation (same size, type)

        for bb in bounding_boxes:
            # resize and add padding to bounding box
            bb.normalize()

            # Apply filter on bounding box
            _, bb.area_norm = cv2.threshold(bb.area_norm, 127, 255, cv2.THRESH_BINARY_INV)

            if self.__use_train_data:
                # if we have train data this is another type of normalization (NOT USED HERE)
                bb.area_norm -= self.__train_mean
                bb.area_norm /= self.__train_std_dev

            else:
                # Post-normalization
                bb.area_norm = bb.area_norm.astype(np.float32) / 255.

