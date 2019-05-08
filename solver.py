import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import network as net
import cv2
from boundingBox import BoundingBox
from expression import Expression
from filters import standard_filter


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
        self.__bb_plot = bb_plot
        self.__bb_size_threshold = 2  # ignore 2x2 bb
        self.__bb_color = [255, 0, 0]  # red
        self.filter = image_filter

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
        This method recognize equation with constants and solve it.
        :param image:  non-filtered image (original - 3 channel)
        :return: latex code and result of equation
        """
        # ----------LEXICAL ANALYSIS-------------
        # Find bounding boxes
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filtered_image = self.filter(image)
        bBoxes = self.__get_bounding_boxes(gray_image, filtered_image)

        if self.__bb_plot:
            new_img = self.merge_bb_with_image(bBoxes, filtered_image)
            plt.figure(figsize=(20, 10))
            plt.imshow(new_img, cmap="gray")
            plt.show()

        # Prepare bounding boxes for classification
        self.__prepare_bb_for_classification(bBoxes)
        my_eval_data = np.array([bb.area_norm.flatten() for bb in bBoxes])

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': my_eval_data},
            shuffle=False
        )

        # Prediction
        predictions = self.__classifier.predict(input_fn=eval_input_fn)

        for prediction, bb in zip(predictions, bBoxes):
            index = prediction['classes']
            bb.add_prediction(
                symbol=self.__labels_dic[index],
                accuracy=prediction['probabilities'][index]
            )

        # ----------SYNTAX AND SEMANTIC ANALYSIS---------------
        latex, result = Expression(symbols=bBoxes).resolve()
        return latex, result

    def merge_bb_with_image(self, bBoxes, image):
        image_arr = np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

        for bb in bBoxes:
            image_arr[bb.ymin:bb.ymax, bb.xmax-1] = self.__bb_color
            image_arr[bb.ymin:bb.ymax, bb.xmin] = self.__bb_color
            image_arr[bb.ymin, bb.xmin:bb.xmax] = self.__bb_color
            image_arr[bb.ymax - 1, bb.xmin:bb.xmax] = self.__bb_color

        return image_arr

    def __get_bounding_boxes(self, gray_image, filtered_image):
        bounding_boxes = []
        _, contours, _ = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove too small bounding boxes
        id = 0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if min(width, height) > self.__bb_size_threshold:
                bounding_boxes.append(BoundingBox(id, x, y, width, height, gray_image))
                id += 1

        return bounding_boxes

    def __prepare_bb_for_classification(self, bounding_boxes):
        # Preparation must be like train images preparation (same size, type)

        for bb in bounding_boxes:
            # resize and add padding to bounding box
            bb.normalize()

            # if we have train data this is another type of normalization
            if self.__use_train_data:
                bb.area_norm -= self.__train_mean
                bb.area_norm /= self.__train_std_dev

            # Apply filter on bounding box
            _, bb.area_norm = cv2.threshold(bb.area_norm, 127, 255, cv2.THRESH_BINARY_INV)

            # Post-normalization
            bb.area_norm = bb.area_norm.astype(np.float32) / 255.
