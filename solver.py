import tensorflow as tf
import cv2
from boundingBox import BoundingBox
import numpy as np
import matplotlib.pyplot as plt
import network as net
from expression import Expression

class Solver:
    def __init__(self, model_dir, labels_file, train_mean, train_std_dev):
        # Constants
        self.__train_mean = train_mean
        self.__train_std_dev = train_std_dev
        self.__bb_size_threshold = 2
        self.__bb_color = [255, 0, 0]

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
        bb -> bounding box
        :param image:
        :return:
        """
        # ----------LEXICAL ANALYSIS-------------
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bBoxes = self.__get_bounding_boxes(image)


        new_img = self.get_image_with_bb(image)
        print("Start bounding boxes: ")
        plt.figure(figsize=(20,10))
        plt.imshow(new_img, cmap="gray")
        plt.show()

        self.__normalize_bb(bBoxes)

        bBoxes = sorted(bBoxes, key=lambda x: (x.xmin, x.ymin))
        # Create input for prediction
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


        # ----------SYNTAX ANALYSIS---------------
        bBoxes = sorted(bBoxes, key=lambda x: (x.xmin, x.ymin))

        latex, rez = Expression(symbols=bBoxes).get_data
        print(latex, '=',rez)
        return latex, rez


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
        _, image_inv = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
        _, contours, _ = cv2.findContours(image_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Remove too small bounding boxes
        id = 0
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)

            if min(width, height) > self.__bb_size_threshold:
                bounding_boxes.append(BoundingBox(id, x, y, width, height, image))
                id += 1

        return bounding_boxes

    def __normalize_bb(self, bounding_boxes):

        for bb in bounding_boxes:
            # plt.figure(figsize=(20, 10))
            # plt.imshow(bb.area, cmap="gray")
            # plt.show()
            #new_image = cv2.resize(bb.area, (28, 28))
            bb.normalize()
            #
            # plt.figure(figsize=(20, 10))
            # plt.imshow(bb.area_norm, cmap="gray")
            # plt.show()


            _, new_image = cv2.threshold(bb.area_norm, 127, 255, cv2.THRESH_BINARY_INV)

            bb.area_norm = new_image
            # plt.figure(figsize=(20, 10))
            # plt.imshow(bb.area_norm, cmap="gray")
            # plt.show()


            bb.area_norm = new_image.astype(np.float32) / 255.

            # plt.figure(figsize=(20, 10))
            # plt.imshow(bb.area_norm, cmap="gray")
            # plt.show()

            #bb.area_norm -= self.__train_mean
            #bb.area_norm /= self.__train_std_dev
