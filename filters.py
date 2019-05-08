import cv2
import numpy as np

"""
Input: 3-channel image
Filters convert input image and prepare it for bounding box extraction.
Output:
    Image with:
        Black-> Area around symbols 
        White-> Symbols 
"""

def standard_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, new_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    #_, new_image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)
    return new_image


def basic_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image, (5, 5), 0)
    thresh, img_bin = cv2.threshold(image, 128, 255, cv2.THRESH_OTSU)
    _, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY_INV)
    return img_bin

def future_filter(image):
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def old_filter(image):
    # convert to gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # trunc
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)

    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 13, 4)

    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # remove noise
    image = cv2.medianBlur(image, 7)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    return image
