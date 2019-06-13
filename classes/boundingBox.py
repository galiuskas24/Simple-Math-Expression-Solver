import math
import cv2
import numpy as np


class BoundingBox:

    def __init__(self, id, x, y, width, height, image):

        # Constants
        self.__norm_resize = 42
        self.__norm_with_pad_size = 48

        # Variables
        self.id = id
        self.xmin = x
        self.xmax = x + width
        self.ymin = y
        self.ymax = y + height
        self.xcenter = x + width/2
        self.ycenter = y + height/2

        self.symbol = None
        self.sym_accuracy = None

        self.value = None # real value
        self.type = None # real type
        self.latex = None # real latex

        self.area = image[y:(y+height), x:(x+width)]
        self.area_norm = None

    def add_prediction(self, symbol, accuracy):
        self.symbol = symbol
        self.sym_accuracy = accuracy
        self.type = 'SYMBOL'

    def normalize(self):
        area = np.copy(self.area).astype(np.float32)

        # For very long symbols
        a = (self.xmax - self.xmin) / (self.ymax - self.ymin)
        if not 0.25 < a < 4:
            kernel = np.ones((2, 2), np.uint8)
            area = cv2.erode(area, kernel, iterations=2)

            kernel = np.ones((1, 1), np.uint8)
            area = cv2.dilate(area, kernel, iterations=1)

        # Resize image
        rows, columns = area.shape
        new_size = self.__norm_resize
        if rows > columns:
            columns = max(int(round(columns*(new_size/rows))), 2)
            rows = new_size

        else:
            rows = max(int(round(rows*(new_size/columns))), 2)
            columns = new_size

        area = cv2.resize(area, (columns, rows))



        # Add padding to image
        real_size = self.__norm_with_pad_size
        col_pad_ceil = int(math.ceil((real_size-columns)/2.))
        col_pad_floor = int(math.floor((real_size-columns)/2.))
        rows_pad_ceil = int(math.ceil((real_size-rows)/2.))
        rows_pad_floor = int(math.floor((real_size-rows)/2.))
        pad_size = ((rows_pad_ceil, rows_pad_floor), (col_pad_ceil, col_pad_floor))
        self.area_norm = np.pad(area, pad_size, 'constant', constant_values=255)

    def is_above(self, symbol):
        if symbol.xmin < self.xcenter < symbol.xmax:
            if self.ycenter < symbol.ymin: return True
        return False

    def is_under(self, symbol):
        if symbol.xmin < self.xcenter < symbol.xmax:
            if self.ycenter > symbol.ymax: return True
        return False

    def is_above_and_right(self, symbol):
        return symbol.xcenter < self.xmin and symbol.ycenter > self.ymax


    def update_borders(self, union):
        self.xmin = min([bb.xmin for bb in union])
        self.xmax = max([bb.xmax for bb in union])
        self.ymin = min([bb.ymin for bb in union])
        self.ymax = max([bb.ymax for bb in union])

        self.xcenter = (self.xmin + self.xmax)/2.
        self.ycenter = (self.ymin + self.ymax)/2.

    def __eq__(self, other):
        return self.id == other.id

    def __str__(self):
        return self.symbol

    def __hash__(self):
        return hash(self.id)
