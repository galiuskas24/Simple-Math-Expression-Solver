import math, cv2
import numpy as np
from skimage.util import invert
from scipy.ndimage.measurements import center_of_mass


class BoundingBox:

    def __init__(self, id, x, y, width, height, image):
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

        # Constants
        self.__norm_resize = 42
        self.__norm_pad_size = 48

    def add_prediction(self, symbol, accuracy):
        self.symbol = symbol
        self.sym_accuracy = accuracy

    def normalize(self):
        area = np.copy(self.area).astype(np.float32)

        # 1.------> normalize all to 0-1
        ##area /= np.max(area)

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
        pad_size = ((rows_pad_ceil, rows_pad_floor), (col_pad_ceil, col_pad_floor))
        real_area = np.pad(area, pad_size, 'constant', constant_values=255)
        # rows, columns = real_area.shape
        #
        # # 4.------> center the mass
        # inverted = invert(real_area)
        # centerY, centerX = center_of_mass(inverted)
        # shiftX = np.round(columns/2. - centerX).astype(int)
        # shiftY = np.round(rows/2. - centerY).astype(int)
        #
        # dim = np.float32([[1, 0, shiftX], [0, 1, shiftY]])
        # real_area = cv2.warpAffine(real_area, dim, (columns, rows), borderValue=1)

        self.area_norm = real_area

    def isAbove(self, symbol):
        if symbol.xmin < self.xcenter < symbol.xmax:
            if self.ycenter < symbol.ymin: return True
        return False

    def isUnder(self, symbol):
        if symbol.xmin < self.xcenter < symbol.xmax:
            if self.ycenter > symbol.ymax: return True
        return False

    def updateBorders(self, union):
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