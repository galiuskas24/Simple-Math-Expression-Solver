from solver import Solver
import numpy as np
from skimage import io
import time

mean = np.load("utility/train_images_mean.npy")
std_dev = np.load("utility/train_images_std.npy")

solver = Solver(
    model_dir='models/model',
    labels_file='utility/labels.txt',
    train_mean=mean,
    train_std_dev=std_dev
)

image_path = 'data/test_expressions/123.jpg'
image = io.imread(image_path)

aa = time.time()
latex, result = solver.solve(image)
bb = time.time()-aa
print('end', bb)