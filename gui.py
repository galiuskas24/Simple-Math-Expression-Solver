from solver import Solver
import numpy as np
from skimage import io

mean = np.load("utility/train_images_mean.npy")
std_dev = np.load("utility/train_images_std.npy")

solver = Solver(
    model_dir='models/model',
    labels_file='utility/labels.txt',
    train_mean=mean,
    train_std_dev=std_dev
)

image_path = 'utility/vg1.jpg'
image = io.imread(image_path)
latex, result = solver.solve(image)
print('end')