from solver import Solver
import numpy as np
from skimage import io
import time

# Old model with mean and std_dev
# mean = np.load("utility/train_images_mean.npy")
# std_dev = np.load("utility/train_images_std.npy")
#
# solver = Solver(
#     model_dir='models/model',
#     labels_file='utility/labels.txt',
#     train_mean=mean,
#     train_std_dev=std_dev
# )

solver = Solver(
    model_dir='models/mnist_convnet_model_48x48',
    labels_file='utility/labels.txt',
    bb_plot=False
)

image_path = 'data/test_expressions/new4b.jpg'
image = io.imread(image_path)

start_time = time.time()
latex, result = solver.solve(image)
print('LaTeX:', latex, '\nResult:', result)
print('Time elapsed: {0:.2f} seconds.'.format(time.time() - start_time))
