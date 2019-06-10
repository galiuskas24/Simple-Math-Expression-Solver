from classes.solver import Solver
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

model_path = 'models/mnist_convnet_model_48x48_future'
solver = Solver(
    model_dir=model_path,
    labels_file='utility/labels.txt',
    bb_plot=False,
)

image_path = 'test_expressions/pow3.jpg'
image = io.imread(image_path)

start_time = time.time()
latex, result = solver.solve(image)
print('LaTeX:', latex, '\nResult:', result)
print('Time elapsed: {0:.2f} seconds.'.format(time.time() - start_time))

solver.plot_prediction()