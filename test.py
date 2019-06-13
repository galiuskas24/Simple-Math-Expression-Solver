from classes.solver import Solver
from skimage import io
import time


model_path = 'models/convnet_model_48x48_future'
solver = Solver(
    model_dir=model_path,
    labels_file='utility/symbols.json',
    bb_plot=False,
)

image_path = 'test_expressions/pr5.jpg'
image = io.imread(image_path)

start_time = time.time()
latex, result = solver.solve(image)
print('LaTeX:', latex, '\nResult:', result)
print('Time elapsed: {0:.2f} seconds.'.format(time.time() - start_time))

solver.plot_prediction()