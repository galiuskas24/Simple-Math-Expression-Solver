from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from classes import network as net

import numpy as np
import tensorflow as tf

# -----GLOBAL CONSTANTS------
net.NUM_OF_LABELS = 22
BATCH_SIZE = 100
STEPS = 20000
OUTPUT_MODEL_PATH = 'models/abc'

# Load training and eval data
train_data = np.load('utility/train_images.npy')
train_labels = np.load('utility/train_labels.npy')
eval_data = np.load('utility/test_images.npy')
eval_labels = np.load('utility/test_labels.npy')
# ---------------------------


def main(unused_argv):
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=net.cnn_model_fn, model_dir=OUTPUT_MODEL_PATH)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=STEPS,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
