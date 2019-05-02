import tensorflow as tf
import numpy as np
from network import cnn_model_fn


class Solver:
    def __init__(self, model_dir):
        # Load model
        self.__classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=model_dir
        )


    def solve(self, image):
        # ----------Lexical part-------------
        #Obrada slike
        symbols, s_bb = 1, 2

        # Create input for prediction
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.array(symbols)},
            shuffle=False
        )
        # Prediction
        predictions = self.__classifier.predict(input_fn=eval_input_fn)


        # ---------- Syntax part ---------------
        # ---------- Semantic part --------------



        return "latex", "rezz"

    def __pri(self):
        pass
