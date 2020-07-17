import numpy as np


class PredictionResults:
    def __init__(self):
        self.train_output: np.array = None
        self.test_output: [np.array] = None
