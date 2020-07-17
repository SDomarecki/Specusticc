import numpy as np


class DataSet:
    def __init__(self):
        self.input = None
        self.context = None

        self.output = None
        self.output_scaler = None
        self.output_columns = None
        self.output_dates = None

    def get_input(self, model_name: str) -> []:
        if model_name not in ['basic', 'mlp', 'cnn', 'lstm', 'gan']:
            return [self.context, self.input]
        else:
            return [np.dstack((self.context, self.input))]

    def get_output(self) -> np.array:
        return self.output
