import numpy as np
import pandas as pd

from specusticc.data_processing.ratio_to_class import ratio_to_class


class NeuralNetworkDataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config

        self.samples = None
        self.timestamps = config['preprocessing']['sequence_length']
        self.prediction = config['preprocessing']['sequence_prediction_time']
        self.sample_time_diff = config['preprocessing']['sample_time_difference']
        self.features = 0

        self.input = None
        self.output = None
        self.input_len = None

        self._count_features()

    def transform_for_model(self, input_data: dict, output_data: dict) -> tuple:
        target = self.config['model']['target']

        self._reshape_input_data(input_data)
        if target == 'regression':
            self._shift_for_regression(output_data)
        elif target == 'classification':
            self._calculate_classes(output_data)
        else:
            raise NotImplementedError
        return self.input, self.output

    def _reshape_input_data(self, data: dict):
        data_np = []
        self.samples = 0
        for k, v in data.items():
            samples = int((len(v) - self.prediction - self.timestamps) / self.sample_time_diff)
            self.samples += samples
            for i in range(samples):
                start_index = i * self.sample_time_diff
                end_index = start_index + self.timestamps
                data_np.append(v.drop(columns='date').iloc[start_index: end_index])

        data_np = pd.concat(data_np).to_numpy()
        first_layer_type = self.config['model']['layers'][0]['type']
        if first_layer_type == 'dense':
            data_np = data_np.reshape(self.samples, self.timestamps * self.features)
        else:
            data_np = data_np.reshape(self.samples, self.timestamps, self.features)
        self.input = data_np

    def _calculate_classes(self, data: dict) -> None:
        output_labels = []
        for k, v in data.items():
            samples = int((len(v) - self.prediction - self.timestamps) / self.sample_time_diff)
            for i in range(samples - 1):
                present_val_index = i * self.sample_time_diff + self.timestamps - 1
                future_val_index = present_val_index + self.prediction

                present_val = v.loc[:, 'close'].iloc[present_val_index]
                future_val = v.loc[:, 'close'].iloc[future_val_index]

                ratio = future_val / present_val

                ratio_class = ratio_to_class(ratio)
                output_labels.append(ratio_class)

            ratio_class = ratio_to_class(1.)
            output_labels.append(ratio_class)

        self.output = np.array(output_labels)

    def _count_features(self):
        columns = self.config['import']['input']['columns']
        self.features += len(columns) - 1

    def _shift_for_regression(self, data: dict):
        output_values = []

        for k, v in data.items():
            samples = int((len(v) - self.prediction - self.timestamps) / self.sample_time_diff)
            for i in range(samples):
                start_index = i * self.sample_time_diff + self.timestamps
                end_index = start_index + self.prediction
                output_values.append(v.drop(columns='date').iloc[start_index: end_index])

        output_values = pd.concat(output_values).to_numpy()
        self.output = output_values.reshape(self.samples, self.prediction)