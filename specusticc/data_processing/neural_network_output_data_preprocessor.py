from specusticc.configs_init.preprocessor_config import PreprocessorConfig

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from specusticc.data_processing.ratio_to_class import ratio_to_class


class NeuralNetworkOutputDataPreprocessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config
        self.timestamps = config.seq_length
        self.prediction = config.seq_prediction_time
        self.sample_time_diff = config.sample_time_diff
        self.features = config.features

    def transform_output(self, data):
        target = self.config.machine_learning_target
        if target == 'regression':
            return self._shift_for_regression(data)
        elif target == 'classification':
            return self._calculate_classes(data)
        else:
            raise NotImplementedError

    def _calculate_classes(self, data: dict):
        output_labels = []
        for v in data.values():
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

        output = np.array(output_labels)
        return output

    def _shift_for_regression(self, data):
        data = data.drop(columns='date')

        output_values = []
        samples = int((len(data) - self.prediction - self.timestamps) / self.sample_time_diff)
        for i in range(samples):
            start_index = i * self.sample_time_diff + self.timestamps
            end_index = start_index + self.prediction
            output_values.append(data.iloc[start_index: end_index])

        output_values = pd.concat(output_values).to_numpy()

        scaler = MinMaxScaler(feature_range=(0, 1))
        output_values = scaler.fit_transform(output_values)

        output = output_values.reshape(samples, self.prediction)
        return output, scaler

