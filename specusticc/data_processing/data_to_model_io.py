import pandas as pd
import numpy as np


class DataToModelIO:
    def __init__(self, config: dict) -> None:
        self.config = config

        self.seq_len = config['data']['sequence_length']
        self.columns_num = len(self.config['data']['columns']) - 1
        self.input = None
        self.output = None
        self.input_len = None

    def transform_for_model(self, data: pd.DataFrame) -> tuple:
        data_aligned = self._align_to_sequences(data)
        data_aligned = data_aligned.drop(columns=['date'])

        target = self.config['model']['target']
        if target == 'regression':
            self._divide_for_regression(data_aligned)
        elif target == 'classification':
            self._calculate_classes(data_aligned)
            self._reshape_for_classification(data_aligned)
        else:
            raise NotImplementedError
        return self.get_io()

    def _align_to_sequences(self, data: pd.DataFrame) -> pd.DataFrame:
        seq_len = self.config['data']['sequence_length']
        shift = self.config['data']['sequence_shift']

        len_data = len(data)
        self.input_len = int((len_data - seq_len) / shift)
        last_input_index = self.input_len * seq_len
        return data.iloc[:last_input_index]

    def _divide_for_regression(self, data: pd.DataFrame) -> None:
        data_np = data.to_numpy()
        tensor = data_np.reshape(self.input_len, self.seq_len, self.columns_num)

        self.input = tensor[:, :-1]
        self.output = tensor[:, -1, 0]

    def _reshape_for_classification(self, data: pd.DataFrame) -> None:
        # IF model is Decision Tree then leaves data model as Pandas
        # Else reshapes it to 3D numpy array
        model_type = self.config['model']['type']
        if model_type == 'decision_tree':
            self.input = data
            return

        sample_shift = self.config['data']['sequence_shift']
        shifts = 0
        data_np = []
        while shifts < self.input_len:
            data_np.append(data.iloc[shifts * sample_shift: shifts * sample_shift + self.seq_len])
            shifts += 1
        data_np = pd.concat(data_np).to_numpy()
        data_np = data_np.reshape(self.input_len, self.seq_len, self.columns_num)
        self.input = data_np

    def _calculate_classes(self, data: pd.DataFrame) -> None:
        from specusticc.data_processing.ratio_to_class import ratio_to_class
        prediction_shift = self.config['data']['prediction_shift']
        sequence_shift = self.config['data']['sequence_shift']

        output_labels = []
        i = 0
        while i < self.input_len - 1:
            present_val_index = i * sequence_shift + self.seq_len - 1
            future_val_index = present_val_index + prediction_shift

            present_val = data.loc[:, 'close'].iloc[present_val_index]
            future_val = data.loc[:, 'close'].iloc[future_val_index]
            ratio = future_val / present_val

            ratio_class = ratio_to_class(ratio)
            output_labels.append(ratio_class)
            i += 1

        ratio_class = ratio_to_class(1.)
        output_labels.append(ratio_class)
        self.output = np.array(output_labels)

    def get_io(self) -> tuple:
        return self.input, self.output
