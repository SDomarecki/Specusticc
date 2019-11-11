import pandas as pd
import numpy as np


class DataToModelIO:
    def __init__(self, data: pd.DataFrame, config: dict) -> None:
        columns = config['data']['columns']
        self.data = data.loc[:, columns]
        len_data = len(data)
        self.seq_len = config['data']['sequence_length']
        self.model_type = config['model']['type']
        self.x_len = int(len_data/self.seq_len)

        last_index = self.x_len * self.seq_len
        self.data_aligned_to_sequences = self.data.iloc[:last_index]

        self.x = None
        self.y = None
        self.divide_for_model(config['model']['target'])

    def divide_for_model(self, target: str) -> None:
        if target == 'regression':
            self.divide_for_regression()
        elif target == 'classification':
            self.reshape_for_classification()
            self.calculate_classes()
        else:
            raise NotImplementedError

    def divide_for_regression(self) -> None:
        data_np = self.data_aligned_to_sequences.to_numpy()
        data_np = data_np.reshape(self.x_len, self.seq_len, 2)

        self.x = data_np[:, :-1]
        self.y = data_np[:, -1, 0]

    def reshape_for_classification(self) -> None:
        # IF model is Decision Tree then leaves data model as Pandas
        # Else reshapes it to 3D numpy array
        if self.model_type == 'decision_tree':
            self.x = self.data_aligned_to_sequences
            return
        data_np = self.data_aligned_to_sequences.to_numpy()
        data_np = data_np.reshape(self.x_len, self.seq_len, 2)
        self.x = data_np

    def calculate_classes(self) -> None:
        from specusticc.data_processing.ratio_to_class import ratio_to_class
        self.y = []
        i = 0
        while i < self.x_len - 1:
            if self.model_type == 'decision_tree':
                present_val = self.x.loc[:, 'close'].iloc[i]
                next_val = self.x.loc[:, 'close'].iloc[i+1]
            else:
                present_val = self.x[i, -1, 0]
                next_val = self.x[i+1, -1, 0]
            ratio = next_val / present_val

            ratio_class = ratio_to_class(ratio, self.model_type)
            self.y.append(ratio_class)

            i += 1

        ratio_class = ratio_to_class(1, self.model_type)
        self.y.append(ratio_class)
        self.y = np.array(self.y)

    def get_io(self) -> tuple:
        return self.x, self.y
