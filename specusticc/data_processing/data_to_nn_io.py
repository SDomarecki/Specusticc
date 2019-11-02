import pandas as pd
import numpy as np

def ratio_to_class(ratio: float) -> []:
    if ratio > 1.2:
        return [0, 0, 0, 0, 1]  # Strong Buy
    elif ratio > 1.05:
        return [0, 0, 0, 1, 0]  # Buy
    elif ratio > 0.95:
        return [0, 0, 1, 0, 0]  # Hold
    elif ratio > 0.8:
        return [0, 1, 0, 0, 0]  # Sell
    else:
        return [1, 0, 0, 0, 0]  # Strong Sell


class DataToNNIO:
    def __init__(self, data: pd.DataFrame, target: str, seq_len: float) -> None:
        self.data = data.loc[:, ['close', 'vol']]
        self.len_data = len(data)
        self.seq_len = seq_len

        self.x_len = int(self.len_data/self.seq_len)

        self.x = None
        self.y = None

        if target == 'regression':
            self.divide_for_regression()
        elif target == 'classification':
            self.divide_for_classification()
        else:
            raise NotImplementedError

    def divide_for_regression(self) -> None:
        last_index = self.x_len*self.seq_len
        data_np = self.data.iloc[:last_index].to_numpy()
        data_np = data_np.reshape(self.x_len, self.seq_len, 2)

        self.x = data_np[:, :-1]
        self.y = data_np[:, -1, 0]
        pass

    def divide_for_classification(self) -> None:
        last_index = self.x_len * self.seq_len
        data_np = self.data.iloc[:last_index].to_numpy()
        data_np = data_np.reshape(self.x_len, self.seq_len, 2)

        self.x = data_np
        self.y = []
        i = 0
        while i < self.x_len:
            if i != self.x_len - 1:
                present_val = data_np[i, -1, 0]
                next_val = data_np[i+1, -1, 0]
                ratio = next_val / present_val
            else:
                ratio = 1.0

            self.y.append(ratio_to_class(ratio))
            i += 1
        self.y = np.array(self.y)
        pass

    def get_io(self) -> tuple:
        return self.x, self.y
