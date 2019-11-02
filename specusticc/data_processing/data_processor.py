import numpy as np


class DataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.ticker = config['import']['ticker']
        self.test_size = config['data']['test_size']
        self.seq_len = config['data']['sequence_length']
        self.target = config['model']['target']

        from specusticc.data_processing.data_loader import DataLoader
        loader = DataLoader(config)
        self.dataframe = loader.load_data()

        from specusticc.data_processing.data_to_nn_io import DataToNNIO
        data2io = DataToNNIO(self.dataframe, self.target, self.seq_len)
        self.input, self.output = data2io.get_io()

        from specusticc.data_processing.data_splitter import DataSplitter
        splitter = DataSplitter(self.input, self.output, self.test_size, self.target)
        self.input_train, self.output_train = splitter.get_train()
        self.input_test, self.output_test = splitter.get_test()

    def get_train_data(self) -> tuple:
        return self.input_train, self.output_train

    def get_test_data(self) -> tuple:
        return self.input_test, self.output_test

    def _normalise_windows(self, window_data: []) -> np.array:
        """Normalise window with a base value of zero"""
        normalised_data = []
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            # reshape and transpose array back into original multidimensional format
            normalised_window = np.array(normalised_window).T

            normalised_data.append(normalised_window)
        return np.array(normalised_data)
