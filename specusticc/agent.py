def load_config(config_path) -> dict:
    import json
    return json.load(open(config_path))


def create_save_dir(save_dir: str) -> None:
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


class Agent:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.data_proc = None
        self.model = None
        self.predictions = None
        self.report = None
        self.input_train = None
        self.input_test = None
        self.output_train = None
        self.output_test = None

        create_save_dir(self.config['model']['save_dir'])
        self.load_data()
        self.create_model_from_config()
        self.prepare_data_batches()
        self.train_model()
        self.test_model()
        self.print_report()

    def load_data(self):
        from specusticc.data_processing.data_processor import DataProcessor
        self.data_proc = DataProcessor(self.config)

    def create_model_from_config(self):
        from specusticc.neural_networks.neural_network_builder import NeuralNetworkBuilder
        self.model = NeuralNetworkBuilder.build_network(self.config)

    def prepare_data_batches(self):
        self.input_train, self.output_train = self.data_proc.get_train_data()
        self.input_test, self.output_test = self.data_proc.get_test_data()

    def train_model(self):
        epochs = self.config['training']['epochs']
        batch_size = self.config['training']['batch_size']
        save_dir = self.config['model']['save_dir']
        self.model.train(self.input_train, self.output_train, epochs, batch_size, save_dir)

    def test_model(self):
        if self.config['model']['target'] == 'regression':
            seq_len = self.config['data']['sequence_length']
            self.predictions = self.model.predict_sequences_multiple(self.input_test, seq_len, seq_len)
        else:
            self.predictions = self.model.predict_classification(self.input_test)

    def print_report(self):
        target = self.config['model']['target']
        if target == 'regression':
            self.print_regression_report()
        elif target == 'classification':
            self.print_classification_report()
        else:
            raise NotImplementedError

    def print_regression_report(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(self.output_test, label='True Data')
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(self.predictions):
            padding = [None for p in range(i * self.config['data']['sequence_length'])]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        plt.show()

    def print_classification_report(self):
        import matplotlib.pyplot as plt
        import numpy as np
        seq_len = self.config['data']['sequence_length']

        y = self.input_test[:, :, 0].reshape(self.input_test.shape[0]*self.input_test.shape[1])
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(y, label='True Data')
        for i, data in enumerate(self.predictions):
            padding = (i+1) * seq_len
            arg_max = np.argmax(data)
            if arg_max > 2:
                marker = '^'
            elif arg_max == 2:
                marker = '>'
            else:
                marker = 'v'
            plt.plot(padding, 1, label='Prediction', marker=marker)
            plt.legend()
        plt.grid()
        plt.show()