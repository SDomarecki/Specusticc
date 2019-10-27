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
        self.x = None
        self.y = None
        self.x_test = None
        self.y_test = None

        create_save_dir(self.config['model']['save_dir'])
        self.load_data()
        self.create_model_from_config()
        self.train_model()
        self.test_model()
        self.print_report()

    def load_data(self):
        from specusticc.data_processor import DataProcessor
        self.data_proc = DataProcessor(self.config)

    def create_model_from_config(self):
        from specusticc.neural_networks.neural_network_builder import NeuralNetworkBuilder
        self.model = NeuralNetworkBuilder.build_network(self.config)

    def train_model(self):
        self.x, self.y = self.data_proc.get_train_data(
                seq_len=self.config['data']['sequence_length'],
                normalise=self.config['data']['normalise']
        )

        self.x_test, self.y_test = self.data_proc.get_test_data(
                seq_len=self.config['data']['sequence_length'],
                normalise=self.config['data']['normalise']
        )

        self.model.train(self.x, self.y,
                         epochs=self.config['training']['epochs'],
                         batch_size=self.config['training']['batch_size'],
                         save_dir=self.config['model']['save_dir'])

    def test_model(self):
        self.predictions = self.model.predict_sequences_multiple(
                self.x_test,
                self.config['data']['sequence_length'],
                self.config['data']['sequence_length'])

    def print_report(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        ax.plot(self.y_test, label='True Data')
        # Pad the list of predictions to shift it in the graph to it's correct start
        for i, data in enumerate(self.predictions):
            padding = [None for p in range(i * self.config['data']['sequence_length'])]
            plt.plot(padding + data, label='Prediction')
            plt.legend()
        plt.show()
