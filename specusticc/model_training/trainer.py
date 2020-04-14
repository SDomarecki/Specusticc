from specusticc.configs_init.training_config import TrainingConfig
from specusticc.data_processing.data_holder import DataHolder


class Trainer:
    def __init__(self, model, data: DataHolder, config: TrainingConfig):
        self.model = model
        self.data = data
        self.config = config # atm (10.04) empty class

    def train(self):
        self.model.train(self.data)

    def get_model(self):
        return self.model
