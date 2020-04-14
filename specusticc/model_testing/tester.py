from specusticc.configs_init.testing_config import TestingConfig
from specusticc.data_processing.data_holder import DataHolder


class Tester:
    def __init__(self, model, data: DataHolder, config: TestingConfig):
        self.model = model
        self.data = data
        self.config = config

        self.test_results = {}

    def test(self):
        self.test_results['test'] = self.model.predict(self.data.test_input)
        if self.config.test_on_learning_base:
            self.test_results['learn'] = self.model.predict(self.data.train_input)

    def get_test_results(self) -> dict:
        return self.test_results
