from specusticc.configs_init.testing_config import TestingConfig
from specusticc.data_preprocessing.data_holder import DataHolder


class Tester:
    def __init__(self, model, data: DataHolder, config: TestingConfig):
        self.model = model
        self.data = data
        self.config = config

        self.test_results = {}

    def test(self):
        test_input = self.data.get_test_input()
        self.test_results['test'] = self.model.predict(test_input)
        if self.config.test_on_learning_base:
            train_input = self.data.get_train_input()
            self.test_results['learn'] = self.model.predict(train_input)

    def get_test_results(self) -> dict:
        return self.test_results
