from specusticc.data_preprocessing.data_holder import DataHolder
from specusticc.model_testing.prediction_results import PredictionResults


class Tester:
    def __init__(self, model, model_name: str, data: DataHolder):
        self.model = model
        self.data: DataHolder = data
        self.model_name = model_name

        self.prediction_results: PredictionResults = PredictionResults()

    def test(self):
        train_input = self.data.get_train_input(self.model_name)
        self.prediction_results.train_output = self.model.predict(train_input)

        test_input = self.data.get_test_input(self.model_name)
        self.prediction_results.test_output = self.model.predict(test_input)

    def get_test_results(self) -> PredictionResults:
        return self.prediction_results
