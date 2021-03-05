from specusticc.data_preprocessing.preprocessed_data import PreprocessedData
from specusticc.model_testing.prediction_results import PredictionResults


class Tester:
    def __init__(self, model, model_name: str, data: PreprocessedData):
        self._model = model
        self._data: PreprocessedData = data
        self._model_name = model_name

        self.prediction_results: PredictionResults = PredictionResults()

    def test(self):
        train_set = self._data.train_set
        input_data = train_set.get_input(self._model_name)
        output_data = train_set.get_output()
        self.prediction_results.train_output = self._model.predict(input_data)

        print("Evaluate on train data")
        self._model.evaluate(input_data, output_data, batch_size=128)

        test_sets = self._data.test_sets
        self.prediction_results.test_output = []
        for test_set in test_sets:
            input_data = test_set.get_input(self._model_name)
            output_data = test_set.get_output()
            prediction = self._model.predict(input_data)
            self.prediction_results.test_output.append(prediction)
            print("Evaluate on test data")
            self._model.evaluate(input_data, output_data, batch_size=128)

    def get_test_results(self) -> PredictionResults:
        return self.prediction_results
