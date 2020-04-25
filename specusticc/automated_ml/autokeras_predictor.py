import autokeras as ak

from specusticc.data_preprocessing.data_holder import DataHolder


class AutokerasPredictor:
    def __init__(self, data: DataHolder):
        self.data = data
        self.reg = ak.StructuredDataRegressor()
        self.test_on_learn = True
        self.test_results = {}

    def fit_predict(self):
        self._fit()
        self._predict()

    def _fit(self):
        self.reg.fit(x=self.data.train_input, y=self.data.train_output)

    def _predict(self):
        self.test_results['test'] = self.reg.predict(x=self.data.test_input)
        if self.test_on_learn:
            self.test_results['learn'] = self.reg.predict(x=self.data.train_input)

    def get_test_results(self) -> dict:
        return self.test_results
