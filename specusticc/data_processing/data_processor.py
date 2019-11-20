class DataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.train = None
        self.test = None

        self._load_data()
        self._preprocess_data()

    def _load_data(self) -> None:
        from specusticc.data_processing.data_loader import DataLoader
        loader = DataLoader(self.config)
        self.loaded_data = loader.load_data()

    def _preprocess_data(self) -> None:
        self._split_data()
        self._transform_data_to_model_input_output()

    def _split_data(self) -> None:
        from sklearn.model_selection import train_test_split
        test_size = self.config['data']['test_size']
        self.train, self.test = train_test_split(self.loaded_data, test_size=test_size, shuffle=False)

    def _transform_data_to_model_input_output(self):
        from specusticc.data_processing.data_to_model_io import DataToModelIO
        data2io = DataToModelIO(self.config)
        self.input_train, self.output_train = data2io.transform_for_model(self.train)
        self.input_test, self.output_test = data2io.transform_for_model(self.test)

    def get_train_data(self) -> tuple:
        return self.input_train, self.output_train

    def get_test_data(self) -> tuple:
        return self.input_test, self.output_test
