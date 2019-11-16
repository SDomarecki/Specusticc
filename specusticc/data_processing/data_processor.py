class DataProcessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.ticker = config['import']['ticker']
        self.test_size = config['data']['test_size']
        self.seq_len = config['data']['sequence_length']
        self.target = config['model']['target']
        self.model_type = config['model']['type']

        from specusticc.data_processing.data_loader import DataLoader
        loader = DataLoader(config)
        self.dataframe = loader.load_data()

        from sklearn.model_selection import train_test_split
        self.org_train, self.org_test = train_test_split(self.dataframe, test_size=self.test_size, shuffle=False)

        from specusticc.data_processing.data_to_model_io import DataToModelIO
        data2io = DataToModelIO(self.dataframe, self.config)
        self.input, self.output = data2io.get_io()

        from specusticc.data_processing.data_splitter import DataSplitter
        splitter = DataSplitter(self.input, self.output, self.test_size)
        self.input_train, self.output_train = splitter.get_train()
        self.input_test, self.output_test = splitter.get_test()

    def get_train_data(self) -> tuple:
        return self.input_train, self.output_train

    def get_test_data(self) -> tuple:
        return self.input_test, self.output_test

    def get_org_data(self) -> tuple:
        return self.org_train, self.org_test
