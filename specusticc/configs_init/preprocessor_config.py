class PreprocessorConfig:
    def __init__(self):
        self.input_columns = []
        self.output_columns = []
        self.context_columns = []
        self.train_date = {}
        self.test_date = {}
        self.input_dim = 0
        self.seq_length = 0
        self.seq_prediction_time = 0
        self.sample_time_diff = 0
        self.features = 0
        self.context_features = 0
        pass
