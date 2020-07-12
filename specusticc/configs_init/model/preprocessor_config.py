class PreprocessorConfig:
    def __init__(self):
        self.input_columns: [] = []
        self.output_columns: [] = []
        self.context_columns: [] = []

        self.train_date: {} = {}
        self.test_date: {} = {}

        self.seq_length: int = 0
        self.seq_prediction_time: int = 0
        self.sample_time_diff: int = 0
        self.features: int = 0
        self.context_features: int = 0
