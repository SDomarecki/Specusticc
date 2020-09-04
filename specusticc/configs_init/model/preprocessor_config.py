from datetime import datetime

class PreprocessorConfig:
    def __init__(self):
        self.input_columns: [] = []
        self.output_columns: [] = []
        self.context_columns: [] = []

        self.train_date: DateRange = None
        self.test_dates: [DateRange] = None

        self.window_length: int = 0
        self.rolling: int = 0
        self.horizon: int = 0
        self.features: int = 0
        self.context_features: int = 0


class DateRange:
    def __init__(self, from_date: datetime, to_date: datetime):
        self.from_date: datetime = from_date
        self.to_date: datetime = to_date
