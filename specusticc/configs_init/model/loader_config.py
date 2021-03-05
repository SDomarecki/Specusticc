class LoaderConfig:
    def __init__(self):
        self.datasource: str = ""  # 'mongodb' or 'csv'
        self.input_tickers: [] = []
        self.output_tickers: [] = []
        self.context_tickers: [] = []
