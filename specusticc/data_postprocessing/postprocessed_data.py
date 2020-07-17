import pandas as pd


class PostprocessedData:
    def __init__(self):
        self.train_true_data: pd.DataFrame = None
        self.train_prediction: pd.DataFrame = None
        self.test_true_datas: [pd.DataFrame] = []
        self.test_predictions: [pd.DataFrame] = []
