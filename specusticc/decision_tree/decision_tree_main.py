import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pymongo
import pandas as pd
import models.indicators as ind


class DecisionTree:
    def __init__(self, ticker: str, timedelta: int) -> None:
        self.ticker = ticker
        self.timedelta = timedelta
        self.now = datetime.datetime.utcnow()
        self.tree_clf = DecisionTreeClassifier(max_depth=4)
        self.history = None
        self.target_labels = None
        self.feature_names = ['roc', 'macd_val', 'rsi']
        self.target_names = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']

    def normal_use(self):
        self.load_from_database()
        self.boost_AT_info()
        self.calculate_forward_position_class()
        self.split_data_to_train_and_test()
        self.train_model()
        self.visualize()
        self.convert_dot_to_png()

    def load_from_database(self):
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["stocks"]
        prices = db['prices']
        history = prices.find_one({"ticker": self.ticker})['history']
        self.history = pd.DataFrame(history)[['date', 'open', 'high', 'low', 'close']]
        print(self.history)

    def boost_AT_info(self):
        self.history = ind.macd(self.history, close_col='close')
        self.history = ind.rsi(self.history, close_col='close')
        self.history = ind.roc(self.history, close_col='close')

    def calculate_forward_position_class(self):
        td = self.timedelta
        key_history_frame = self.history[30::td]
        target_labels = []
        for i in range(len(key_history_frame)-1):
            ratio = key_history_frame.iloc[i+1]['close'] / key_history_frame.iloc[i]['close']
            target_labels.append(self.ratio_to_label(ratio))
        target_labels.append('Hold')
        self.target_labels = target_labels

    def ratio_to_label(self, ratio: float) -> str:
        if ratio > 1.20:
            return 'Strong Buy'
        elif ratio > 1.05:
            return 'Buy'
        elif ratio > 0.95:
            return 'Hold'
        elif ratio > 0.8:
            return 'Sell'
        else:
            return 'Strong Sell'

    def split_data_to_train_and_test(self):
        pass

    def train_model(self):
        key_history = self.history[30::self.timedelta]
        X = key_history[self.feature_names]
        y = self.target_labels

        self.tree_clf.fit(X, y)

    def test_model(self):
        pass

    def visualize(self):
        export_graphviz(
            self.tree_clf,
            out_file="drzewo.dot",
            feature_names=self.feature_names,
            class_names=self.target_names,
            rounded=True,
            filled=True
        )

    def convert_dot_to_png(self):
        from subprocess import check_call
        check_call(['dot', '-Tpng', 'drzewo.dot', '-o', 'drzewo.png'])
