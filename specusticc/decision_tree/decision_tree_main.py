import datetime

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

import models.indicators as ind


class DecisionTree:
    def __init__(self, timedelta: int) -> None:
        self.timedelta = timedelta
        self.now = datetime.datetime.utcnow()
        self.tree_clf = DecisionTreeClassifier(max_depth=4)
        self.history = None
        self.target_labels = None
        self.feature_names = ['roc', 'macd_val', 'rsi']
        self.target_names = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']

    def normal_use(self):
        self.boost_AT_info()

        self.train_model()
        self.visualize()
        self.convert_dot_to_png()

    def boost_AT_info(self):
        self.history = ind.macd(self.history, close_col='close')
        self.history = ind.rsi(self.history, close_col='close')
        self.history = ind.roc(self.history, close_col='close')


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
