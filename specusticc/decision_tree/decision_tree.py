import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def boost_AT_info(source: pd.DataFrame) -> pd.DataFrame:
    source = source.reset_index(drop=True)

    import specusticc.data_processing.indicators as ind
    print('[Tree] Boosting techincal analysis info')
    history = ind.macd(source, close_col='close')
    history = ind.rsi(history, close_col='close')
    history = ind.roc(history, close_col='close')
    return history


class DecisionTree:
    def __init__(self, config: dict) -> None:
        self.timedelta = config['data']['sequence_length']
        self.tree_clf = DecisionTreeClassifier(max_depth=config['model']['max_depth'])
        self.history = None
        self.target_labels = None
        self.feature_names = ['roc', 'macd_val', 'rsi']

    def train(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        x = boost_AT_info(x)
        print('[Tree] Training Started')

        key_history = x[30::self.timedelta]
        X = key_history[self.feature_names]

        self.tree_clf.fit(X, y)

    def predict_classification(self, test_data: pd.DataFrame) -> []:
        print('[Tree] Testing Started')
        test_data = boost_AT_info(test_data)

        key_history = test_data[30::self.timedelta]
        X = key_history[self.feature_names]

        y_predict = self.tree_clf.predict(X)
        return y_predict
