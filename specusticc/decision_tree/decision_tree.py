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
        print('[Tree] Tree initialization')
        self.start_sample = config['data']['sequence_length']
        self.timedelta = config['data']['sequence_shift']
        self.tree_clf = DecisionTreeClassifier(max_depth=config['model']['max_depth'])
        self.feature_names = config['data']['features']

    def save(self, save_dir: str) -> None:
        from joblib import dump
        dump(self.tree_clf, save_dir)

    def train(self, train_input: pd.DataFrame, train_output: pd.DataFrame) -> None:
        x = boost_AT_info(train_input)
        print('[Tree] Training Started')

        key_history = x[self.start_sample::self.timedelta]
        key_history = key_history[:-1]
        X = key_history[self.feature_names]

        self.tree_clf.fit(X, train_output)

    def predict_classification(self, test_data: pd.DataFrame) -> []:
        print('[Tree] Testing Started')
        test_data = boost_AT_info(test_data)

        key_history = test_data[self.start_sample::self.timedelta]
        X = key_history[self.feature_names]

        y_predict = self.tree_clf.predict(X)
        return y_predict
