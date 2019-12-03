import pandas as pd
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, decision_tree: DecisionTreeClassifier) -> None:
        self.tree = decision_tree

    def save(self, save_dir: str) -> None:
        from joblib import dump
        dump(self.tree, save_dir)

    def train(self, train_input: pd.DataFrame, train_output: pd.DataFrame) -> None:
        print('[Tree] Training Started')
        self.tree.fit(train_input, train_output)

    def predict_classification(self, test_input: pd.DataFrame) -> []:
        print('[Tree] Testing Started')
        predictions = self.tree.predict(test_input)
        return predictions
