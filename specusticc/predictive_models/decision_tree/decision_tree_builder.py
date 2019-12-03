from sklearn.tree import DecisionTreeClassifier


class DecisionTreeBuilder:
    def __init__(self, config: dict):
        print('[Tree] Tree initialization')
        self.decision_tree = DecisionTreeClassifier(max_depth=config['model']['max_depth'])

    def build(self) -> DecisionTreeClassifier:
        return self.decision_tree
