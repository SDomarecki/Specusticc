import pydotplus as pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image


class DecisionTreeVisualizer:
    def __init__(self, tree_clf: DecisionTreeClassifier, config: dict) -> None:
        self.tree = tree_clf
        self.feature_names = config['data']['features']
        self.target_names = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']

    def visualize(self) -> None:
        # Create DOT data
        dot_data = tree.export_graphviz(self.tree, out_file=None,
                                        feature_names=self.feature_names,
                                        class_names=self.target_names,
                                        rounded=True, filled=True)

        # Draw graph
        graph = pydotplus.graph_from_dot_data(dot_data)

        # Show graph
        Image(graph.create_png())

        # Create PNG
        graph.write_png("tree.png")
