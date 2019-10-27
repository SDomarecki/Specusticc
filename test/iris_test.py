from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

class IrisTest:
    def __init__(self):
        self.tree_clf = DecisionTreeClassifier(max_depth=3)

    def normal_route(self):
        self.init_iris()
        self.visualize()
        self.convert_dot_to_png()

    def init_iris(self):
        self.iris = load_iris()
        X = self.iris.data[:, 2:] #długość i szerokość płatka
        y = self.iris.target

        self.tree_clf.fit(X, y)

    def visualize(self):
        export_graphviz(
                self.tree_clf,
                out_file="drzewo.dot",
                feature_names=self.iris.feature_names[2:],
                class_names=self.iris.target_names,
                rounded=True,
                filled=True
        )

