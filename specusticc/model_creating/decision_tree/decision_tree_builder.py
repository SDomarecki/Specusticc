from sklearn.tree import DecisionTreeClassifier

from specusticc.configs_init.model_creator_config import ModelCreatorConfig


class DecisionTreeBuilder:
    def __init__(self, config: ModelCreatorConfig):
        self.config = config

    def build(self) -> DecisionTreeClassifier:
        print('[Tree] Tree initialization')
        return DecisionTreeClassifier(max_depth=self.config.t_max_depth)
