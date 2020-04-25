# TODO Data preprocessing - move to dataProc

# x = boost_AT_info(train_input)
# key_history = x[self.start_sample::self.timedelta]
# key_history = key_history[:-1]
# X = key_history[self.feature_names]

import pandas as pd

from specusticc.configs_init.preprocessor_config import PreprocessorConfig


def boost_AT_info(source: pd.DataFrame) -> pd.DataFrame:
    source = source.reset_index(drop=True)

    import specusticc.model_creating.decision_tree.indicators as ind
    print('[Tree] Boosting techincal analysis info')
    history = ind.macd(source, close_col='close')
    history = ind.rsi(history, close_col='close')
    history = ind.roc(history, close_col='close')
    return history


class TreeDataProcessor:
    def __init__(self, config: PreprocessorConfig):
        self.config = config

    def transform_input(self):
        pass

    def transform_output(self):
        pass