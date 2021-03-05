from specusticc.data_preprocessing.data_set import DataSet


class PreprocessedData:
    def __init__(self):
        self.train_set: DataSet = None
        self.test_sets: [DataSet] = []
