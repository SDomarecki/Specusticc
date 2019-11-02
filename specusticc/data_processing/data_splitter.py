import numpy as np


class DataSplitter:
    def __init__(self, input: np.array, output: np.array, test_size: float, target: str) -> None:
        self.input = input
        self.output = output
        self.test_size = test_size
        self.target = target

        from sklearn.model_selection import train_test_split
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(
                input, output, test_size=test_size, shuffle=False
        )

    def get_train(self) -> tuple:
        return self.input_train, self.output_train

    def get_test(self) -> tuple:
        return self.input_test, self.output_test
