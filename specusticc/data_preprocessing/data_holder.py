class DataHolder:
    def __init__(self):
        self.train_input = None
        self.train_input_scaler = None
        self.train_input_columns = None
        self.train_input_dates = None

        self.test_input = None
        self.test_input_scaler = None
        self.test_input_columns = None
        self.test_input_dates = None

        self.train_output = None
        self.train_output_scaler = None
        self.train_output_columns = None
        self.train_output_dates = None

        self.test_output = None
        self.test_output_scaler = None
        self.test_output_columns = None
        self.test_output_dates = None

        self.train_context = None
        self.train_context_scaler = None
        self.train_context_columns = None
        self.train_context_dates = None

        self.test_context = None
        self.test_context_scaler = None
        self.test_context_columns = None
        self.test_context_dates = None

    def get_train_input(self):
        if self.train_context is not None:
            return [self.train_context, self.train_input]
        else:
            return self.train_input

    def get_train_output(self):
        return self.train_output

    def get_test_input(self):
        if self.test_context is not None:
            return [self.test_context, self.test_input]
        else:
            return self.test_input

    def get_test_output(self):
        return self.test_output
