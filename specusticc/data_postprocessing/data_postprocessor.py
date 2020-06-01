from specusticc.configs_init.postprocessor_config import PostprocessorConfig
from specusticc.data_preprocessing.data_holder import DataHolder


class DataPostprocessor:
    def __init__(self, processed_data: DataHolder, test_results: dict, config: PostprocessorConfig):
        self.config = config
        self.processed_data = processed_data
        self.test_results = test_results
        self.postprocessed_data = {}

    def get_data(self):
        self._postprocess()
        return self.postprocessed_data

    def _postprocess(self):
        test_scaler = self.processed_data.test_output_scaler
        test_true_output_2D = test_scaler.inverse_transform(self.processed_data.test_output)
        test_predicted_output_2D = test_scaler.inverse_transform(self.test_results['test'])
        self.postprocessed_data['test'] = {}
        self.postprocessed_data['test']['true_data'] = test_true_output_2D.flatten()
        self.postprocessed_data['test']['prediction'] = test_predicted_output_2D.flatten()
        if self.config.test_on_learning_base:
            learn_scaler = self.processed_data.train_output_scaler
            learn_true_output_2D = learn_scaler.inverse_transform(self.processed_data.train_output)
            learn_predicted_output_2D = learn_scaler.inverse_transform(self.test_results['learn'])
            self.postprocessed_data['learn'] = {}
            self.postprocessed_data['learn']['true_data'] = learn_true_output_2D.flatten()
            self.postprocessed_data['learn']['prediction'] = learn_predicted_output_2D.flatten()

