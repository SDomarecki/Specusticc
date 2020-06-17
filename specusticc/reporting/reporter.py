from specusticc.configs_init.reporter_config import ReporterConfig


class Reporter:
    def __init__(self, configs: dict, test_results: dict, model, rep_config: ReporterConfig):
        self.all_configs = configs
        self.test_results = test_results
        self.config = rep_config
        self.model = model

    def print_report(self):
        self._save_model()
        self._save_results()

    def _save_model(self):
        save_path = self.config.report_directory + "/model.h5"
        self.model.save(save_path)

    def _save_results(self):
        self._save_one_data_csv(self.test_results['learn']['true_data'], 'true_learn_data.csv')
        self._save_one_data_csv(self.test_results['learn']['prediction'], 'prediction_learn_data.csv')
        self._save_one_data_csv(self.test_results['test']['true_data'], 'true_test_data.csv')
        self._save_one_data_csv(self.test_results['test']['prediction'], 'prediction_test_data.csv')

    def _save_one_data_csv(self, data, path):
        full_path = self.config.report_directory + '/' + path
        data.to_csv(full_path, index=False, header=True)
