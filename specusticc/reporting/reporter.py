import specusticc.utilities.directories as dirs
from specusticc.configs_init.reporter_config import ReporterConfig
from specusticc.reporting.regression_plotter import RegressionPlotter


class Reporter:
    def __init__(self, configs: dict, test_results: dict, model, rep_config: ReporterConfig) -> None:
        self.all_configs = configs
        self.test_results = test_results
        self.config = rep_config
        self.model = model

    def print_report(self):
        self._create_report_directory()
        self._save_model()
        self._save_prediction_plot()

    def _create_report_directory(self):
        dirs.create_save_dir(self.config.report_directory)

    def _save_model(self):
        save_path = self.config.report_directory + "/model.h5"
        self.model.save(save_path)

    def _save_prediction_plot(self):
        plotter = RegressionPlotter(self.config)

        plotter.draw_prediction_plot(self.test_results)
        plotter.save_prediction_plot('blank')
