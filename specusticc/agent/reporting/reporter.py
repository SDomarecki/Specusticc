from specusticc.predictive_models.neural_networks.model import Model
import specusticc.utilities.directories as dirs
import json
from specusticc.agent.reporting.plotter import Plotter


class Reporter:
    def __init__(self, config: dict, test: tuple,  predictions, model: Model) -> None:
        self.config = config
        self.input_test, self.output_test = test
        self.predictions = predictions
        self.model = model
        self.report_directory = None

    def print_report(self) -> None:
        self._create_report_directory()
        self._save_model()
        self._save_config()
        self._save_prediction_plot()

    def _create_report_directory(self) -> None:
        self.report_directory = 'output/' + dirs.get_timestamp_dir()
        dirs.create_save_dir(self.report_directory)

    def _save_model(self) -> None:
        save_path = self.report_directory + "/model.h5"
        self.model.save(save_path)

    def _save_config(self) -> None:
        save_path = self.report_directory + '/used_config.json'
        self._restore_string_data()
        with open(save_path, 'w') as outfile:
            json.dump(self.config, outfile, indent=4)

    def _restore_string_data(self):
        fromDT = self.config['import']['train_date']['from']
        self.config['import']['train_date']['from'] = fromDT.strftime('%Y-%m-%d')
        toDT = self.config['import']['train_date']['to']
        self.config['import']['train_date']['to'] = toDT.strftime('%Y-%m-%d')
        fromDT = self.config['import']['test_date']['from']
        self.config['import']['test_date']['from'] = fromDT.strftime('%Y-%m-%d')
        toDT = self.config['import']['test_date']['to']
        self.config['import']['test_date']['to'] = toDT.strftime('%Y-%m-%d')

    def _save_prediction_plot(self) -> None:
        target = self.config['model']['target']
        p = Plotter(self.config, self.input_test)
        if target == 'regression':
            p.draw_regression_plots(self.predictions)
        elif target == 'classification':
            p.draw_classification_plot(self.output_test, self.predictions)
        else:
            raise NotImplementedError
        p.save_prediction_plot(self.report_directory)
