from specusticc.configs_init.load_config import load_and_preprocess_config
from specusticc.configs_init.loader_config import LoaderConfig
from specusticc.configs_init.postprocessor_config import PostprocessorConfig
from specusticc.configs_init.preprocessor_config import PreprocessorConfig
from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.configs_init.training_config import TrainingConfig
from specusticc.configs_init.testing_config import TestingConfig
from specusticc.configs_init.reporter_config import ReporterConfig

import specusticc.utilities.directories as dirs


class Configer:
    context_models_list = ['encoder-decoder', 'lstm-attention', 'transformer']

    def __init__(self, config_path: str, model_name: str):
        self.dict_config_from_json = load_and_preprocess_config(config_path, model_name)
        self.class_configs = {}

        self._create_all_class_configs()

    def _create_all_class_configs(self):
        self._create_loader_config()
        self._create_preprocessor_config()
        self._create_model_creator_config()
        self._create_training_config()
        self._create_testing_config()
        self._create_postprocessing_config()
        self._create_reporter_config()

    def _create_loader_config(self):
        loader_config = LoaderConfig()
        loader_config.input_tickers = self.dict_config_from_json['import']['input']['tickers']
        loader_config.output_tickers = self.dict_config_from_json['import']['target']['tickers']
        if self.dict_config_from_json['model']['name'] in self.context_models_list:
            loader_config.context_tickers = self.dict_config_from_json['import']['context']['tickers']
        loader_config.source = 'mongodb'  # TODO to upgrade...someday
        loader_config.database_url = self.dict_config_from_json['import']['database_url']
        self.class_configs['loader'] = loader_config

    def _create_preprocessor_config(self):
        preprocessor_config = PreprocessorConfig()
        preprocessor_config.input_columns = self.dict_config_from_json['import']['input']['columns']
        if 'date' not in preprocessor_config.input_columns:
            preprocessor_config.input_columns.append('date')
        preprocessor_config.output_columns = self.dict_config_from_json['import']['target']['columns']
        if 'date' not in preprocessor_config.output_columns:
            preprocessor_config.output_columns.append('date')

        if 'context' in self.dict_config_from_json['import']:
            preprocessor_config.context_columns = self.dict_config_from_json['import']['context']['columns']
            if 'date' not in preprocessor_config.context_columns:
                preprocessor_config.context_columns.append('date')
            preprocessor_config.context_features = len(
                    self.dict_config_from_json['import']['context']['columns']) - 1  # minus data

        preprocessor_config.train_date = self.dict_config_from_json['import']['train_date']
        preprocessor_config.test_date = self.dict_config_from_json['import']['test_date']
        preprocessor_config.machine_learning_target = self.dict_config_from_json['model']['target']
        preprocessor_config.model_type = self.dict_config_from_json['model']['type']
        preprocessor_config.input_dim = self.dict_config_from_json['model']['input_dim']
        preprocessor_config.seq_length = self.dict_config_from_json['preprocessing']['sequence_length']
        preprocessor_config.seq_prediction_time = self.dict_config_from_json['preprocessing']['sequence_prediction_time']
        preprocessor_config.sample_time_diff = self.dict_config_from_json['preprocessing']['sample_time_difference']
        preprocessor_config.features = len(self.dict_config_from_json['import']['input']['columns']) -1 # minus data
        self.class_configs['preprocessor'] = preprocessor_config

    def _create_model_creator_config(self):
        model_creator_config = ModelCreatorConfig()
        model_creator_config.model_type = self.dict_config_from_json['model']['type']
        model_creator_config.machine_learning_target = self.dict_config_from_json['model']['target']
        if model_creator_config.model_type == 'neural_network':
            model_creator_config.nn_name = self.dict_config_from_json['model']['name']
            model_creator_config.nn_input_timesteps = self.dict_config_from_json['preprocessing']['sequence_length']
            model_creator_config.nn_input_features = len(self.dict_config_from_json['import']['input']['columns']) -1 # minus data
            model_creator_config.nn_output_timesteps = self.dict_config_from_json['preprocessing']['sequence_prediction_time']

            if self.dict_config_from_json['model']['name'] in self.context_models_list:
                model_creator_config.nn_context_timesteps = self.dict_config_from_json['preprocessing']['sequence_length']
                model_creator_config.nn_context_features = len(self.dict_config_from_json['import']['context']['columns']) -1 # minus data
        elif model_creator_config.model_type == 'decision_tree':
            model_creator_config.t_max_depth = self.dict_config_from_json['model']['max_depth']
        else:
            raise NotImplementedError
        self.class_configs['model_creator'] = model_creator_config

    def _create_training_config(self):
        training_config = TrainingConfig()
        # as of 10.04 nothing, still empty class for the sake of consistency
        self.class_configs['training'] = training_config

    def _create_testing_config(self):
        testing_config = TestingConfig()
        testing_config.test_on_learning_base = self.dict_config_from_json['model']['test_on_learning']
        self.class_configs['testing'] = testing_config

    def _create_postprocessing_config(self):
        postprocessing_config = PostprocessorConfig()
        postprocessing_config.test_on_learning_base = self.dict_config_from_json['model']['test_on_learning']
        self.class_configs['postprocessor'] = postprocessing_config

    def _create_reporter_config(self):
        reporter_config = ReporterConfig()
        reporter_config.model_type = self.dict_config_from_json['model']['type']
        reporter_config.machine_learning_target = self.dict_config_from_json['model']['target']
        reporter_config.test_on_learning_base = self.dict_config_from_json['model']['test_on_learning']
        reporter_config.report_directory = 'output/' + dirs.get_timestamp_dir()
        self.class_configs['reporter'] = reporter_config

    def get_class_configs(self) -> dict:
        return self.class_configs
