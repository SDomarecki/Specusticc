from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.configs_init.model.configs_wrapper import ConfigsWrapper
from specusticc.configs_init.load_config import load_and_preprocess_config
from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.configs_init.model.market_config import MarketConfig
from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig

class Configer:
    def __init__(self, config_path: str, market_save_path: str):
        self.market_save_path = market_save_path
        self.dict_config_from_json = load_and_preprocess_config(config_path, backup_path=self.market_save_path)
        self.configs: ConfigsWrapper = ConfigsWrapper()

        self._create_all_class_configs()

    def _create_all_class_configs(self):
        self._create_loader_config()
        self._create_preprocessor_config()
        self._create_agent_config()
        self._create_market_config()

    def _create_loader_config(self):
        loader_config = LoaderConfig()
        loader_config.input_tickers = self.dict_config_from_json['import']['input']['tickers']
        loader_config.output_tickers = self.dict_config_from_json['import']['target']['tickers']
        loader_config.context_tickers = self.dict_config_from_json['import']['context']['tickers']
        loader_config.source = 'mongodb'  # TODO to upgrade...someday
        loader_config.database_url = self.dict_config_from_json['import']['database_url']
        self.configs.loader = loader_config

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
        preprocessor_config.seq_length = self.dict_config_from_json['preprocessing']['sequence_length']
        preprocessor_config.seq_prediction_time = self.dict_config_from_json['preprocessing']['sequence_prediction_time']
        preprocessor_config.sample_time_diff = self.dict_config_from_json['preprocessing']['sample_time_difference']
        preprocessor_config.features = len(self.dict_config_from_json['import']['input']['columns']) -1 # minus data
        self.configs.preprocessor = preprocessor_config

    def _create_agent_config(self):
        agent_config = AgentConfig()
        agent_config.input_timesteps = self.dict_config_from_json['preprocessing']['sequence_length']
        features = (len(self.dict_config_from_json['import']['input']['columns']) - 1) * len(
                self.dict_config_from_json['import']['input']['tickers'])
        agent_config.input_features = features
        agent_config.output_timesteps = self.dict_config_from_json['preprocessing']['sequence_prediction_time']

        agent_config.context_timesteps = self.dict_config_from_json['preprocessing']['sequence_length']
        features = (len(self.dict_config_from_json['import']['context']['columns']) - 1) * len(
                self.dict_config_from_json['import']['context']['tickers'])
        agent_config.context_features = features

        agent_config.hyperparam_optimization_method = self.dict_config_from_json['agent']['hyperparam_optimization_method']
        agent_config.horizon_prediction_method = self.dict_config_from_json['agent']['horizon_prediction_method']
        agent_config.market_save_path = self.market_save_path
        self.configs.agent = agent_config

    def _create_market_config(self):
        market_config = MarketConfig()
        market_config.n_folds = self.dict_config_from_json['agent']['folds']
        self.configs.market = market_config

    def get_class_configs(self) -> ConfigsWrapper:
        return self.configs
