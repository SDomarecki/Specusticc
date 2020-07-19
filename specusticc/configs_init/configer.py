from specusticc.configs_init.load_config import load_and_preprocess_config
from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.configs_init.model.configs_wrapper import ConfigsWrapper
from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.configs_init.model.market_config import MarketConfig
from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig, DateRange


class Configer:
    def __init__(self, config_path: str, market_save_path: str):
        self.market_save_path = market_save_path
        self._dict_config = load_and_preprocess_config(config_path, backup_path=self.market_save_path)
        self._configs: ConfigsWrapper = ConfigsWrapper()

        self._create_all_configs()

    def get_configs_wrapper(self) -> ConfigsWrapper:
        return self._configs

    def _create_all_configs(self):
        self._create_loader_config()
        self._create_preprocessor_config()
        self._create_agent_config()
        self._create_market_config()

    def _create_loader_config(self):
        loader_config = LoaderConfig()
        loader_config.input_tickers = self._dict_config['import']['input']['tickers']
        loader_config.output_tickers = self._dict_config['import']['target']['tickers']
        loader_config.context_tickers = self._dict_config['import']['context']['tickers']
        self._configs.loader = loader_config

    def _create_preprocessor_config(self):
        preprocessor_config = PreprocessorConfig()
        preprocessor_config.input_columns = self._dict_config['import']['input']['columns']
        if 'date' not in preprocessor_config.input_columns:
            preprocessor_config.input_columns.append('date')
        preprocessor_config.output_columns = self._dict_config['import']['target']['columns']
        if 'date' not in preprocessor_config.output_columns:
            preprocessor_config.output_columns.append('date')
        preprocessor_config.context_columns = self._dict_config['import']['context']['columns']
        if 'date' not in preprocessor_config.context_columns:
            preprocessor_config.context_columns.append('date')

        preprocessor_config.context_features = len(
                self._dict_config['import']['context']['columns']) - 1  # minus data

        train_date_range = DateRange(self._dict_config['import']['train_date']['from'],
                                     self._dict_config['import']['train_date']['to'])
        preprocessor_config.train_date = train_date_range
        test_date_ranges = []
        for test_date in self._dict_config['import']['test_date']:
            date_range = DateRange(test_date['from'], test_date['to'])
            test_date_ranges.append(date_range)
        preprocessor_config.test_dates = test_date_ranges

        preprocessor_config.seq_length = self._dict_config['preprocessing']['sequence_length']
        preprocessor_config.seq_prediction_time = self._dict_config['preprocessing']['sequence_prediction_time']
        preprocessor_config.sample_time_diff = self._dict_config['preprocessing']['sample_time_difference']
        preprocessor_config.features = len(self._dict_config['import']['input']['columns']) -1 # minus data
        self._configs.preprocessor = preprocessor_config

    def _create_agent_config(self):
        agent_config = AgentConfig()
        agent_config.input_timesteps = self._dict_config['preprocessing']['sequence_length']
        features = (len(self._dict_config['import']['input']['columns']) - 1) * len(
                self._dict_config['import']['input']['tickers'])
        agent_config.input_features = features
        agent_config.output_timesteps = self._dict_config['preprocessing']['sequence_prediction_time']

        agent_config.context_timesteps = self._dict_config['preprocessing']['sequence_length']
        features = (len(self._dict_config['import']['context']['columns']) - 1) * len(
                self._dict_config['import']['context']['tickers'])
        agent_config.context_features = features

        agent_config.hyperparam_optimization_method = self._dict_config['agent']['hyperparam_optimization_method']
        agent_config.horizon_prediction_method = self._dict_config['agent']['horizon_prediction_method']
        agent_config.market_save_path = self.market_save_path
        self._configs.agent = agent_config

    def _create_market_config(self):
        market_config = MarketConfig()
        market_config.n_folds = self._dict_config['agent']['folds']
        self._configs.market = market_config
