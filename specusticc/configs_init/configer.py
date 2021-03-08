from specusticc.configs_init.config_loader import ConfigLoader
from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.configs_init.model.configs_wrapper import ConfigsWrapper
from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.configs_init.model.market_config import MarketConfig
from specusticc.configs_init.model.preprocessor_config import (
    PreprocessorConfig,
    DateRange,
)


class Configer:
    def __init__(self, save_path: str):
        self.__save_path = save_path
        self.__dict_config = {}
        self.__configs: ConfigsWrapper = ConfigsWrapper()

    def get_configs_wrapper(self) -> ConfigsWrapper:
        return self.__configs

    def create_all_configs(self):
        self.__fetch_dict_config()
        self.__create_loader_config()
        self.__create_preprocessor_config()
        self.__create_agent_config()
        self.__create_market_config()

    def __fetch_dict_config(self):
        config_path = f"{self.__save_path}/config.json"

        loader = ConfigLoader()
        loader.load_and_preprocess_config(config_path)
        self.__dict_config = loader.get_config()

    def __create_loader_config(self):
        loader_config = LoaderConfig()

        import_dict_config = self.__dict_config.get("import") or {}
        input_dict_config = import_dict_config.get("input") or {}
        target_dict_config = import_dict_config.get("target") or {}
        context_dict_config = import_dict_config.get("context") or {}

        loader_config.datasource = import_dict_config.get("datasource", "csv")
        loader_config.database_path = import_dict_config.get("database_path", "")
        loader_config.input_tickers = input_dict_config.get("tickers", [])
        loader_config.output_tickers = target_dict_config.get("tickers", [])
        loader_config.context_tickers = context_dict_config.get("tickers", [])
        self.__configs.loader = loader_config

    def __create_preprocessor_config(self):
        preprocessor_config = PreprocessorConfig()

        import_dict_config = self.__dict_config.get("import") or {}
        input_dict_config = import_dict_config.get("input") or {}
        target_dict_config = import_dict_config.get("target") or {}
        context_dict_config = import_dict_config.get("context") or {}
        train_date_dict = import_dict_config.get("train_date") or {}
        test_dates = import_dict_config.get("test_date", [])
        preprocessing_dict_config = self.__dict_config.get("preprocessing") or {}

        input_columns = input_dict_config.get("columns", [])
        if "date" not in input_columns:
            input_columns.append("date")
        preprocessor_config.input_columns = input_columns

        target_columns = target_dict_config.get("columns", [])
        if "date" not in target_columns:
            target_columns.append("date")
        preprocessor_config.output_columns = target_columns

        context_columns = context_dict_config.get("columns", [])
        if "date" not in context_columns:
            context_columns.append("date")
        preprocessor_config.context_columns = context_columns

        preprocessor_config.context_features = len(context_columns) - 1  # minus date

        train_date_range = DateRange(
            train_date_dict.get("from"), train_date_dict.get("to")
        )
        preprocessor_config.train_date = train_date_range
        test_date_ranges = []
        for test_date in test_dates:
            date_range = DateRange(test_date.get("from"), test_date.get("to"))
            test_date_ranges.append(date_range)
        preprocessor_config.test_dates = test_date_ranges

        preprocessor_config.window_length = preprocessing_dict_config.get(
            "window_length", 1
        )
        preprocessor_config.horizon = preprocessing_dict_config.get("horizon", 1)
        preprocessor_config.rolling = preprocessing_dict_config.get("rolling", 1)
        preprocessor_config.features = len(input_columns) - 1  # minus date
        self.__configs.preprocessor = preprocessor_config

    def __create_agent_config(self):
        agent_config = AgentConfig()

        import_dict_config = self.__dict_config.get("import") or {}
        input_dict_config = import_dict_config.get("input") or {}
        context_dict_config = import_dict_config.get("context") or {}
        preprocessing_dict_config = self.__dict_config.get("preprocessing") or {}
        agent_dict_config = self.__dict_config.get("agent") or {}

        agent_config.input_timesteps = preprocessing_dict_config.get("window_length", 1)

        features = (len(input_dict_config.get("columns", 1)) - 1) * len(
            input_dict_config.get("tickers", 0)
        )
        agent_config.input_features = features
        agent_config.output_timesteps = preprocessing_dict_config.get("horizon", 1)
        agent_config.context_timesteps = preprocessing_dict_config.get(
            "window_length", 1
        )

        features = (len(context_dict_config.get("columns", 1)) - 1) * len(
            context_dict_config.get("tickers", 0)
        )
        agent_config.context_features = features

        agent_config.hyperparam_optimization_method = agent_dict_config.get(
            "hyperparam_optimization_method", "none"
        )
        agent_config.save_path = self.__save_path
        self.__configs.agent = agent_config

    def __create_market_config(self):
        market_config = MarketConfig()

        agent_dict_config = self.__dict_config.get("agent") or {}

        market_config.n_folds = agent_dict_config.get("folds", 1)
        self.__configs.market = market_config
