from datetime import datetime

from specusticc.configs_init.configer import Configer
from specusticc.configs_init.model.preprocessor_config import DateRange


def test_get_configs_wrapper_withoutFetchingJson_returnsSampleConfig():
    path = "sample/path"
    configer = Configer(path)

    wrapper = configer.get_configs_wrapper()

    assert (
        wrapper.loader is None
        and wrapper.preprocessor is None
        and wrapper.market is None
        and wrapper.agent is None
    )


def test___fetch_dict_config_validPath_fetchesDictConfigIntoVariable():
    path = "tests/specusticc/configs_init"
    configer = Configer(path)

    configer._Configer__fetch_dict_config()

    config_dict = configer._Configer__dict_config
    assert (
        config_dict["test_parameter"] == "test_value"
        and config_dict["import"]["train_date"]["from"] == datetime(2004, 1, 1)
        and config_dict["import"]["train_date"]["to"] == datetime(2017, 12, 31)
        and config_dict["import"]["test_date"][0]["from"] == datetime(2018, 1, 1)
        and config_dict["import"]["test_date"][0]["to"] == datetime(2019, 6, 30)
    )


def test___create_loader_config_validDict_createsLoaderConfig():
    path = "sample/path"
    configer = Configer(path)
    config_dict = {
        "import": {
            "datasource": "mongodb",
            "input": {"tickers": ["a", "b", "c"]},
            "target": {"tickers": ["d", "e", "f"]},
            "context": {"tickers": ["g", "h", "i"]},
        }
    }
    configer._Configer__dict_config = config_dict

    configer._Configer__create_loader_config()

    wrapper = configer.get_configs_wrapper()
    loader = wrapper.loader

    assert (
        loader.datasource == "mongodb"
        and loader.input_tickers == ["a", "b", "c"]
        and loader.output_tickers == ["d", "e", "f"]
        and loader.context_tickers == ["g", "h", "i"]
    )


def test___create_preprocessor_config_validDict_createsLoaderConfig():
    path = "sample/path"
    configer = Configer(path)
    config_dict = {
        "import": {
            "datasource": "mongodb",
            "input": {
                "tickers": ["a", "b", "c"],
                "columns": ["open", "high", "low", "close", "vol"],
            },
            "target": {"tickers": ["d", "e", "f"], "columns": ["close"]},
            "context": {
                "tickers": ["g", "h", "i"],
                "columns": ["open", "high", "low", "close", "vol"],
            },
            "train_date": {"from": datetime(2008, 1, 1), "to": datetime(2016, 12, 31)},
            "test_date": [
                {"from": datetime(2017, 1, 1), "to": datetime(2019, 12, 31)},
                {"from": datetime(2005, 1, 1), "to": datetime(2007, 12, 31)},
            ],
        },
        "preprocessing": {"window_length": 100, "horizon": 50, "rolling": 25},
    }
    configer._Configer__dict_config = config_dict

    configer._Configer__create_preprocessor_config()

    wrapper = configer.get_configs_wrapper()
    preprocessor = wrapper.preprocessor

    assert (
        preprocessor.input_columns == ["open", "high", "low", "close", "vol", "date"]
        and preprocessor.output_columns == ["close", "date"]
        and preprocessor.context_columns
        == ["open", "high", "low", "close", "vol", "date"]
        and preprocessor.features == 5
        and preprocessor.context_features == 5
        and preprocessor.train_date
        == DateRange(datetime(2008, 1, 1), datetime(2016, 12, 31))
        and preprocessor.test_dates
        == [
            DateRange(datetime(2017, 1, 1), datetime(2019, 12, 31)),
            DateRange(datetime(2005, 1, 1), datetime(2007, 12, 31)),
        ]
        and preprocessor.window_length == 100
        and preprocessor.horizon == 50
        and preprocessor.rolling == 25
    )


def test___create_agent_config_validDict_createsLoaderConfig():
    path = "sample/path"
    configer = Configer(path)
    config_dict = {
        "import": {
            "datasource": "mongodb",
            "input": {
                "tickers": ["a", "b", "c"],
                "columns": ["open", "high", "low", "close", "vol"],
            },
            "context": {
                "tickers": ["g", "h", "i"],
                "columns": ["open", "high", "low", "close", "vol"],
            },
        },
        "preprocessing": {"window_length": 100, "horizon": 50},
        "agent": {"hyperparam_optimization_method": "grid"},
    }
    configer._Configer__dict_config = config_dict

    configer._Configer__create_agent_config()

    wrapper = configer.get_configs_wrapper()
    agent = wrapper.agent

    assert (
        agent.input_timesteps == 100
        and agent.input_features == 12
        and agent.output_timesteps == 50
        and agent.context_timesteps == 100
        and agent.context_features == 12
        and agent.hyperparam_optimization_method == "grid"
        and agent.save_path == "sample/path"
    )


def test___create_market_config_validDict_createsLoaderConfig():
    path = "sample/path"
    configer = Configer(path)
    config_dict = {"agent": {"folds": 10}}
    configer._Configer__dict_config = config_dict

    configer._Configer__create_market_config()

    wrapper = configer.get_configs_wrapper()
    market = wrapper.market

    assert market.n_folds == 10
