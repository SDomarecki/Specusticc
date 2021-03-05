from datetime import datetime

from specusticc.configs_init.config_loader import ConfigLoader


def test_load_and_preprocess_config_validPath_returnsPreprocessedConfigDict():
    config_path = "tests/specusticc/configs_init/config.json"

    loader = ConfigLoader()
    loader.load_and_preprocess_config(config_path)
    config_dict = loader.get_config()

    assert (
        config_dict["test_parameter"] == "test_value"
        and config_dict["import"]["train_date"]["from"] == datetime(2004, 1, 1)
        and config_dict["import"]["train_date"]["to"] == datetime(2017, 12, 31)
        and config_dict["import"]["test_date"][0]["from"] == datetime(2018, 1, 1)
        and config_dict["import"]["test_date"][0]["to"] == datetime(2019, 6, 30)
    )


def test__load_config_validPath_returnsConfigDict():
    config_path = "tests/specusticc/configs_init/config.json"

    loader = ConfigLoader()
    loader._ConfigLoader__load_config(config_path)
    config_dict = loader.get_config()

    assert (
        config_dict["test_parameter"] == "test_value"
        and config_dict["import"]["train_date"]["from"] == "2004-01-01"
        and config_dict["import"]["train_date"]["to"] == "2017-12-31"
        and config_dict["import"]["test_date"][0]["from"] == "2018-01-01"
        and config_dict["import"]["test_date"][0]["to"] == "2019-06-30"
    )
