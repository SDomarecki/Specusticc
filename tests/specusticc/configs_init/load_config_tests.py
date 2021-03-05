from datetime import datetime

from specusticc.configs_init.load_config import load_and_preprocess_config


def test_load_and_preprocess_config_validPath_returnsPreprocessedConfigDict():
    config_path = "tests/specusticc/configs_init/test_config.json"

    config_dict = load_and_preprocess_config(config_path)

    assert (
        config_dict["test_parameter"] == "test_value"
        and config_dict["import"]["train_date"]["from"] == datetime(2004, 1, 1)
        and config_dict["import"]["train_date"]["to"] == datetime(2017, 12, 31)
        and config_dict["import"]["test_date"][0]["from"] == datetime(2018, 1, 1)
        and config_dict["import"]["test_date"][0]["to"] == datetime(2019, 6, 30)
    )
