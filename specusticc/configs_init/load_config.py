import json
from datetime import datetime


def load_and_preprocess_config(config_path: str) -> dict:
    config = _load_config(config_path)
    config = _preprocess_config(config)
    return config


def _load_config(config_path: str) -> dict:
    file = open(config_path)
    return json.load(file)


def _preprocess_config(config: dict) -> dict:
    config = _transform_datestring_to_datetime(config)
    return config


def _transform_datestring_to_datetime(config: dict) -> dict:
    date_format = "%Y-%m-%d"

    import_train_date = config["import"]["train_date"]
    import_train_date["from"] = datetime.strptime(
        import_train_date["from"], date_format
    )
    import_train_date["to"] = datetime.strptime(import_train_date["to"], date_format)
    config["import"]["train_date"] = import_train_date

    import_test_date = config["import"]["test_date"]
    date_ranges = []
    for date_range in import_test_date:
        from_date_obj = datetime.strptime(date_range["from"], date_format)
        to_date_obj = datetime.strptime(date_range["to"], date_format)
        date_ranges.append({"from": from_date_obj, "to": to_date_obj})
    config["import"]["test_date"] = date_ranges
    return config
