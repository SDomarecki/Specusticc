import json
from datetime import datetime


class ConfigLoader:
    def __init__(self):
        self.__config = {}

    def get_config(self) -> dict:
        return self.__config

    def load_and_preprocess_config(self, config_path: str):
        self.__load_config(config_path)
        self.__preprocess_config()

    def __load_config(self, config_path: str):
        with open(config_path) as file:
            self.__config = json.load(file)

    def __preprocess_config(self):
        self.__transform_datestring_to_datetime()

    def __transform_datestring_to_datetime(self):
        date_format = "%Y-%m-%d"

        import_dict_config = self.__config.get("import") or {}

        import_train_date = import_dict_config.get("train_date") or {}
        import_train_date["from"] = datetime.strptime(
            import_train_date["from"], date_format
        )
        import_train_date["to"] = datetime.strptime(
            import_train_date["to"], date_format
        )
        self.__config["import"]["train_date"] = import_train_date

        import_test_date = import_dict_config.get("test_date") or []
        date_ranges = []
        for date_range in import_test_date:
            from_date_obj = datetime.strptime(date_range["from"], date_format)
            to_date_obj = datetime.strptime(date_range["to"], date_format)
            date_ranges.append({"from": from_date_obj, "to": to_date_obj})
        self.__config["import"]["test_date"] = date_ranges
