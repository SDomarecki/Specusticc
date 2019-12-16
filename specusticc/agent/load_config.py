import json
from datetime import datetime


def load_config(config_path) -> dict:
    file = open(config_path)
    config = json.load(file)

    config = _transform_datestring_to_datetime(config)
    return config


def _transform_datestring_to_datetime(config: dict) -> dict:
    date_format = '%Y-%m-%d'

    if 'train_date' in config['import']:
        import_train_date = config['import']['train_date']
        import_train_date['from'] = datetime.strptime(import_train_date['from'], date_format)
        import_train_date['to'] = datetime.strptime(import_train_date['to'], date_format)
        config['import']['train_date'] = import_train_date

    if 'test_date' in config['import']:
        import_test_date = config['import']['test_date']
        import_test_date['from'] = datetime.strptime(import_test_date['from'], date_format)
        import_test_date['to'] = datetime.strptime(import_test_date['to'], date_format)
        config['import']['test_date'] = import_test_date

    return config
