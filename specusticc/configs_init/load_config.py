import json
from datetime import datetime


def load_and_preprocess_config(config_path: str, model_name: str) -> dict:
    config = _load_config(config_path)
    config = _preprocess_config(config, model_name)
    return config


def _load_config(config_path: str) -> dict:
    file = open(config_path)
    return json.load(file)


def _preprocess_config(config: dict, model_name: str) -> dict:
    config = _transform_datestring_to_datetime(config)
    config['model']['name'] = model_name
    config = _boost_config_with_model(config)
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


def _boost_config_with_model(config: dict) -> dict:
    simple_dim_list = ['basic', 'mlp']
    if config['model']['name'] in simple_dim_list:
        config['model']['input_dim'] = 2
    else:
        config['model']['input_dim'] = 3
    return config
