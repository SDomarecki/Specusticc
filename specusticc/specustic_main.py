import sys
import logging
import time

from specusticc.market.market import Market


def _configure_logger():
    # Create and configure logger
    logging.basicConfig(filename='newfile.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')

    # Setting the threshold of logger to INFO (DEBUG sends some strange data)
    logging.getLogger().setLevel(logging.INFO)


if __name__ == "__main__":
    start_time = time.time()

    _configure_logger()
    logging.info('Starting Specusticc')

    config_file = sys.argv[1]
    model_names = sys.argv[2:]
    logging.info('Running with args:')
    logging.info(f'Config file path: {config_file}')
    logging.info(f'Predictive model name: {model_names}')

    config_path = 'raw_configs/' + config_file
    market = Market(config_path=config_path, models=model_names)
    market.run()

    end_time = time.time()
    diff_time = end_time - start_time
    logging.info(f'Ending Specusticc, time elapsed in seconds: {diff_time}')


