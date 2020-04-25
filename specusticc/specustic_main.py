import sys

from specusticc.agent.agent import Agent

if __name__ == "__main__":
    config_file = sys.argv[1]
    model_name = sys.argv[2]

    config_path = 'raw_configs/' + config_file
    agent = Agent(config_path=config_path, model_name=model_name)
    agent.run()
