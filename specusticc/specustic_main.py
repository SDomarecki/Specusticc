import sys

from specusticc.agent.agent import Agent

if __name__ == "__main__":
    config_path = sys.argv[1]
    model_name = sys.argv[2]
    agent = Agent(config_path=config_path, model_name=model_name)
