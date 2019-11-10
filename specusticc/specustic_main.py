from specusticc.agent import Agent
import sys

if __name__ == "__main__":
    config_path = sys.argv[1]
    agent = Agent(config_path=config_path)
