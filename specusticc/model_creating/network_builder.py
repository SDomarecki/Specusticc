from configs_init.model.agent_config import AgentConfig
from model_creating.choose_network import choose_network


class NetworkBuilder:
    def __init__(self, model_name: str, config: AgentConfig):
        self.model_name = model_name
        self.config: AgentConfig = config

    def build(self):
        model_builder = choose_network(self.model_name, self.config)
        self.config.epochs = model_builder.epochs
        return model_builder.build_model()
