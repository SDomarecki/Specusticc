from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.model_creating.models.gan import GAN


class GANWrapper:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.epochs = 200
        pass

    def build_model(self):
        return GAN(self.config, self.epochs)