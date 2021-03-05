from specusticc.configs_init.model.agent_config import AgentConfig
from specusticc.configs_init.model.loader_config import LoaderConfig
from specusticc.configs_init.model.market_config import MarketConfig
from specusticc.configs_init.model.preprocessor_config import PreprocessorConfig


class ConfigsWrapper:
    def __init__(self):
        self.loader: LoaderConfig = None
        self.preprocessor: PreprocessorConfig = None
        self.market: MarketConfig = None
        self.agent: AgentConfig = None
