class AgentConfig:
    def __init__(self):
        self.input_timesteps: int = 0
        self.input_features: int = 0
        self.output_timesteps: int = 0

        self.context_timesteps: int = 0
        self.context_features: int = 0

        self.hyperparam_optimization_method: str = ''
        self.horizon_prediction_method: str = 'mimo'

        self.market_save_path: str = ''
