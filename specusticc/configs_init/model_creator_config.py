class ModelCreatorConfig:
    def __init__(self):
        self.model_type = ''
        self.machine_learning_target = ''
        self.nn_name = ''
        self.nn_input_timesteps = 0
        self.nn_input_features = 0
        self.nn_output_timesteps = 0
        self.t_max_depth = 0

        self.nn_context_timesteps = 0
        self.nn_context_features = 0
        pass
