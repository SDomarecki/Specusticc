from specusticc.configs_init.loader_config import LoaderConfig
from specusticc.configs_init.postprocessor_config import PostprocessorConfig
from specusticc.configs_init.preprocessor_config import PreprocessorConfig
from specusticc.configs_init.model_creator_config import ModelCreatorConfig
from specusticc.configs_init.training_config import TrainingConfig
from specusticc.configs_init.testing_config import TestingConfig
from specusticc.configs_init.reporter_config import ReporterConfig


class ConfigsWrapper:
    def __init__(self):
        self.loader: LoaderConfig = None
        self.preprocessor: PreprocessorConfig = None
        self.model_creator: ModelCreatorConfig = None
        self.training: TrainingConfig = None
        self.testing: TestingConfig = None
        self.postprocessor: PostprocessorConfig = None
        self.reporter: ReporterConfig = None
