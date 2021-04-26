import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from .feature_config import FeatureParams
from .split_config import SplittingParams
from .train_config import TrainingParams


@dataclass()
class TrainingPipelineParams:
    """
    Class with full training pipeline config (splitting, column definition, training)
    """

    raw_data: str
    experiment_name: str
    random_state: int
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    label: str = field(default="target")
    threshold: float = field(default=0.5)


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    """
    Reading config file and converting to train pipeline config
    :param path: path to config
    :return: Class with parsed configs
    """
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
