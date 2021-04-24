import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema

from .feature_config import FeatureParams
from .split_config import SplittingParams
from .train_config import TrainingParams


@dataclass()
class TrainingPipelineParams:
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
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
