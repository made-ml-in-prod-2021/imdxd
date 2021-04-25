import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema


@dataclass()
class EvaluationParams:
    """
    Class for evaluation pipeline config
    """
    model: str
    raw_data: str
    proceed_data: str
    threshold: float = field(default=0.5)


EvaluationParamsSchema = class_schema(EvaluationParams)


def read_evaluation_pipeline_params(path: str) -> EvaluationParams:
    """
    Reading evaluation params from file and converting to dataclass
    :param path: path of config
    :return: dataclass with config
    """
    with open(path, "r") as input_stream:
        schema = EvaluationParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
