from dataclasses import dataclass, field


@dataclass()
class TrainingParams:

    model_type: str
    mean_alpha: int = field(default=20)
    imput_strategy: str = field(default="mean")
