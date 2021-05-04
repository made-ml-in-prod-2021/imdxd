from dataclasses import dataclass, field


@dataclass()
class SplittingParams:

    """
    Class with config for data splitting
    """

    val_size: float = field(default=0.2)
    stratify: bool = field(default=False)
