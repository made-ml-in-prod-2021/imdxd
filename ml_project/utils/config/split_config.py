from dataclasses import dataclass, field


@dataclass()
class SplittingParams:

    label: str
    random_state: float = field(default=42)
    val_size: float = field(default=0.2)
    stratify: bool = field(default=False)
