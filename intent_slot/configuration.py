from dataclasses import dataclass
import yaml


@dataclass
class ClassifierConfig:
    """Class for configuring an intent-slot model training experiment"""

    name: str
    base_model: str
    data: str
    max_len: int
    learning_rate: float
    batch_size: int
    seed: int
    max_epochs: int
    max_steps: int

    def __post_init__(self):
        self.learning_rate = float(self.learning_rate)


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
