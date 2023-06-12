from dataclasses import dataclass
import yaml


@dataclass
class IntentSlotConfig:
    """Class for configuring an intent-slot model training experiment"""

    name: str
    base_model: str
    data: str
    max_len: int


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
