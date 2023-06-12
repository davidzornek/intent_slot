import dataclasses


@dataclass
class IntentSlotConfig:
	"""Class for configuring an intent-slot model training experiment"""
	name: str
	base_model: str
	data_path: str
	max_len: int
