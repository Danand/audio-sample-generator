from audio_sample_generator import constants

from dataclasses import dataclass

@dataclass
class CommonData:
    width: int = 0
    height: int = 0
    output_model_name: str = constants.OUTPUT_MODEL_NAME

