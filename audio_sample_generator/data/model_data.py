from dataclasses import dataclass

from torch import Tensor

@dataclass
class ModelData:
    input_size: int
    hidden_size: int
    output_size: int
    output_height: int
    output_width: int
    path: str
    normalization_mean: Tensor
    normalization_std: Tensor
