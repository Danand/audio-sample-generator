import torch

import torch.nn as nn

class SpectrogramsGeneratorModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
    ):
        super(SpectrogramsGeneratorModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, device=device),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.layers(input)
