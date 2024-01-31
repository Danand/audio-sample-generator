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

        self.layer1 = nn.Linear(input_size, hidden_size, device=device)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size, device=device)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.tanh(x)

        return x

