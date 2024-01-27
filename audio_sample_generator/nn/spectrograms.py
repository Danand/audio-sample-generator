import torch.nn as nn

class SpectrogramsModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SpectrogramsModule, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU(0.1)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)

        return x

