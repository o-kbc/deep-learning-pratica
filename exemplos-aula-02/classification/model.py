
import torch
import torch.nn as nn

from efficientnet import EfficientNet

class NeuralNetwork(nn.Module):
    def __init__(self, num_out=1):
        super().__init__()
        self.num_out = num_out
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, num_out)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def get_efficientNet(output_size):
    return EfficientNet(output_size)