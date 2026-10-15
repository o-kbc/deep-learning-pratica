import torch
import torch.nn as nn

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_out=1):
        super().__init__()
        self.num_out = num_out
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_out))

    def forward(self, x):
        return self.linear_relu_stack(torch.flatten(x))


if __name__ == "__main__":

    model = NeuralNetwork(1)
    input = torch.randn((3,3))
    output = model(input)

    print(model)

    print(dir(model))

    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model = NeuralNetwork(10)

    print(model)

    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

