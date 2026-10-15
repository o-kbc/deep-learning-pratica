
import torch
from torchviz import make_dot

#######################################################################
#######################################################################

def sample_1():
    
    x = torch.tensor(1.0, requires_grad=True)

    o = x + 2

    o.backward()

    print(x.grad)

#######################################################################
#######################################################################

def sample_2():

    x = torch.randn((3, 3), requires_grad=True)

    print(x)

    def f(x):
        return x**2 + 10*x + 3

    o = f(x).sum()

    o.backward()
    print(x)

#######################################################################
#######################################################################

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(3*3, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1))

    def forward(self, x):
        return self.linear_relu_stack(torch.flatten(x))


def sample_3():

    x = torch.randn((3,3))
    model = NeuralNetwork()
    opt = torch.optim.SGD(params=model.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()
    
    print(x)
    print(model)
    print(model(x))

    o = loss(model(x), torch.tensor(100.))

    print(o)

    o.backward()

    print(f"First weights line before step: {model.linear_relu_stack[0].weight[0]}")
    print(f"First Grad line before step: {model.linear_relu_stack[0].weight.grad[0]}")

    opt.step()

    print(f"First weights line after step: {model.linear_relu_stack[0].weight[0]}")
    print(f"First Grad line after step: {model.linear_relu_stack[0].weight.grad[0]}")

    opt.zero_grad()

    print(f"First weights line after zero_grad: {model.linear_relu_stack[0].weight[0]}")
    print(f"First Grad line after zero_grad: {model.linear_relu_stack[0].weight.grad}")

#######################################################################
#######################################################################

if __name__ == "__main__":
    #sample_1()
    #sample_2()
    sample_3()
    #pass