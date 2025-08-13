import torch
import torch.nn as nn


class basicNN(nn.Module):
    def __init__(self, width, depth, activation):
        super(basicNN, self).__init__()
        self.width = width
        self.depth = depth
        self.activation = activation

        # Register layers using ModuleList
        layers = []
        layers.append(nn.Linear(2, width))  # Input layer for (x, y)
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, 1))  # Output layer

        self.layers = nn.ModuleList(layers)

    def forward(self, x, y):
        # Ensure x and y are tensors
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        # Broadcast if necessary to match batch dimensions
        x = x.expand_as(y)

        # Stack x and y along last dimension
        out = torch.stack((x, y), dim=-1)  # Shape: [batch_size, 2]

        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)  # No activation at final layer

        return out.squeeze(-1)  # Shape: [batch_size]

# The Input to the Neural Operator is a function f(x) defined on a 1D grid of points
# Have to feed in grid of points x as well as the y points for the integral
class Integral_NO(nn.Module):
    def __init__(self, width, depth, activation = torch.tanh):
        super(Integral_NO, self).__init__()
        self.nn = basicNN(width, depth, activation)

    def forward(self, func_x, y):

        # Input to NN will be [batch, 1, length]
        batch = func_x.shape[0]
        length = func_x.shape[2
        # Reshape to get a 2D tensor with shape [batch * length, 1]
        func_x = func_x.repeat(1, 1, length)
        func_x = func_x.permute(0, 2, 1).reshape(-1, 1)

        # y should be of shape [length, 1]
        # Now to tile the y points to match x
        dA = y[1] - y[0]
        y_repeat = y.repeat(length, 1)

        x_repeat = y.repeat_interleave(length, dim = 0)
        
        # Pass through the NN
        outputs = self.nn(x_repeat, y_repeat).repeat(batch, 1)

        outputs = outputs.reshape(length * batch, length)
        func_x = func_x.reshape(length * batch, length)

        # Approximate integral using dot product with dA
        integral = torch.einsum('ij, ij -> i', outputs, func_x) * dA
        integral = integral.reshape(batch,length).unsqueeze(1)
        return integral
