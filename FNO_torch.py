import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from typing import Callable, List, Union


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)
        self.real_weights = nn.Parameter(
            torch.rand(in_channels, out_channels, modes) * 2 * scale - scale
        )
        self.imag_weights = nn.Parameter(
            torch.rand(in_channels, out_channels, modes) * 2 * scale - scale
        )

    def complex_mult1d(self, x_hat, w):
        return torch.einsum("iM,ioM->oM", x_hat, w)

    def forward(self, x):
        # x: (batch, channels, spatial_points)
        batchsize, channels, spatial_points = x.shape

        x_hat = torch.fft.rfft(x, dim=-1)  # (B, C, F)
        x_hat_under_modes = x_hat[:, :, :self.modes]

        weights = torch.complex(self.real_weights, self.imag_weights)
        out_hat_under_modes = torch.einsum("bim, iom -> bom", x_hat_under_modes, weights)

        out_hat = torch.zeros(
            batchsize, self.out_channels, x_hat.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        out_hat[:, :, :self.modes] = out_hat_under_modes

        out = torch.fft.irfft(out_hat, n=spatial_points, dim=-1)
        return out


class FNOBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, activation=None):
        super().__init__()
        self.activation = activation
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.bypass_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        sc = self.spectral_conv(x)
        bc = self.bypass_conv(x)
        out = sc + bc
        return out if self.activation is None else self.activation(out)


class Projection_NN(nn.Module):
    def __init__(self, input_dim, output_dim, width, depth, activation=torch.tanh):
        super().__init__()
        self.activation = activation
        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, output_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: (B, N, in_features)
        for layer in self.layers[:-1]:
            x = layer(x) if self.activation is None else self.activation(layer(x))
        return self.layers[-1](x)


class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        width,
        block_activation=None,
        lifting_activation=None,
        n_blocks=4,
        padding=0,
        NN=False,
        NN_params=None,
        bias=True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.block_activation = block_activation
        self.lifting_activation = lifting_activation
        self.n_blocks = n_blocks
        self.NN = NN
        self.NN_params = NN_params
        self.bias = bias
        self.padding = padding

        if not NN:
            self.lifting = nn.Conv1d(in_channels, width, kernel_size=1, bias=bias)
            self.projection = nn.Conv1d(width, out_channels, kernel_size=1)
        else:
            self.lifting = Projection_NN(
                input_dim=1,
                output_dim=width,
                width=NN_params["width"],
                depth=NN_params["depth"],
                activation=lifting_activation,
            )
            self.projection = Projection_NN(
                input_dim=width,
                output_dim=1,
                width=NN_params["width"],
                depth=NN_params["depth"],
                activation=lifting_activation,
            )

        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, width, modes, block_activation)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x: (B, C, N)
        if self.padding != 0:
            x = F.pad(x, (self.padding, self.padding), mode='constant', value=0)

        if self.NN:
            # reshape to (B * N, 1)
            B, C, N = x.shape
            assert C == 1, "Projection_NN requires in_channels=1"
            x = x.permute(0, 2, 1).reshape(B * N, 1)
            x = self.lifting(x)  # (B*N, width)
            x = x.reshape(B, N, self.width).permute(0, 2, 1)  # (B, width, N)
        else:
            x = self.lifting(x)

        if self.lifting_activation is not None:
            x = self.lifting_activation(x)

        for block in self.fno_blocks:
            x = block(x)

        if self.NN:
            # reshape to (B * N, width)
            B, W, N = x.shape
            x = x.permute(0, 2, 1).reshape(B * N, W)
            x = self.projection(x)  # (B*N, 1)
            x = x.reshape(B, N).unsqueeze(1)  # (B, 1, N)
        else:
            x = self.projection(x)

        if self.padding != 0:
            x = x[:, :, self.padding:-self.padding]

        return x

class dual_FNO(nn.Module):
    def __init__(self, FNO_heal: FNO1d, FNO_state: FNO1d):
        super().__init__()
        self.FNO_heal = FNO_heal
        self.FNO_state = FNO_state

    def forward(self, x):
        x_state = x[:,0:1,:]
        x_heal = x[:,1:2,:]
        healing = self.FNO_heal(x_heal)
        state = self.FNO_state(x_state)
        return state + healing


