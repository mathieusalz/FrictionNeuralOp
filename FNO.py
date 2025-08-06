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
    def __init__(self, input_dim, output_dim, width, depth, activation):
        super().__init__()
        self.activation = activation
        self.width = width
        self.output_dim = output_dim

        layers = [nn.Linear(input_dim, width)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(activation)

        layers.append(nn.Linear(width, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        bsize, _, N = x.shape
        x = x.permute(0, 2, 1).contiguous()  # (B, N, C)
        x = x.view(-1, self.layers[0].in_features)  # (B*N, C)

        for layer in self.layers[:-1]:
            x = layer(x) if self.activation is None else self.activation(layer(x))
        
        x = self.layers[-1](x)
        x = x.view(bsize, N, self.output_dim)  # (B, N, out_features)
        x = x.permute(0, 2, 1).contiguous()    # (B, out_features, N)
        
        return x

def build_conv_network(in_channels, out_channels, activation):
    conv_network = torch.nn.Sequential()
    conv_network.append(
        nn.Conv1d(in_channels, int(out_channels / 2), kernel_size = 1)
    )
    conv_network.append(activation)
    conv_network.append(
        nn.Conv1d(int(out_channels / 2), out_channels, kernel_size = 1)
    )

    return conv_network


class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
        width,
        block_activation = None,
        n_blocks = 4,
        padding = 0,
        coord_features = False,
        lift_activation = None,
        lift_NN = False,
        lift_NN_params = None,
        decode_activation = None,
        decode_NN = False,
        decode_NN_params = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.block_activation = block_activation
        self.n_blocks = n_blocks
        self.padding = padding
        self.coord_features = coord_features
        
        self.lift_activation = lift_activation
        self.lift_NN = lift_NN
        self.lift_NN_params = lift_NN_params

        self.decode_activation = decode_activation
        self.decode_NN_params = decode_NN_params

        if coord_features == True:
            self.in_channels += 1

        if lift_NN:
            self.lift_network = Projection_NN(
                input_dim = self.in_channels,
                output_dim = self.width,
                width = self.lift_NN_params["width"],
                depth = self.lift_NN_params["depth"],
                activation = self.lift_activation,
            )
        else:
            self.lift_network = build_conv_network(self.in_channels, self.width, 
                                                   self.lift_activation)
            
        if decode_NN:
            self.decode_network = Projection_NN(
                input_dim = self.width,
                output_dim = self.out_channels,
                width = self.decode_NN_params["width"],
                depth = self.decode_NN_params["depth"],
                activation = decode_activation,
            )
        else:
            self.decode_network = build_conv_network(self.width, 
                                                     self.out_channels,
                                                     self.decode_activation)

        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, width, modes, block_activation)
            for _ in range(n_blocks)
        ])

    # x: (B, C, N)
    def forward(self, x):

        if self.coord_features:
            bsize, size_x = x.shape[0], x.shape[2]
            grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=x.device)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
            x = torch.cat((x, grid_x), dim=1)

        x = self.lift_network(x)

        if self.padding != 0:
            x = F.pad(x, (self.padding, self.padding), mode='constant', value=0)

        for block in self.fno_blocks:
            x = block(x)

        x = self.decode_network(x)

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


