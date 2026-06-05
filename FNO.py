import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.fft
from typing import Callable, List, Union    

drop_NN = 0.01
drop_FNO = 0.05

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class NeuronWiseActivation(nn.Module):
    def __init__(self, num_neurons: int, base_activation: nn.Module):
        """
            y = activation(a_i * x_i)
        """
        super().__init__()
        self.base_activation = base_activation
        self.a = nn.Parameter(torch.ones(num_neurons))

    def forward(self, x):
        # x shape: (B, C, L)
        if x.dim() == 3:  # (B, C, L)
            return self.base_activation(self.a.view(1, -1, 1) * x)
        elif x.dim() == 2:  # (B, features)
            return self.base_activation(self.a.view(1, -1) * x)
        else:
            raise ValueError(f"Unsupported input dimension {x.dim()} in NeuronWiseActivation")

class SpectralConv1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes: int):
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
        # x: (B, C, N)
        batchsize, channels, spatial_points = x.shape

        x_hat = torch.fft.rfft(x, dim=-1)                    # (B, C, F)
        x_hat_under_modes = x_hat[:, :, :self.modes]

        weights = torch.complex(self.real_weights, self.imag_weights)
        out_hat_under_modes = torch.einsum("bim,iom->bom", x_hat_under_modes, weights)

        out_hat = torch.zeros(
            batchsize, self.out_channels, x_hat.shape[-1],
            dtype=torch.cfloat, device=x.device
        )
        out_hat[:, :, :self.modes] = out_hat_under_modes

        out = torch.fft.irfft(out_hat, n=spatial_points, dim=-1)
        return out


class Projection_NN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 width: int,
                 depth: int,
                 activation: nn.Module = nn.Identity(),
                 adaptive: bool = False):
        super().__init__()

        def make_activation(num_neurons):
            return NeuronWiseActivation(num_neurons, activation) if adaptive else activation

        layers = [nn.Linear(input_dim, width)]
        layers.append(nn.BatchNorm1d(width))
        layers.append(make_activation(width))
        layers.append(nn.Dropout(drop_NN))

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.BatchNorm1d(width))
            layers.append(make_activation(width))
            layers.append(nn.Dropout(drop_NN))

        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, N)
        bsize, _, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, self.network[0].in_features)
        x = self.network(x)
        x = x.view(bsize, N, self.network[-1].out_features).permute(0, 2, 1).contiguous()
        return x


class ConvNet1d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: nn.Module = nn.Identity(),
                 adaptive: bool = False,
                 replicate_modulus: bool = False):
        super().__init__()

        def make_activation(num_neurons):
            return NeuronWiseActivation(num_neurons, activation) if adaptive else activation

        if replicate_modulus:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
                nn.BatchNorm1d(out_channels // 2),
                make_activation(out_channels // 2),
                nn.Dropout(drop_NN),
                nn.Conv1d(out_channels // 2, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                activation
            )

    def forward(self, x):
        return self.net(x)


class FNOBlock1d(nn.Module):
    """
    One FNO block with separable forward stages so intermediate tensors
    can be inspected easily.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 modes: int,
                 activation: nn.Module = nn.Identity(),
                 adaptive: bool = False):
        super().__init__()

        self.activation = NeuronWiseActivation(out_channels, activation) if adaptive else activation

        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.bypass_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels)
        self.do = nn.Dropout(drop_FNO)

    def forward_spectral(self, x: torch.Tensor):
        return self.spectral_conv(x)

    def forward_bypass(self, x: torch.Tensor):
        return self.bypass_conv(x)

    def forward_pre_activation(self, x: torch.Tensor):
        """
        Output after spectral + bypass + BN, before activation.
        """
        sc = self.forward_spectral(x)
        bc = self.forward_bypass(x)
        out = sc + bc
        out = self.bn(out)
        return out

    def forward_activation(self, x: torch.Tensor):
        return self.activation(x)

    def forward_dropout(self, x: torch.Tensor):
        return self.do(x)

    def forward(self, x: torch.Tensor):
        out = self.forward_pre_activation(x)
        out = self.forward_activation(out)
        out = self.forward_dropout(out)
        return out

    def forward_with_intermediates(self, x: torch.Tensor):
        """
        Returns final block output and a dict of intermediate tensors.
        """
        sc = self.forward_spectral(x)
        bc = self.forward_bypass(x)
        pre_act = self.bn(sc + bc)
        post_act = self.forward_activation(pre_act)
        post_dropout = self.forward_dropout(post_act)

        intermediates = {
            "input": x,
            "spectral": sc,
            "bypass": bc,
            "pre_activation": pre_act,
            "post_activation": post_act,
            "output": post_dropout,
        }
        return post_dropout, intermediates


class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        width: int,
        block_activation: nn.Module = nn.Identity(),
        n_blocks: int = 4,
        padding: int = 0,
        coord_features: bool = False,
        adaptive: bool = False,
        lift_activation: nn.Module = nn.Identity(),
        lift_NN: bool = False,
        lift_NN_params: dict = {},
        decode_activation: nn.Module = nn.Identity(),
        decode_NN: bool = False,
        decode_NN_params: dict = {},
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
        self.adaptive = adaptive

        self.lift_activation = lift_activation
        self.lift_NN = lift_NN
        self.lift_NN_params = lift_NN_params

        self.decode_activation = decode_activation
        self.decode_NN = decode_NN
        self.decode_NN_params = decode_NN_params

        lift_in_channels = self.in_channels + 1 if coord_features else self.in_channels

        if lift_NN:
            self.lift_network = Projection_NN(
                input_dim=lift_in_channels,
                output_dim=self.width,
                width=self.lift_NN_params["width"],
                depth=self.lift_NN_params["depth"],
                activation=self.lift_activation,
                adaptive=self.adaptive
            )
        else:
            self.lift_network = ConvNet1d(
                in_channels=lift_in_channels,
                out_channels=self.width,
                activation=self.lift_activation,
                adaptive=self.adaptive
            )

        if decode_NN:
            self.decode_network = Projection_NN(
                input_dim=self.width,
                output_dim=self.out_channels,
                width=self.decode_NN_params["width"],
                depth=self.decode_NN_params["depth"],
                activation=self.decode_activation,
                adaptive=self.adaptive
            )
        else:
            self.decode_network = ConvNet1d(
                in_channels=self.width,
                out_channels=self.out_channels,
                activation=self.decode_activation,
                adaptive=self.adaptive
            )

        self.fno_blocks = nn.ModuleList([
            FNOBlock1d(width, width, modes, block_activation, adaptive)
            for _ in range(n_blocks)
        ])

        self.apply(init_weights)

    def forward_input(self, x: torch.Tensor):
        """
        Optionally append coordinate features.
        Input/output shape: (B, C, N)
        """
        if self.coord_features:
            bsize, size_x = x.shape[0], x.shape[2]
            grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=x.device)
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
            x = torch.cat((x, grid_x), dim=1)
        return x

    def forward_lift(self, x: torch.Tensor):
        return self.lift_network(x)

    def forward_pad(self, x: torch.Tensor):
        if self.padding != 0:
            x = F.pad(x, (self.padding, self.padding), mode="constant", value=0)
        return x

    def forward_unpad(self, x: torch.Tensor):
        if self.padding != 0:
            x = x[:, :, self.padding:-self.padding]
        return x

    def forward_block(self, block_idx: int, x: torch.Tensor):
        return self.fno_blocks[block_idx](x)

    def forward_decode(self, x: torch.Tensor):
        return self.decode_network(x)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        """
        Standard forward pass.

        If return_intermediates=True, returns:
            output, intermediates_dict
        else:
            output
        """
        if not return_intermediates:
            x = self.forward_input(x)
            x = self.forward_lift(x)
            x = self.forward_pad(x)

            for i in range(self.n_blocks):
                x = self.forward_block(i, x)

            x = self.forward_decode(x)
            x = self.forward_unpad(x)
            return x

        intermediates = {}

        # Input preparation
        x_in = x
        x = self.forward_input(x)
        intermediates["input"] = x_in
        intermediates["input_processed"] = x

        # Lift
        x = self.forward_lift(x)
        intermediates["lift"] = x

        # Pad
        x = self.forward_pad(x)
        intermediates["padded_lift"] = x

        # Blocks
        intermediates["blocks"] = []
        for i, block in enumerate(self.fno_blocks):
            x, block_intermediates = block.forward_with_intermediates(x)
            block_intermediates["block_idx"] = i
            intermediates["blocks"].append(block_intermediates)

        intermediates["before_decode"] = x

        # Decode
        x = self.forward_decode(x)
        intermediates["decoded_padded"] = x

        # Unpad
        x = self.forward_unpad(x)
        intermediates["output"] = x

        return x, intermediates
    
class dual_FNO(nn.Module):
    def __init__(self, FNO_heal: FNO1d, FNO_state: FNO1d):
        """
        Initializes a dual FNO model composed of two independent FNO1d networks.

        Args:
            FNO_heal (FNO1d): FNO network responsible for predicting the healing component.
            FNO_state (FNO1d): FNO network responsible for predicting the state component.
        """

        super().__init__()
        self.FNO_heal = FNO_heal
        self.FNO_state = FNO_state

    def forward(self, x):
        x_state = x[:,0:1,:]
        x_heal = x[:,1:2,:]
        healing = self.FNO_heal(x_heal)
        state = self.FNO_state(x_state)
        return state + healing


