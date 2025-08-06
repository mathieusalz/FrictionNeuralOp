import torch
import torch.nn as nn
import torch.fft
from typing import Callable, List, Union    

class SpectralConv1d(nn.Module):
    def __init__(self, 
                 in_channels : int, 
                 out_channels: int, 
                 modes: int):
        
        """
        Initializes a 1D spectral convolution layer using complex weights in the Fourier domain.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (int): Number of Fourier modes to retain (low-frequency modes).
        """
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
    def __init__(self, 
                 in_channels : int, 
                 out_channels : int, 
                 modes : int, 
                 activation : nn.Module = nn.Identity()):
        """
        Initializes a single Fourier Neural Operator (FNO) block for 1D data.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (int): Number of retained Fourier modes in the spectral convolution.
            activation (nn.Module, optional): Activation function applied after the block output. Default is nn.Identity().
        """
        
        super().__init__()
        self.activation = activation
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.bypass_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        sc = self.spectral_conv(x)
        bc = self.bypass_conv(x)
        out = sc + bc
        return self.activation(out)

class Projection_NN(nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 output_dim : int, 
                 width : int, 
                 depth : int, 
                 activation: nn.Module = nn.Identity()):
        
        """
        Initializes a fully-connected feedforward projection network.

        Args:
            input_dim (int): Input dimensionality.
            output_dim (int): Output dimensionality.
            width (int): Width (number of hidden units) in each hidden layer.
            depth (int): Number of layers in the network (including input/output layers).
            activation (nn.Module, optional): Activation function applied between layers. Default is nn.Identity().
        """
        
        super().__init__()
        layers = [nn.Linear(input_dim, width), activation]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(activation)
        layers.append(nn.Linear(width, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        bsize, _, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, self.network[0].in_features)
        x = self.network(x)
        x = x.view(bsize, N, self.network[-1].out_features).permute(0, 2, 1).contiguous()
        return x

class ConvNet1d(nn.Module):
    """
    A simple 1D convolutional network with a hidden layer and activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (nn.Module, optional): Activation function between the convolutional layers. Default is nn.Identity().
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 activation: nn.Module = nn.Identity()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 2, kernel_size=1),
            activation,
            nn.Conv1d(out_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

class FNO1d(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        modes : int,
        width : int,
        block_activation : nn.Module = nn.Identity(),
        n_blocks : int = 4,
        padding : int = 0,
        coord_features : bool = False,
        lift_activation : nn.Module = nn.Identity(),
        lift_NN : bool = False,
        lift_NN_params : dict = {},
        decode_activation : nn.Module = nn.Identity(),
        decode_NN : bool = False,
        decode_NN_params : dict = {},
    ):
        """
        Initializes a 1D Fourier Neural Operator (FNO) model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (int): Number of Fourier modes retained in spectral convolutions.
            width (int): Width of the internal FNO layers.
            block_activation (nn.Module, optional): Activation applied after each FNO block. Default is nn.Identity().
            n_blocks (int, optional): Number of stacked FNO blocks. Default is 4.
            padding (int, optional): Amount of zero-padding added to input and removed from output. Default is 0.
            coord_features (bool, optional): Whether to append coordinate information to input. Default is False.
            lift_activation (nn.Module, optional): Activation function for the lifting network. Default is nn.Identity().
            lift_NN (bool, optional): If True, use a projection MLP to lift input; else use convolution. Default is False.
            lift_NN_params (dict, optional): Dictionary with keys "width" and "depth" for lifting MLP. Default is {}.
            decode_activation (nn.Module, optional): Activation function for the decoding network. Default is nn.Identity().
            decode_NN (bool, optional): If True, use a projection MLP to decode output; else use convolution. Default is False.
            decode_NN_params (dict, optional): Dictionary with keys "width" and "depth" for decoding MLP. Default is {}.
        """

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
            self.lift_network = ConvNet1d(self.in_channels, self.width, 
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
            self.decode_network = ConvNet1d(self.width,
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
            x = nn.functional.pad(x, (self.padding, self.padding), mode='constant', value=0)

        for block in self.fno_blocks:
            x = block(x)

        x = self.decode_network(x)

        if self.padding != 0:
            x = x[:, :, self.padding:-self.padding]

        return x

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


