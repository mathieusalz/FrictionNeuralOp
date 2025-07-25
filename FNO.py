import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List, Union

class SpectralConv1d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes: int

    def __init__(
            self,
            in_channels,
            out_channels,
            modes,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1.0 / (in_channels * out_channels)

        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(
            real_key,
            (in_channels, out_channels, modes),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights = jax.random.uniform(
            imag_key,
            (in_channels, out_channels, modes),
            minval=-scale,
            maxval=+scale,
        )

    def complex_mult1d(
            self,
            x_hat,
            w,
    ):
        return jnp.einsum("iM,ioM->oM", x_hat, w)


    def __call__(
            self,
            x,
    ):
        channels, spatial_points = x.shape

        # shape of x_hat is (in_channels, spatial_points//2+1)
        x_hat = jnp.fft.rfft(x)
        # shape of x_hat_under_modes is (in_channels, self.modes)
        x_hat_under_modes = x_hat[:, :self.modes]
        weights = self.real_weights + 1j * self.imag_weights
        # shape of out_hat_under_modes is (out_channels, self.modes)
        out_hat_under_modes = self.complex_mult1d(x_hat_under_modes, weights)

        # shape of out_hat is (out_channels, spatial_points//2+1)
        out_hat = jnp.zeros(
            (self.out_channels, x_hat.shape[-1]),
            dtype=x_hat.dtype
        )
        out_hat = out_hat.at[:, :self.modes].set(out_hat_under_modes)

        out = jnp.fft.irfft(out_hat, n=spatial_points)

        return out


class FNOBlock1d(eqx.Module):
    spectral_conv: SpectralConv1d
    bypass_conv: eqx.nn.Conv1d
    activation: Union[Callable, None]

    def __init__(
            self,
            in_channels,
            out_channels,
            modes,
            activation,
            *,
            key,
    ):
        self.activation = activation
        spectral_conv_key, bypass_conv_key = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(
            in_channels,
            out_channels,
            modes,
            key=spectral_conv_key,
        )
        self.bypass_conv = eqx.nn.Conv1d(
            in_channels,
            out_channels,
            1,  # Kernel size is one
            key=bypass_conv_key,
        )

    def __call__(
            self,
            x,
    ):
        if self.activation is None:
            return self.spectral_conv(x) + self.bypass_conv(x)
        else: 
            return self.activation(self.spectral_conv(x) + self.bypass_conv(x))
        
    
class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d

    # Store hyperparameters
    in_channels: int
    out_channels: int
    modes: int
    width: int
    activation: Callable
    n_blocks: int
    padding: int
    padding_array: jax.Array

    def __init__(
            self,
            in_channels,
            out_channels,
            modes,
            width,
            activation=None,
            n_blocks = 4,
            padding = 0,
            *,
            key=jax.random.PRNGKey(0),
    ):
        # Save hyperparameters as attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.activation = activation
        self.n_blocks = n_blocks
        self.padding = padding
        if self.padding != 0:
            self.padding_array = jnp.zeros(padding)[None,:]
        else:
            self.padding_array = None

        key, lifting_key = jax.random.split(key)
        self.lifting = eqx.nn.Conv1d(
            in_channels,
            width,
            1,
            key=lifting_key,
            use_bias = False
        )

        self.fno_blocks = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)
            self.fno_blocks.append(FNOBlock1d(
                width,
                width,
                modes,
                activation,
                key=subkey,
            ))

        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv1d(
            width,
            out_channels,
            1,
            key=projection_key,
        )

    def __call__(self, x):
        if self.padding_array is not None:
            x = jnp.concatenate([self.padding_array, x, self.padding_array], axis=1) 
        x = self.lifting(x)
        for fno_block in self.fno_blocks:
            x = fno_block(x)
        x = self.projection(x)
        x = x[:,self.padding:x.shape[1]-self.padding]
        return x