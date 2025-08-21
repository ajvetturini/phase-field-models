from flax import linen as nn
import jax.numpy as jnp

class FourierFeatures(nn.Module):
    output_dim: int
    scale: float = 10.0  # This is a tunable hyperparameter

    @nn.compact
    def __call__(self, x):
        # x has shape (..., input_dim)
        # B matrix for random Fourier features
        B = self.param('B', nn.initializers.normal(stddev=self.scale), (x.shape[-1], self.output_dim // 2))

        # Project input and apply sin/cos
        proj = x @ B
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

class MLP(nn.Module):
    input_dim: int        # e.g., 3 for (x,y,t)
    output_dim: int       # e.g., N species
    layers: list[int]     # e.g., [64, 64, 64]
    activation: callable = nn.tanh
    use_fourier_features: bool = True
    fourier_dim: int = 256  # Number of fourier features
    fourier_scale: float = 10.0  # How rapidly the features vary

    @nn.compact
    def __call__(self, xyt):
        h = xyt
        if self.use_fourier_features:
            # Apply fourier features and concatenate with original input
            f = FourierFeatures(self.fourier_dim, self.fourier_scale)(h)
            h = jnp.concatenate([h, f], axis=-1)

        for width in self.layers:
            h = nn.Dense(width)(h)
            h = self.activation(h)
        out = nn.Dense(self.output_dim)(h)
        return out