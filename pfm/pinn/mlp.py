from flax import linen as nn
import jax.numpy as jnp
import jax

class FourierFeatures(nn.Module):
    output_dim: int
    scale: float = 1.0
    trainable: bool = False  # new flag

    @nn.compact
    def __call__(self, x):
        # x has shape (..., input_dim)
        key = self.make_rng("params")
        shape = (x.shape[-1], self.output_dim // 2)
        if self.trainable:
            B = self.param("B", nn.initializers.normal(stddev=self.scale), shape)
        else:
            B = self.variable("constants", "B",
                              lambda _: jax.random.normal(key, shape) * self.scale).value
        proj = x @ B
        return jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)

class MLP(nn.Module):
    input_dim: int        # e.g., 3 for (x,y,t)
    output_dim: int       # e.g., N species
    layers: list[int]     # e.g., [64, 64, 64]
    activation: callable = nn.tanh
    use_fourier_features: bool = True
    fourier_dim: int = 64  # Number of fourier features, however can lead to overfitting
    fourier_scale: float = 1.0  # How rapidly the features vary

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
        out = nn.Dense(self.output_dim,
                       kernel_init=nn.initializers.glorot_normal(),
                       bias_init=nn.initializers.zeros)(h)
        return out