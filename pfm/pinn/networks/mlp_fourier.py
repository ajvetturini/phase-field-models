import flax.linen as nn
import jax
import jax.numpy as jnp

class MLPFourier(nn.Module):
    features: list
    B_scale: float = 2.0  # Scale for Fourier features
    output_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        # Generate Fourier features with adaptive scaling
        D = x.shape[-1]  # Input dimension
        M = 16  # Number of Fourier features per dimension

        # Create a fixed random matrix for Fourier features
        key = jax.random.PRNGKey(0)
        B = jax.random.normal(key, (D, M)) * self.B_scale

        # Apply Fourier feature mapping
        if x.ndim == 1:
            x = x[None, :]
        x_proj = 2.0 * jnp.pi * x @ B
        x_fourier = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

        # Original inputs might also be useful
        x = jnp.concatenate([x, x_fourier], axis=-1)

        # Apply MLP layers with skip connections
        residual = x
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)

            # Add skip connection every 2 layers if dimensions match
            if i % 2 == 1 and i > 0 and x.shape[-1] == residual.shape[-1]:
                x = x + residual
                residual = x

        # Final layer without activation
        x = nn.Dense(self.features[-1])(x)
        return self.output_scale * x
