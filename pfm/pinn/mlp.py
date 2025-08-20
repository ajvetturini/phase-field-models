from flax import linen as nn


class MLP(nn.Module):
    input_dim: int        # e.g., 3 for (x,y,t)
    output_dim: int       # e.g., N species
    layers: list[int]     # e.g., [64, 64, 64]
    activation: callable = nn.tanh

    @nn.compact
    def __call__(self, xyt):
        h = xyt
        for width in self.layers:
            h = nn.Dense(width)(h)
            h = self.activation(h)
        out = nn.Dense(self.output_dim)(h)
        return out