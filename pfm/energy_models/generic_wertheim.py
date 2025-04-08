from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)

class GenericWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)

