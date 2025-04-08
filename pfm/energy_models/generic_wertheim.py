from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
from pfm.utils.constants import kb, Species
from pfm.utils.delta import Delta

# TODO: Need to come back to this, the C++ feels wrong?

class GenericWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)
        self.species = []
        wertheim_config = config.get('generic_wertheim')
        self._delta = Delta(wertheim_config.get('delta'))
        self._n_patches = 0
        self._B2 = wertheim_config.get('B2')
        self._B3 = wertheim_config.get('B3')

        # Scale constants if necesary, usually the scaling factor is 1 unless distance parameter set
        self._B2 *= (self._inverse_scaling_factor ** 3)
        self._B3 *= (self._inverse_scaling_factor ** 3)
        self._delta.delta *= (self._inverse_scaling_factor ** 3)  # Need to access actual value of delta (_delta.delta)


