from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial
jax.config.update("jax_enable_x64", True)
from pfm.utils.delta import Delta

class SalehWertheim(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)

        # Saleh Wertheim specific properties
        saleh_config = config.get('saleh')
        self._B2 = saleh_config.get('B2')
        self._B3 = saleh_config.get('B3', 0.)
        self._valence = saleh_config.get('valence')
        self._linker_half_valence = self._valence[2] / 2

        # Read in the Deltas from the config:
        self._delta_AA = Delta(saleh_config.get('delta_AA'))
        self._delta_BB = Delta(saleh_config.get('delta_BB'))

        # Scale values
        self._B2 *= (self._inverse_scaling_factor ** 3)
        self._B3 *= (self._inverse_scaling_factor ** 3)
        self._delta_AA.delta *= (self._inverse_scaling_factor ** 3)
        self._delta_BB.delta *= (self._inverse_scaling_factor ** 3)

    def N_species(self):
        # Saleh uses 3 species, please see GenericWertheim for a more generalized implementation
        return 3

