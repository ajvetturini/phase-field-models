from pfm.energy_models.free_energy_model import FreeEnergyModel
import jax.numpy as jnp
import jax
from functools import partial

class jLandau(FreeEnergyModel):

    def __init__(self, config):
        super().__init__(config)  # First init the relevent FreeEnergyModel config options
        self._epsilon = config.get('epsilon', 1.0)

        if self._user_to_internal != 1.0:
            raise Exception('Landau free energy model does not support distance_scaling_factors from 1.0')

    def N_species(self):
        # For a simple Landau model, we assume a single component, N, that can have varying concentration.
        # If you're modeling a mixture with multiple conserved quantities, you'll need to adjust this
        return 1

    @partial(jax.jit, static_argnums=(0,))
    def der_bulk_free_energy(self, species, rho_species):
        # Note: Because of jax principles, all of `self` is considered static, thus you can not use this if epsilon
        #       were to be varied during a simulation.
        op = rho_species[species]
        return -self._epsilon * op + op**3

    def bulk_free_energy(self, rho_species):
        op = rho_species[0]
        return -0.5 * self._epsilon * op**2 + 0.25 * op**4

    @partial(jax.jit, static_argnums=(0,))
    def average_free_energy(self, grid):
        """ Takes the density grid and evaluates the average free energy """
        return jnp.mean(self.bulk_free_energy([grid]))




